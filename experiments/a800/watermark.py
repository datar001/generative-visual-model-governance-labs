"""Chapter 5: DCT coefficient-difference watermark, strengths, distortions and genuine negative controls."""
from common import *
import cv2, numpy as np

BITS = np.random.default_rng(2026).integers(0, 2, 16)
COORDS = [(r, c) for r in range(0, 512, 8) for c in range(0, 512, 8)]


def embed(image, strength):
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype("float32")
    for k, (r, c) in enumerate(COORDS):
        d = cv2.dct(ycc[r : r + 8, c : c + 8, 0])
        change = (2 * BITS[k % 16] - 1) * strength - (d[2, 3] - d[3, 2])
        d[2, 3] += change / 2
        d[3, 2] -= change / 2
        ycc[r : r + 8, c : c + 8, 0] = cv2.idct(d)
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype("uint8"), cv2.COLOR_YCrCb2BGR)


def decode(image):
    lum = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    # 重复嵌入的同一位累积投票，按符号恢复消息。
    votes = np.zeros(16)
    for k, (r, c) in enumerate(COORDS):
        d = cv2.dct(lum[r : r + 8, c : c + 8].astype("float32"))
        votes[k % 16] += d[2, 3] - d[3, 2]
    return (votes > 0).astype(int)


def transform(image, kind):
    if kind.startswith("jpeg"):
        ok, b = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(kind[4:])])
        assert ok
        return cv2.imdecode(b, cv2.IMREAD_COLOR)
    if kind == "resize256":
        return cv2.resize(cv2.resize(image, (256, 256)), (512, 512))
    if kind == "crop10":
        return cv2.resize(image[26:486, 26:486], (512, 512))
    if kind == "noise2":
        return np.clip(
            image.astype("float32")
            + np.random.default_rng(2026).normal(0, 2, image.shape),
            0,
            255,
        ).astype("uint8")
    return image.copy()


def main():
    with run("05_watermark") as out:
        generated = json.loads(
            (ROOT / "results/02_generation/samples.json").read_text()
        )
        inputs = [
            r
            for r in generated
            if r["kind"] in ["red", "blue", "landscape"] and r["output_flag"] is False
        ]
        rows = []
        for item in inputs:
            image = cv2.imread(str(ROOT / "results/02_generation" / item["image"]))
            image = cv2.resize(image, (512, 512))
            for strength in [8, 16, 24]:
                marked = embed(image, strength)
                for condition in [
                    "png",
                    "jpeg100",
                    "jpeg75",
                    "jpeg40",
                    "resize256",
                    "crop10",
                    "noise2",
                    "unmarked",
                ]:
                    test = (
                        image.copy()
                        if condition == "unmarked"
                        else transform(marked, condition)
                    )
                    recovered = decode(test)
                    errors = int(np.sum(recovered != BITS))
                    name = f"{item['id']}_s{strength}_{condition}.png"
                    cv2.imwrite(str(out / name), test)
                    save_row(
                        out,
                        rows,
                        dict(
                            id=item["id"],
                            strength=strength,
                            condition=condition,
                            image=name,
                            source_bits="".join(map(str, BITS)),
                            decoded_bits="".join(map(str, recovered)),
                            bit_errors=errors,
                            total_bits=16,
                            bit_accuracy=1 - errors / 16,
                            psnr_db=None
                            if condition == "unmarked"
                            else float(cv2.PSNR(image, test)),
                        ),
                    )
        write_json(
            out / "scope.json",
            dict(
                algorithm="Educational block-DCT coefficient-difference watermark with repeated 16-bit message; not TreeRing/HiDDeN",
                independent_images=len(inputs),
                strengths=[8, 16, 24],
                conditions=8,
                expected_rows=len(inputs) * 3 * 8,
                decoding_recovers="message only, not pre-embedding image",
                negative_controls="All unmarked copies are included; repeated strengths share the same unmarked input.",
            ),
        )
        assert len(rows) == len(inputs) * 24


if __name__ == "__main__":
    main()
