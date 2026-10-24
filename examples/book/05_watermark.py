"""书内完整示例；固定环境和输入见本节说明，不依赖本仓库的Python辅助模块。"""
import os

# A800实验只用物理0号卡；须在导入torch前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(os.environ.get("LAB_ROOT", ".")).resolve()
# 与2026-10-22的原始实测目录分开，避免覆盖书中已有证据。
RESULTS = ROOT / "book_examples_20261024"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def save_row(out, rows, row):
    rows.append(row)
    write_json(out / "samples.json", rows)


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
    out = RESULTS / "05_watermark"
    out.mkdir(parents=True, exist_ok=True)
    generated = json.loads((RESULTS / "02_generation/samples.json").read_text())
    inputs = [
        r
        for r in generated
        if r["kind"] in ["red", "blue", "landscape"] and r["output_flag"] is False
    ]
    rows = []
    for item in inputs:
        image = cv2.imread(str(RESULTS / "02_generation" / item["image"]))
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
    assert len(rows) == len(inputs) * 24


if __name__ == "__main__":
    main()
