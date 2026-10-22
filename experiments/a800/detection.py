"""Chapter 4: fixed CNNSpot threshold on sourced photographs and all chapter-2 outputs."""
from common import *
import torch, csv, io
from PIL import Image
from torchvision import models, transforms

EXPECTED = "3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a"


def main():
    with run("04_detection") as out:
        weights = ROOT / "models/blur_jpg_prob0.5.pth"
        if sha(weights) != EXPECTED:
            raise ValueError("CNNSpot weight hash mismatch")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 空间域CNN二分类：最后一层只输出一个生成类logit。
        net = models.resnet50(weights=None, num_classes=1).eval().to(device)
        net.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)["model"],
            strict=True,
        )
        pre = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        source = json.loads((ROOT / "inputs/real_photos/manifest.json").read_text())
        inputs = [
            dict(
                id=r["id"], source="real", path=ROOT / "inputs/real_photos" / r["file"]
            )
            for r in source
        ]
        generated = json.loads(
            (ROOT / "results/02_generation/samples.json").read_text()
        )
        inputs += [
            dict(
                id=r["id"],
                source="fake",
                path=ROOT / "results/02_generation" / r["image"],
            )
            for r in generated
            if r["output_flag"] is False
        ]
        rows = []
        for item in inputs:
            original = Image.open(item["path"]).convert("RGB")
            # 原始PNG与三档JPEG分别记录，固定阈值不随结果调整。
            for quality in ["original", 100, 75, 40]:
                test = original
                name = f"{item['id']}_{quality}" + (
                    ".png" if quality == "original" else ".jpg"
                )
                if quality == "original":
                    test.save(out / name)
                else:
                    original.save(out / name, quality=quality)
                    test = Image.open(out / name).convert("RGB")
                with torch.inference_mode():
                    score = float(
                        net(pre(test).unsqueeze(0).to(device)).sigmoid().item()
                    )
                row = dict(
                    id=item["id"],
                    source=item["source"],
                    source_sha256=sha(item["path"]),
                    condition=str(quality),
                    fake_score=score,
                    prediction="fake" if score >= 0.5 else "real",
                    image=name,
                )
                save_row(out, rows, row)
        write_json(
            out / "summary.json",
            dict(
                threshold=0.5,
                independent_real_sources=len(source),
                generated_outputs=len(inputs) - len(source),
                evaluations=len(rows),
                scope="Small heterogeneous teaching set, not a benchmark or independent population accuracy estimate.",
            ),
        )


if __name__ == "__main__":
    main()
