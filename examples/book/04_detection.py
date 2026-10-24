"""书内完整示例；固定环境和输入见本节说明，不依赖本仓库的Python辅助模块。"""
import os

# A800实验只用物理0号卡；须在导入torch前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models

ROOT = Path(os.environ.get("LAB_ROOT", ".")).resolve()
# 与2026-10-22的原始实测目录分开，避免覆盖书中已有证据。
RESULTS = ROOT / "book_examples_20261024"


def sha(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def save_row(out, rows, row):
    rows.append(row)
    write_json(out / "samples.json", rows)


EXPECTED = "3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a"


def main():
    out = RESULTS / "04_detection"
    out.mkdir(parents=True, exist_ok=True)
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
        dict(id=r["id"], source="real", path=ROOT / "inputs/real_photos" / r["file"])
        for r in source
    ]
    generated = json.loads((RESULTS / "02_generation/samples.json").read_text())
    inputs += [
        dict(
            id=r["id"],
            source="fake",
            path=RESULTS / "02_generation" / r["image"],
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
                score = float(net(pre(test).unsqueeze(0).to(device)).sigmoid().item())
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
