"""Actual generated image/text cosine measurements, separate from observed visual labels."""
from common import *
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main():
    net = (
        CLIPModel.from_pretrained(model("clip"), local_files_only=True)
        .to("cuda")
        .eval()
    )
    processor = CLIPProcessor.from_pretrained(model("clip"), local_files_only=True)

    def text_feature(text):
        with torch.inference_mode():
            t = net.get_text_features(
                **processor(text=[text], return_tensors="pt", padding=True).to("cuda")
            )
        return F.normalize(t.float(), dim=-1)

    def image_feature(path):
        with Image.open(path) as im:
            inputs = processor(images=im.convert("RGB"), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            t = net.get_image_features(**inputs)
        return F.normalize(t.float(), dim=-1)

    style = text_feature("a painting in the style of Vincent van Gogh")
    cup = text_feature("a red ceramic cup on a wooden table")
    for chapter in ["02_generation", "03_pseudoword", "06_esd", "06_sld"]:
        folder = ROOT / "results" / chapter
        if not (folder / "samples.json").exists():
            continue
        rows = json.loads((folder / "samples.json").read_text())
        measurements = []
        features = {}
        for row in rows:
            if row["output_flag"] is not False:
                measurements.append(
                    dict(id=row["id"], excluded="output_checker_flagged_or_unavailable")
                )
                continue
            image = image_feature(folder / row["image"])
            features[row["id"]] = image.cpu()
            measured = dict(
                id=row["id"],
                prompt_cosine=float((image @ text_feature(row["prompt"]).T).item()),
            )
            if chapter == "06_esd":
                measured["van_gogh_text_cosine"] = float((image @ style.T).item())
            if chapter == "03_pseudoword":
                measured["red_cup_text_cosine"] = float((image @ cup.T).item())
            measurements.append(measured)
        if chapter in ["06_esd", "06_sld"]:
            left, right = ("base", "esd") if chapter == "06_esd" else ("off", "strong")
            for measured in measurements:
                if measured["id"].endswith("_" + right):
                    before = measured["id"][: -len(right)] + left
                    if before in features and measured["id"] in features:
                        measured["paired_image_cosine"] = float(
                            (features[before] @ features[measured["id"]].T).item()
                        )
        write_json(
            folder / "clip_measurements.json",
            dict(
                scope="Descriptive CLIP cosine only; not calibrated quality, identity, risk or style-erasure success probability.",
                model_revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                samples=measurements,
            ),
        )


if __name__ == "__main__":
    main()
