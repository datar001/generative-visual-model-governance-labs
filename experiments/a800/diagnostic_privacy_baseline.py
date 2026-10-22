"""Prespecified privacy baseline diagnostic: framing and adapter strength, not protection optimization."""
from common import *
from PIL import Image
import torch
from diffusers import DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from facenet_pytorch import MTCNN
from make_figures import grid


def main():
    out = ROOT / "diagnostics/privacy_baseline"
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(ROOT / "results/08_privacy/face_2026/original.png").convert(
        "RGB"
    )
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        model("ip_adapter"),
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = load_sd()
    pipe.image_encoder = encoder.to("cuda")
    pipe.load_ip_adapter(
        model("ip_adapter"),
        subfolder="models",
        weight_name="ip-adapter-plus-face_sd15.bin",
        image_encoder_folder=None,
        local_files_only=True,
    )
    detector = MTCNN(keep_all=True, device="cuda")
    rows = []
    cells = []
    prompt = "a close-up frontal headshot photograph of an adult woman, neutral expression, face centered and large in the frame, fully clothed, soft daylight, blurred garden background"
    negative = (
        "low quality, blurry, distorted face, deformed eyes, cartoon, illustration"
    )
    original_scheduler = pipe.scheduler
    for scheduler in ["pndm", "ddim"]:
        pipe.scheduler = (
            original_scheduler
            if scheduler == "pndm"
            else DDIMScheduler.from_config(
                original_scheduler.config, clip_sample=False, set_alpha_to_one=False
            )
        )
        for strength in [0.3, 0.5, 0.7]:
            pipe.set_ip_adapter_scale(strength)
            result = pipe(
                prompt,
                negative_prompt=negative,
                ip_adapter_image=image,
                num_inference_steps=30,
                height=512,
                width=512,
                guidance_scale=5.0,
                generator=torch.Generator("cpu").manual_seed(2026),
            )
            generated = result.images[0]
            flag = normalize_output_flag(result.nsfw_content_detected)
            name = f"{scheduler}_{strength}.png"
            generated.save(out / name)
            boxes, probs = detector.detect(generated)
            faces = 0 if boxes is None else sum(p >= 0.95 for p in probs)
            rows.append(
                dict(
                    scheduler=scheduler,
                    strength=strength,
                    output_flag=flag,
                    faces=int(faces),
                    prompt=prompt,
                    negative_prompt=negative,
                    cfg=5.0,
                    image=name,
                )
            )
            cells.append(
                (
                    out / name,
                    f"{scheduler} | scale {strength}",
                    f"faces {faces} | flag {flag}",
                )
            )
    write_json(out / "samples.json", rows)
    grid(cells, 3, out / "comparison.png", tile=300)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
