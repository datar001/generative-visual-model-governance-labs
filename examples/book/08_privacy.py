"""书内完整示例；固定环境和输入见本节说明，不依赖本仓库的Python辅助模块。"""
import os

# A800实验只用物理0号卡；须在导入torch前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection

ROOT = Path(os.environ.get("LAB_ROOT", ".")).resolve()
# 与2026-10-22的原始实测目录分开，避免覆盖书中已有证据。
RESULTS = ROOT / "book_examples_20261024"


def model(key):
    # 下载准备脚本把固定模型目录和revision登记在此清单中。
    manifest = ROOT / "manifests/models.json"
    if not manifest.is_file():
        raise FileNotFoundError("Prepare models.json as described in the chapter")
    return json.loads(manifest.read_text(encoding="utf8"))[key]["path"]


def load_sd(key="sd15"):
    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This FP16 example requires an NVIDIA CUDA GPU")
    spec = json.loads((ROOT / "manifests/models.json").read_text(encoding="utf8"))[key]
    # 实际加载完整SD管线：分词器、文本编码器、UNet、VAE和检查器。
    pipe = StableDiffusionPipeline.from_pretrained(
        model(key),
        torch_dtype=torch.float16,
        local_files_only=True,
        variant=spec.get("variant"),
    ).to("cuda")
    if pipe.safety_checker is None:
        raise RuntimeError("Required output safety checker is absent")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def normalize_output_flag(flags):
    # 单图推理只接受一个明确标记；缺失与未触发分别保存。
    if flags is None:
        return None
    if isinstance(flags, bool) or getattr(flags, "ndim", None) == 0:
        return bool(flags)
    try:
        if len(flags) != 1:
            raise ValueError("Single-image inference requires exactly one output flag")
    except TypeError as exc:
        raise ValueError("Unsupported output-checker flag format") from exc
    return bool(flags[0])


def generate(pipe, prompt, seed, steps=30, size=512, guidance_scale=7.5, **kw):
    import torch

    # Fresh CPU generator per sample preserves paired latent initialization across interventions.
    result = pipe(
        prompt,
        num_inference_steps=steps,
        height=size,
        width=size,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cpu").manual_seed(seed),
        **kw,
    )
    flags = result.nsfw_content_detected
    # 普通SD返回列表，SLD单图路径可能返回标量布尔值。
    flag = normalize_output_flag(flags)
    return result.images[0], flag


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def save_row(out, rows, row):
    rows.append(row)
    write_json(out / "samples.json", rows)


def cosine(a, b):
    return float(F.cosine_similarity(a, b).item())


def main():
    out = RESULTS / "08_privacy"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
    net = (
        InceptionResnetV1(pretrained="vggface2").eval().to("cuda").requires_grad_(False)
    )
    detector = MTCNN(keep_all=True, device="cuda")

    def encode(x):
        return net((x * 255 - 127.5) / 128)

    source = [
        r
        for r in json.loads((RESULTS / "02_generation/samples.json").read_text())
        if r["kind"] == "face" and r["output_flag"] is False
    ]
    rows = []
    references = {}
    variants = []
    for item in source:
        full = Image.open(RESULTS / "02_generation" / item["image"]).convert("RGB")
        boxes, probs = detector.detect(full)
        valid = [] if boxes is None else [b for b, p in zip(boxes, probs) if p >= 0.95]
        if len(valid) != 1:
            save_row(
                out,
                rows,
                dict(
                    id=item["id"],
                    stage="input_validation",
                    status="unsuitable",
                    detected_faces=len(valid),
                ),
            )
            continue
        b = valid[0]
        margin = 0.15 * max(b[2] - b[0], b[3] - b[1])
        box = (
            max(0, int(b[0] - margin)),
            max(0, int(b[1] - margin)),
            min(full.width, int(b[2] + margin)),
            min(full.height, int(b[3] + margin)),
        )
        original = full.crop(box).resize((160, 160), Image.Resampling.BILINEAR)
        folder = out / item["id"]
        folder.mkdir(exist_ok=True)
        original.save(folder / "original.png")
        x = to_tensor(original).unsqueeze(0).to("cuda")
        with torch.no_grad():
            ref = encode(x)
        references[item["id"]] = ref.detach()
        torch.manual_seed(2026)
        epsilon = 8 / 255
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        trajectory = []
        # 沿梯度降低当前FaceNet相似度，每步投影回像素预算。
        for step in range(40):
            delta = delta.detach().requires_grad_(True)
            loss = F.cosine_similarity(encode((x + delta).clamp(0, 1)), ref).mean()
            (grad,) = torch.autograd.grad(loss, delta)
            delta = (delta - (1 / 255) * grad.sign()).clamp(-epsilon, epsilon)
            delta = (x + delta).clamp(0, 1) - x
            trajectory.append(dict(step=step + 1, cosine_before_update=float(loss)))
        pixels = (x + delta).detach().cpu()[0].permute(1, 2, 0).numpy() * 255
        original_bytes = np.asarray(original).astype("int16")
        pixels = np.clip(
            np.rint(pixels),
            np.maximum(original_bytes - 8, 0),
            np.minimum(original_bytes + 8, 255),
        ).astype("uint8")
        protected = Image.fromarray(pixels)
        protected.save(folder / "protected.png")
        protected.save(folder / "protected_q75.jpg", quality=75)
        write_json(folder / "trajectory.json", trajectory)
        for variant in ["original", "protected", "protected_q75"]:
            name = variant + (".jpg" if variant.endswith("q75") else ".png")
            image = Image.open(folder / name).convert("RGB")
            t = to_tensor(image).unsqueeze(0).to("cuda")
            with torch.no_grad():
                sim = cosine(encode(t), ref)
            mse = float(torch.mean((x - t) ** 2))
            psnr = None if mse == 0 else float(-10 * np.log10(mse))
            save_row(
                out,
                rows,
                dict(
                    id=item["id"],
                    stage="feature",
                    variant=variant,
                    image=str((folder / name).relative_to(out)),
                    cosine_to_original=sim,
                    psnr_db=psnr,
                    linf=float((x - t).abs().max()),
                    face_box=box,
                    epsilon=epsilon,
                    steps=40,
                    status="complete",
                ),
            )
            variants.append((item["id"], variant, folder / name))
    if len(references) < 2:
        raise RuntimeError(
            "Need at least two valid synthetic reference faces; see validation records"
        )
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    if "--features-only" in sys.argv:
        return
    # Reuse the same SD1.5 base as chapter 2; the face adapter still performs actual image-conditioned customization.
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        model("ip_adapter"),
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = load_sd()
    pipe.image_encoder = encoder.to("cuda")
    # 加载公开Plus Face适配器，固定强度后比较三类参考图。
    pipe.load_ip_adapter(
        model("ip_adapter"),
        subfolder="models",
        weight_name="ip-adapter-plus-face_sd15.bin",
        image_encoder_folder=None,
        local_files_only=True,
    )
    pipe.set_ip_adapter_scale(0.3)
    image_processor = CLIPImageProcessor()
    for ident in references:
        paths = [x for x in variants if x[0] == ident]
        feature_ref = None
        for _, variant, path in paths:
            pixels = image_processor(
                images=Image.open(path).convert("RGB"), return_tensors="pt"
            ).pixel_values.to("cuda", dtype=torch.float16)
            with torch.inference_mode():
                feature = (
                    encoder(pixels, output_hidden_states=True)
                    .hidden_states[-2]
                    .float()
                    .flatten(1)
                )
            if variant == "original":
                feature_ref = feature
            save_row(
                out,
                rows,
                dict(
                    id=ident,
                    stage="adapter_input_feature",
                    variant=variant,
                    cosine_to_original=cosine(feature, feature_ref),
                    feature="CLIP ViT-H penultimate hidden-state tokens, flattened; actual Plus Face input representation",
                ),
            )
    for ident, variant, path in variants:
        image = Image.open(path).convert("RGB")
        for seed in [2026, 2027, 2028]:
            prompt = "a close-up frontal headshot photograph of an adult woman, neutral expression, face centered and large in the frame, fully clothed, soft daylight, blurred garden background"
            negative = "low quality, blurry, distorted face, deformed eyes, cartoon, illustration"
            # 使用已确定的正常人脸构图配置，对全部参考条件配对生成。
            generated, flag = generate(
                pipe,
                prompt,
                seed,
                ip_adapter_image=image,
                negative_prompt=negative,
                guidance_scale=5.0,
            )
            filename = f"{ident}/custom_{variant}_{seed}.png"
            generated.save(out / filename)
            # Evaluation uses the same fixed recognizer; failed/multiple detections are explicit.
            mt = MTCNN(keep_all=True, device="cuda")
            boxes, probs = mt.detect(generated)
            valid = (
                [] if boxes is None else [b for b, p in zip(boxes, probs) if p >= 0.95]
            )
            del mt
            sim = None
            status = "ok" if len(valid) == 1 and flag is False else "unscorable"
            if status == "ok":
                b = valid[0]
                margin = 0.15 * max(b[2] - b[0], b[3] - b[1])
                crop = generated.crop(
                    (
                        max(0, int(b[0] - margin)),
                        max(0, int(b[1] - margin)),
                        min(512, int(b[2] + margin)),
                        min(512, int(b[3] + margin)),
                    )
                ).resize((160, 160), Image.Resampling.BILINEAR)
                with torch.no_grad():
                    sim = cosine(
                        encode(to_tensor(crop).unsqueeze(0).to("cuda")),
                        references[ident],
                    )
            save_row(
                out,
                rows,
                dict(
                    id=ident,
                    stage="customization",
                    variant=variant,
                    seed=seed,
                    image=filename,
                    output_flag=flag,
                    detected_faces=len(valid),
                    evaluation_status=status,
                    cosine_to_original=sim,
                    adapter_scale=0.3,
                    guidance_scale=5.0,
                    prompt=prompt,
                    negative_prompt=negative,
                    steps=30,
                    size=512,
                ),
            )


if __name__ == "__main__":
    main()
