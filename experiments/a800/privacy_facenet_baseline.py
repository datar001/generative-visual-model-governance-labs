"""Chapter 8: bounded FaceNet perturbation and real SD1.5/IP-Adapter Plus Face transfer test."""
from common import *
import torch, gc, io, sys
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from facenet_pytorch import InceptionResnetV1, MTCNN
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


def cosine(a, b):
    return float(F.cosine_similarity(a, b).item())


def main():
    with run("08_privacy") as out:
        os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
        net = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to("cuda")
            .requires_grad_(False)
        )
        detector = MTCNN(keep_all=True, device="cuda")

        def encode(x):
            return net((x * 255 - 127.5) / 128)

        source = [
            r
            for r in json.loads(
                (ROOT / "results/02_generation/samples.json").read_text()
            )
            if r["kind"] == "face" and r["output_flag"] is False
        ]
        rows = []
        references = {}
        variants = []
        for item in source:
            full = Image.open(ROOT / "results/02_generation" / item["image"]).convert(
                "RGB"
            )
            boxes, probs = detector.detect(full)
            valid = (
                [] if boxes is None else [b for b, p in zip(boxes, probs) if p >= 0.95]
            )
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
                    []
                    if boxes is None
                    else [b for b, p in zip(boxes, probs) if p >= 0.95]
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
        write_json(
            out / "scope.json",
            dict(
                source="Three SD1.5-generated fictional adult faces; no real person used for customization",
                method="PGD against frozen FaceNet followed by actual SD1.5 IP-Adapter Plus Face inference",
                not_reproductions=["Anti-DreamBooth", "IDProtector training"],
                baseline_calibration="Six normal-input diagnostic cases; close-up framing, CFG5 and adapter scale0.3 fixed before all protected-input comparisons",
                reference_count=len(references),
                expected_customization_samples=len(variants) * 3,
                interpretation="A drop on the attacked FaceNet alone does not prove protection against the adapter; unchanged customization is a valid negative result.",
            ),
        )


if __name__ == "__main__":
    main()
