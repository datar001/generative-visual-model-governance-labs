"""书内完整示例；固定环境和输入见本节说明，不依赖本仓库的Python辅助模块。"""
import os

# A800实验只用物理0号卡；须在导入torch前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import gc

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


def main():
    out = RESULTS / "02_generation"
    out.mkdir(parents=True, exist_ok=True)
    # 加载固定SD1.5；两个颜色条件共享同一模型与采样参数。
    pipe = load_sd()
    rows = []
    prompts = [
        ("red", "a red ceramic cup on a wooden table"),
        ("blue", "a blue ceramic cup on a wooden table"),
        ("landscape", "a landscape photograph of a forest and a lake in daylight"),
        (
            "face",
            "a studio headshot photograph of a fictional adult woman aged 35, front facing, neutral expression, fully clothed, plain background",
        ),
    ]
    # 按预设提示和三个种子完整生成，不按画面效果筛选样本。
    for kind, prompt in prompts:
        for seed in [2026, 2027, 2028]:
            name = f"{kind}_{seed}.png"
            image, flag = generate(pipe, prompt, seed)
            image.save(out / name)
            save_row(
                out,
                rows,
                dict(
                    id=name[:-4],
                    kind=kind,
                    prompt=prompt,
                    seed=seed,
                    image=name,
                    output_flag=flag,
                    steps=30,
                    size=512,
                ),
            )
    assert len(rows) == 12
    del pipe
    gc.collect()


if __name__ == "__main__":
    main()
