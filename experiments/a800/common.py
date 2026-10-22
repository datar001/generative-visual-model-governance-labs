"""Small shared runtime: atomic records, fixed models/seeds, complete output hashes."""
import os, json, hashlib, time, platform, importlib.metadata as metadata, traceback
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(os.environ.get("LAB_ROOT", Path.cwd())).resolve()
REPO = Path(
    os.environ.get("LAB_REPOSITORY", Path(__file__).resolve().parents[2])
).resolve()
if (
    not (REPO / "chapters").exists()
    and (REPO / "practice/config_sources/chapters").exists()
):
    REPO = REPO / "practice/config_sources"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def model(key):
    return json.loads((ROOT / "manifests/models.json").read_text())[key]["path"]


def load_sd(key="sd15", safe=False):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionPipelineSafe

    spec = json.loads((ROOT / "manifests/models.json").read_text())[key]
    base = StableDiffusionPipeline.from_pretrained(
        model(key),
        torch_dtype=torch.float16,
        local_files_only=True,
        variant=spec.get("variant"),
    ).to("cuda")
    if base.safety_checker is None:
        raise RuntimeError("Required output safety checker is absent")
    pipe = base
    if safe:
        # 0.32.2的延迟导出名称不一致，直接导入并严格复制同一检查器权重。
        from diffusers.pipelines.stable_diffusion_safe.safety_checker import (
            SafeStableDiffusionSafetyChecker,
        )

        checker = SafeStableDiffusionSafetyChecker(base.safety_checker.config)
        checker.load_state_dict(base.safety_checker.state_dict(), strict=True)
        checker = checker.eval().to(device="cuda", dtype=torch.float16)
        components = dict(base.components)
        components["safety_checker"] = checker
        pipe = StableDiffusionPipelineSafe(
            **components, requires_safety_checker=True
        ).to("cuda")
        del base
    pipe.set_progress_bar_config(disable=True)
    return pipe


@contextmanager
def run(chapter):
    import torch

    out = ROOT / "results" / chapter
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "chapter": chapter,
        "status": "running",
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else 0,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {},
        "scripts": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    }
    for key in [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "numpy",
        "Pillow",
        "facenet-pytorch",
    ]:
        try:
            record["versions"][key] = metadata.version(key)
        except metadata.PackageNotFoundError:
            pass
    write_json(out / "run.json", record)
    start = time.monotonic()
    try:
        yield out
        record["status"] = "complete"
    except BaseException:
        record.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        record["outputs"] = {
            str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "run.json"
        }
        write_json(out / "run.json", record)


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
        **kw
    )
    flags = result.nsfw_content_detected
    # 普通SD返回列表，SLD单图路径可能返回标量布尔值。
    flag = normalize_output_flag(flags)
    return result.images[0], flag


def save_row(out, rows, row):
    rows.append(row)
    write_json(out / "samples.json", rows)
    print(
        json.dumps(
            {k: v for k, v in row.items() if k not in ("prompt", "scores")},
            ensure_ascii=False,
        ),
        flush=True,
    )
