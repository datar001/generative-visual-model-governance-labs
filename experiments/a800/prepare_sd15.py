"""Reuse verified common SD1.x components; download only the distinct official SD1.5 UNet."""
from pathlib import Path
import os, json, time, hashlib
from huggingface_hub import hf_hub_download, HfApi

ROOT = Path(os.environ["LAB_ROOT"])
base = ROOT / "models/sd14"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REV = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
# The official SD1.4 and SD1.5 repository metadata give identical .bin hashes for
# text encoder and VAE. Reusing those components does not substitute the UNet.
info = HfApi().model_info(SD15, revision=REV, files_metadata=True)
baseinfo = HfApi().model_info(
    "CompVis/stable-diffusion-v1-4",
    revision="133a221b8aa7292a167afc5127cb63fb5005638b",
    files_metadata=True,
)
lf = lambda x: {f.rfilename: f.lfs.sha256 for f in x.siblings if f.lfs}
shared = {
    n: lf(info)[n]
    for n in ["text_encoder/pytorch_model.bin", "vae/diffusion_pytorch_model.bin"]
}
assert all(lf(baseinfo)[n] == v for n, v in shared.items())
target = ROOT / "models/sd15_reused"
target.mkdir(parents=True, exist_ok=True)
for folder in [
    "feature_extractor",
    "text_encoder",
    "tokenizer",
    "vae",
    "safety_checker",
]:
    dst = target / folder
    if not dst.exists():
        dst.symlink_to(base / folder, target_is_directory=True)
for folder in ["unet", "scheduler"]:
    (target / folder).mkdir(exist_ok=True)
small = ["model_index.json", "unet/config.json", "scheduler/scheduler_config.json"]
for name in small:
    hf_hub_download(SD15, name, revision=REV, local_dir=str(target))
name = "unet/diffusion_pytorch_model.fp16.safetensors"
for attempt in range(1, 101):
    try:
        print("UNET attempt", attempt, flush=True)
        path = Path(
            hf_hub_download(
                SD15, name, revision=REV, local_dir=str(ROOT / "models/sd15_unet")
            )
        )
        break
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:180], flush=True)
        time.sleep(5)
else:
    raise RuntimeError("UNet retries exhausted")
h = hashlib.sha256(path.read_bytes()).hexdigest()
assert h == "c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21"
dst = target / "unet/diffusion_pytorch_model.safetensors"
if not dst.exists():
    dst.symlink_to(path)
manifest = ROOT / "manifests/models.json"
data = json.loads(manifest.read_text())
data["sd15"] = dict(
    repo=SD15,
    revision=REV,
    path=str(target),
    unet_source=name,
    unet_sha256=h,
    shared_component_official_bin_hashes=shared,
    safety_checker=data["sd14"]["safety_checker_sha256"],
    note="FP16 official SD1.5 UNet; common VAE/text components reused from official SD1.4 cache with identical upstream binary hashes.",
)
manifest.write_text(json.dumps(data, indent=2))
# AIML's official SLD UNet .bin is byte-identical to SD1.5's .bin.
safeinfo = HfApi().model_info("AIML-TUDA/stable-diffusion-safe", files_metadata=True)
assert (
    lf(safeinfo)["unet/diffusion_pytorch_model.bin"]
    == lf(info)["unet/diffusion_pytorch_model.bin"]
)
sld = ROOT / "models/sld_reused"
sld.mkdir(exist_ok=True)
for folder in [
    "feature_extractor",
    "text_encoder",
    "tokenizer",
    "vae",
    "safety_checker",
    "unet",
    "scheduler",
]:
    if not (sld / folder).exists():
        (sld / folder).symlink_to(target / folder, target_is_directory=True)
hf_hub_download(
    "AIML-TUDA/stable-diffusion-safe",
    "model_index.json",
    revision=safeinfo.sha,
    local_dir=str(sld),
)
data = json.loads(manifest.read_text())
data["sld"] = dict(
    repo="AIML-TUDA/stable-diffusion-safe",
    revision=safeinfo.sha,
    path=str(sld),
    unet_same_as_sd15_official_bin_sha256=lf(info)["unet/diffusion_pytorch_model.bin"],
    note="Official safe pipeline config and safety checker class; numerically common SD1.5 components reused; output checker retained.",
)
manifest.write_text(json.dumps(data, indent=2))
print("READY sd15 and sld", flush=True)
