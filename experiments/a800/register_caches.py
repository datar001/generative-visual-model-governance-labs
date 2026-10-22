"""Register immutable read-only caches; independent checker source is disclosed explicitly."""
import json, os, hashlib
from pathlib import Path

ROOT = Path(os.environ["LAB_ROOT"])
hub = Path("/home/zcy/.cache/huggingface/hub")


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


manifest = ROOT / "manifests/models.json"
data = json.loads(manifest.read_text()) if manifest.exists() else {}
clip = (
    hub
    / "models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)
assert (clip / "pytorch_model.bin").is_file()
data["clip"] = dict(
    repo="openai/clip-vit-base-patch32",
    revision=clip.name,
    path=str(clip),
    cache="read-only immutable snapshot",
)
base = (
    hub
    / "models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
)
checker = (
    hub
    / "models--CompVis--stable-diffusion-v1-4/snapshots/2880f2ca379f41b0226444936bb7a6766a227587/safety_checker"
)
assert (checker / "pytorch_model.bin").is_file()
target = ROOT / "models/sd14"
target.mkdir(parents=True, exist_ok=True)
for src in base.rglob("*"):
    if src.is_file():
        dst = target / src.relative_to(base)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve())
assert not (target / "safety_checker").exists()
(target / "safety_checker").symlink_to(checker, target_is_directory=True)
data["sd14"] = dict(
    repo="CompVis/stable-diffusion-v1-4",
    revision=base.name,
    path=str(target),
    safety_checker_revision=checker.parent.name,
    safety_checker_sha256=digest(checker / "pytorch_model.bin"),
    note="Base model uses pinned full-precision official snapshot; independent safety checker is loaded from its complete older official fp16 snapshot. Both retained at inference.",
)
manifest.write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2))
