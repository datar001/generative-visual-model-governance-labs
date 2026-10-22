"""Resumable GET downloader with content validation; no dependence on HEAD metadata."""
import os, json, time, hashlib, shutil
from pathlib import Path
import requests
from huggingface_hub import HfApi

ROOT = Path(os.environ["LAB_ROOT"])
SPECS = [
    (
        "text_filter",
        "michelleli99/NSFW_text_classifier",
        "652bad24f49ec5b7dd218f5d273c7bbe815242ee",
        None,
    ),
    (
        "image_filter",
        "Falconsai/nsfw_image_detection",
        "96cb0d0342c7afb80cab76ecc58b265fa44da256",
        None,
    ),
    (
        "ip_adapter",
        "h94/IP-Adapter",
        "018e402774aeeddd60609b4ecdb7e298259dc729",
        [
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
            "models/ip-adapter-plus-face_sd15.bin",
        ],
    ),
]


def digest(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


for key, repo, revision, selected in SPECS:
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    revision = info.sha
    files = info.siblings
    if selected is None:
        selected = [
            f.rfilename
            for f in files
            if f.rfilename.endswith((".json", ".txt", ".safetensors", ".bin"))
            and not any(
                x in f.rfilename for x in ["training_args", "optimizer", "scheduler.pt"]
            )
        ]
        if any(n.endswith(".safetensors") for n in selected):
            selected = [n for n in selected if not n.endswith(".bin")]
    target = ROOT / "models" / key
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for name in selected:
        spec = next(f for f in files if f.rfilename == name)
        expected = spec.lfs.sha256 if spec.lfs else spec.blob_id
        size = spec.lfs.size if spec.lfs else spec.size
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        partial = dst.with_suffix(dst.suffix + ".download")
        if dst.exists() and digest(dst, not bool(spec.lfs)) == expected:
            records.append(
                dict(file=name, bytes=dst.stat().st_size, sha256=digest(dst))
            )
            continue
        cached = list(
            (target / ".cache/huggingface/download").rglob(
                "*" + expected + ".incomplete"
            )
        )
        if cached and not partial.exists():
            shutil.move(str(cached[0]), str(partial))
        for attempt in range(1, 101):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if (
                    size is not None
                    and offset == size
                    and digest(partial, not bool(spec.lfs)) == expected
                ):
                    partial.replace(dst)
                    break
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(
                    f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                    headers=headers,
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        ):
                            raise ValueError("Invalid range response")
                        mode = "ab"
                    else:
                        mode = "wb"
                    print("GET", key, name, offset, attempt, flush=True)
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                if digest(partial, not bool(spec.lfs)) != expected:
                    raise ValueError("Content checksum mismatch")
                partial.replace(dst)
                break
            except Exception as exc:
                print("RETRY", key, name, type(exc).__name__, flush=True)
                time.sleep(3)
        else:
            raise RuntimeError("Download retries exhausted: " + name)
        records.append(dict(file=name, bytes=dst.stat().st_size, sha256=digest(dst)))
    manifest = ROOT / "manifests/models.json"
    data = json.loads(manifest.read_text())
    data[key] = dict(repo=repo, revision=revision, path=str(target), files=records)
    temp = manifest.with_suffix(".get.tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(manifest)
    print("READY", key, flush=True)
