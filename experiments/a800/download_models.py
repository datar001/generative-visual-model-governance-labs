"""Download fixed public teaching models with range resume and cryptographic content checks."""
import os, json, time, hashlib, argparse
from pathlib import Path
import requests
from huggingface_hub import HfApi

ROOT = Path(os.environ.get("LAB_ROOT", Path.cwd())).resolve()
SPECS = {
    "sd15": (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        None,
    ),
    "sd14": (
        "CompVis/stable-diffusion-v1-4",
        "133a221b8aa7292a167afc5127cb63fb5005638b",
        None,
    ),
    "clip": (
        "openai/clip-vit-base-patch32",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        None,
    ),
    "text_filter": (
        "michelleli99/NSFW_text_classifier",
        "652bad24f49ec5b7dd218f5d273c7bbe815242ee",
        None,
    ),
    "image_filter": (
        "Falconsai/nsfw_image_detection",
        "96cb0d0342c7afb80cab76ecc58b265fa44da256",
        None,
    ),
    "ip_adapter": (
        "h94/IP-Adapter",
        "018e402774aeeddd60609b4ecdb7e298259dc729",
        [
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
            "models/ip-adapter-plus-face_sd15.bin",
        ],
    ),
}


def digest(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo, revision, item, target):
    name = item.rfilename
    expected = item.lfs.sha256 if item.lfs else item.blob_id
    size = item.lfs.size if item.lfs else item.size
    dst = target / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_suffix(dst.suffix + ".download")
    if dst.is_file() and digest(dst, not bool(item.lfs)) == expected:
        return dst
    for attempt in range(1, 31):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            if (
                size is not None
                and offset == size
                and digest(temp, not bool(item.lfs)) == expected
            ):
                temp.replace(dst)
                return dst
            response = requests.get(
                f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                headers={"Range": f"bytes={offset}-"} if offset else {},
                stream=True,
                timeout=(30, 120),
            )
            with response:
                response.raise_for_status()
                append = response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Unexpected range response")
                with temp.open("ab" if append else "wb") as f:
                    for block in response.iter_content(1024 * 1024):
                        f.write(block)
            if digest(temp, not bool(item.lfs)) != expected:
                # Keep a corrupt transfer for diagnosis, without treating it as a usable model.
                temp.replace(temp.with_suffix(".checksum_failed"))
                raise ValueError("Checksum mismatch")
            temp.replace(dst)
            return dst
        except (requests.RequestException, OSError, ValueError) as exc:
            print("RETRY", name, attempt, type(exc).__name__, flush=True)
            time.sleep(min(attempt, 15))
    raise RuntimeError(
        "Unable to finish " + name + "; partial download is retained for retry"
    )


def select_files(key, files, selected):
    if selected is not None:
        return selected, None
    names = [f.rfilename for f in files]
    small = [
        n
        for n in names
        if n.endswith((".json", ".txt"))
        and not any(x in n for x in ["onnx", "openvino"])
    ]
    if key not in ("sd15", "sd14"):
        weights = [n for n in names if n.endswith(".safetensors")]
        if not weights:
            weights = [
                n for n in names if n.endswith(".bin") and "training_args" not in n
            ]
        return small + weights, None
    # Only pipeline components, never redundant multi-gigabyte single-file checkpoints.
    weights = [
        n
        for n in names
        if "/" in n
        and n.endswith((".safetensors", ".bin"))
        and not any(x in n for x in ["onnx", "openvino", "non_ema"])
    ]
    selected_weights = []
    for component in sorted({str(Path(n).parent) for n in weights}):
        choices = [n for n in weights if str(Path(n).parent) == component]
        for ending in [".fp16.safetensors", ".fp16.bin", ".safetensors", ".bin"]:
            chosen = [
                n
                for n in choices
                if n.endswith(ending)
                and (".fp16." in n or not ending.startswith(".fp16"))
            ]
            if chosen:
                selected_weights.append(chosen[0])
                break
    return small + selected_weights, "fp16"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("keys", nargs="*", default=list(SPECS))
    args = p.parse_args()
    manifest = ROOT / "manifests/models.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for key in args.keys:
        if key not in SPECS:
            raise ValueError("Unknown model: " + key)
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        if key in old and Path(old[key]["path"]).is_dir():
            print("READY", key, flush=True)
            continue
        repo, revision, requested = SPECS[key]
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        revision = info.sha
        names, variant = select_files(key, info.siblings, requested)
        target = ROOT / "models" / key
        records = []
        for name in names:
            item = next(f for f in info.siblings if f.rfilename == name)
            path = fetch(repo, revision, item, target)
            records.append(
                dict(file=name, bytes=path.stat().st_size, sha256=digest(path))
            )
        data = json.loads(manifest.read_text()) if manifest.exists() else {}
        data[key] = dict(
            repo=repo,
            revision=revision,
            path=str(target),
            variant=variant,
            files=records,
        )
        tmp = manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(manifest)
        print("READY", key, flush=True)


if __name__ == "__main__":
    main()
