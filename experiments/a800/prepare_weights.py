"""Download the two fixed author-released evaluation checkpoints; no training or weight redistribution."""
from common import ROOT, sha, write_json
import requests, time, argparse

SPECS = {
    "cnnspot": dict(
        filename="blur_jpg_prob0.5.pth",
        url="https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1",
        sha256="3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a",
    ),
    "esd": dict(
        filename="diffusers-VanGogh-ESDx1-UNET.pt",
        url="https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt",
        sha256="8881d70b1bd376c5532e933f60c5a78fc76eb63e153edfbfaaf6d649106b167e",
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("keys", nargs="*", default=list(SPECS))
    a = p.parse_args()
    for key in a.keys:
        spec = SPECS[key]
        dst = ROOT / "models" / spec["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and sha(dst) == spec["sha256"]:
            print("READY", key)
            continue
        partial = dst.with_suffix(".download")
        for attempt in range(1, 31):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                with requests.get(
                    spec["url"],
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    append = response.status_code == 206
                    if append and not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid resumed range")
                    with partial.open("ab" if append else "wb") as f:
                        for block in response.iter_content(1024 * 1024):
                            f.write(block)
                if sha(partial) != spec["sha256"]:
                    partial.replace(partial.with_suffix(".checksum_failed"))
                    raise ValueError("Checkpoint checksum mismatch")
                partial.replace(dst)
                break
            except (OSError, ValueError, requests.RequestException) as exc:
                print("RETRY", key, attempt, type(exc).__name__, flush=True)
                time.sleep(min(15, attempt))
        else:
            raise RuntimeError("Checkpoint download failed: " + key)
        write_json(
            ROOT / "manifests" / (key + "_checkpoint.json"),
            {**spec, "path": str(dst), "bytes": dst.stat().st_size},
        )
        print("READY", key)


if __name__ == "__main__":
    main()
