"""Prepare sourced input fixtures in LAB_ROOT without overwriting changed user inputs."""
from common import ROOT, REPO, sha, write_json
import json, shutil


def main():
    source = REPO / "inputs/real_photos"
    target = ROOT / "inputs/real_photos"
    target.mkdir(parents=True, exist_ok=True)
    records = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    for item in records:
        src = source / item["file"]
        dst = target / item["file"]
        if sha(src) != item["sha256"]:
            raise ValueError("Fixture source checksum mismatch: " + item["file"])
        if dst.exists() and sha(dst) != item["sha256"]:
            raise ValueError(
                "Existing user input differs; choose a new LAB_ROOT: " + str(dst)
            )
        if not dst.exists():
            shutil.copy2(src, dst)
    write_json(target / "manifest.json", records)
    print("READY", len(records), "sourced real-photo fixtures")


if __name__ == "__main__":
    main()
