"""Serial GPU queue: preserve complete runs, wait for precise prerequisites, stop on failures."""
import os, subprocess, sys, time, json, hashlib
from pathlib import Path

ROOT = Path(os.environ["LAB_ROOT"])
HERE = Path(__file__).parent
PLAN = [
    ("esd.py", "06_esd", ["sd14"], 24),
    ("generation.py", "02_generation", ["sd15"], 12),
    ("fairness.py", "07_fairness", ["sd15", "clip"], 400),
    ("sld.py", "06_sld", ["sld"], 18),
    ("detection.py", "04_detection", [], None),
    ("watermark.py", "05_watermark", [], 216),
    ("filters.py", "03_filters", ["text_filter", "image_filter"], None),
    ("pseudoword.py", "03_pseudoword", ["clip"], 7),
    ("privacy.py", "08_privacy", ["clip", "sd15", "ip_adapter"], None),
]
for script, chapter, keys, count in PLAN:
    folder = ROOT / "results" / chapter
    record = folder / "run.json"
    samples = folder / "samples.json"
    if record.exists():
        old = json.loads(record.read_text())
        same = (
            old.get("scripts", {}).get(script)
            == hashlib.sha256((HERE / script).read_bytes()).hexdigest()
        )
        complete = (
            old["status"] == "complete"
            and same
            and samples.exists()
            and (count is None or len(json.loads(samples.read_text())) == count)
        )
        if complete:
            print("PRESERVE_COMPLETE", script, flush=True)
            continue
    while True:
        manifest = ROOT / "manifests/models.json"
        try:
            data = json.loads(manifest.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        sizes = {
            "esd.py": ("diffusers-VanGogh-ESDx1-UNET.pt", 3438307109),
            "detection.py": ("blur_jpg_prob0.5.pth", 282442597),
        }
        extra = True
        if script in sizes:
            name, n = sizes[script]
            p = ROOT / "models" / name
            extra = p.is_file() and p.stat().st_size == n
        if all(k in data for k in keys) and extra:
            break
        print("WAIT", script, keys, "weights_complete", extra, flush=True)
        time.sleep(30)
    print("START", script, flush=True)
    with (ROOT / "logs" / script.replace(".py", ".log")).open("a") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(HERE / script)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print("FINISH", script, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
for script in ["measure_clip.py", "analyse.py", "validate_evidence.py"]:
    with (ROOT / "logs" / script.replace(".py", ".log")).open("a") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(HERE / script)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise SystemExit(result.returncode)
print("ALL_EXPERIMENTS_VALIDATED", flush=True)
