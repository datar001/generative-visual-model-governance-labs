"""Register public SD1.5 for the official Diffusers SLD pipeline, with the output checker retained."""
from pathlib import Path
import os, json, time

ROOT = Path(os.environ["LAB_ROOT"])
manifest = ROOT / "manifests/models.json"
while True:
    data = json.loads(manifest.read_text())
    if "sd15" in data:
        break
    time.sleep(20)
spec = dict(data["sd15"])
spec.update(
    method="Diffusers 0.32.2 StableDiffusionPipelineSafe / SLD",
    checker="Directly imported SafeStableDiffusionSafetyChecker with strictly copied original output-checker state",
    safety_checker_retained=True,
    checkpoint_note="Public SD1.5 base; public SLD implementation. No restricted checkpoint required.",
)
data["sld"] = spec
temp = manifest.with_suffix(".sld.tmp")
temp.write_text(json.dumps(data, indent=2))
temp.replace(manifest)
print("READY public SD1.5 + direct-constructor official SLD", flush=True)
