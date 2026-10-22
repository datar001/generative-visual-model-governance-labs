"""Numerical summaries from actual samples; no expected values or prefilled experiment outcomes."""
from common import *
import math, collections, csv
import numpy as np


def load(ch):
    p = ROOT / "results" / ch / "samples.json"
    return json.loads(p.read_text()) if p.exists() else []


def csv_export(ch, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT / "results" / ch / "samples.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in r.items()
                }
            )


def meanstd(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return dict(
        n=len(values),
        mean=float(np.mean(values)) if values else None,
        std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        min=min(values) if values else None,
        max=max(values) if values else None,
    )


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def main():
    for chapter in [p.name for p in (ROOT / "results").iterdir() if p.is_dir()]:
        csv_export(chapter, load(chapter))
    results = {}
    rows = load("04_detection")
    groups = []
    for condition in ["original", "100", "75", "40"]:
        for source in ["real", "fake"]:
            subset = [
                r for r in rows if r["source"] == source and r["condition"] == condition
            ]
            groups.append(
                dict(
                    condition=condition,
                    source=source,
                    n=len(subset),
                    correct=sum(r["prediction"] == source for r in subset),
                    errors=sum(r["prediction"] != source for r in subset),
                    fake_score=meanstd([r["fake_score"] for r in subset]),
                )
            )
    results["detection"] = groups
    rows = load("05_watermark")
    groups = []
    for strength in [8, 16, 24]:
        for condition in [
            "png",
            "jpeg100",
            "jpeg75",
            "jpeg40",
            "resize256",
            "crop10",
            "noise2",
            "unmarked",
        ]:
            subset = [
                r
                for r in rows
                if r["strength"] == strength and r["condition"] == condition
            ]
            groups.append(
                dict(
                    strength=strength,
                    condition=condition,
                    images=len(subset),
                    bit_errors=sum(r["bit_errors"] for r in subset),
                    total_bits=sum(r["total_bits"] for r in subset),
                    bit_accuracy=meanstd([r["bit_accuracy"] for r in subset]),
                    psnr=meanstd([r["psnr_db"] for r in subset]),
                    exact_match=sum(r["bit_errors"] == 0 for r in subset),
                )
            )
    results["watermark"] = groups
    rows = load("03_filters")
    groups = []
    for group in ["safe", "risk", "safe_context"]:
        subset = [r for r in rows if r["kind"] == "text" and r["group"] == group]
        groups.append(
            dict(
                group=group,
                n=len(subset),
                keyword_blocked=sum(
                    r["keyword"]["status"] == "blocked" for r in subset
                ),
                text_blocked=sum(r["text"]["status"] == "blocked" for r in subset),
            )
        )
    images = [r for r in rows if r["kind"] == "image"]
    results["filters"] = dict(
        text=groups,
        images=len(images),
        image_blocked=sum(r["image"]["status"] == "blocked" for r in images),
        image_risk_recall=None,
    )
    rows = load("07_fairness")
    groups = []
    for job in sorted({r["job"] for r in rows}):
        for mode in ["neutral", "female"]:
            subset = [r for r in rows if r["job"] == job and r["mode"] == mode]
            counts = collections.Counter(r["automatic_label"] for r in subset)
            decidable = counts["feminine"] + counts["masculine"]
            groups.append(
                dict(
                    job=job,
                    mode=mode,
                    total=len(subset),
                    feminine=counts["feminine"],
                    masculine=counts["masculine"],
                    uncertain=counts["uncertain"],
                    review=counts["review"],
                    decidable=decidable,
                    feminine_fraction_decidable=counts["feminine"] / decidable
                    if decidable
                    else None,
                    wilson95_decidable=wilson(counts["feminine"], decidable),
                    unreviewed=sum(
                        r.get("reviewed_label", "unreviewed") == "unreviewed"
                        for r in subset
                    ),
                )
            )
    results["fairness"] = groups
    rows = load("08_privacy")
    groups = []
    for stage in ["feature", "adapter_input_feature", "customization"]:
        for variant in ["original", "protected", "protected_q75"]:
            subset = [
                r for r in rows if r["stage"] == stage and r["variant"] == variant
            ]
            groups.append(
                dict(
                    stage=stage,
                    variant=variant,
                    total=len(subset),
                    scorable=sum(
                        r.get("cosine_to_original") is not None for r in subset
                    ),
                    cosine=meanstd([r.get("cosine_to_original") for r in subset]),
                    psnr=meanstd([r.get("psnr_db") for r in subset]),
                    linf=meanstd([r.get("linf") for r in subset]),
                )
            )
    results["privacy"] = groups
    write_json(ROOT / "results/analysis_summary.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
