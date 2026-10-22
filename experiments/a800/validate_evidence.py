"""Verify actual complete evidence: hashes, denominators, pairs, saved perturbation bounds."""
from common import *
import argparse, math, collections

EXPECTED = {
    "01_governance": None,
    "02_generation": 12,
    "03_filters": None,
    "03_pseudoword": 7,
    "04_detection": None,
    "05_watermark": 216,
    "06_sld": 18,
    "06_esd": 24,
    "07_fairness": 400,
    "08_privacy": None,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    checks = []

    def check(name, ok, detail=None):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))

    for chapter, count in EXPECTED.items():
        folder = ROOT / "results" / chapter
        if not (folder / "run.json").exists():
            check(chapter + ".complete", False, "run.json missing")
            continue
        record = json.loads((folder / "run.json").read_text())
        check(chapter + ".complete", record["status"] == "complete", record["status"])
        missing = []
        mismatch = []
        for name, h in record.get("outputs", {}).items():
            path = folder / name
            if not path.is_file():
                missing.append(name)
            elif sha(path) != h:
                mismatch.append(name)
        check(
            chapter + ".recorded_hashes",
            not missing and not mismatch,
            dict(missing=missing, mismatch=mismatch),
        )
        sample = folder / "samples.json"
        rows = json.loads(sample.read_text()) if sample.exists() else []
        if count is not None:
            check(
                chapter + ".sample_count",
                len(rows) == count,
                dict(expected=count, actual=len(rows)),
            )
        if chapter in [
            "02_generation",
            "03_pseudoword",
            "06_sld",
            "06_esd",
            "07_fairness",
        ]:
            check(chapter + ".unique_ids", len({r["id"] for r in rows}) == len(rows))
            check(
                chapter + ".image_files",
                all((folder / r["image"]).is_file() for r in rows),
            )
            check(
                chapter + ".explicit_output_flags",
                all(
                    r["output_flag"] in [True, False, None] and "output_flag" in r
                    for r in rows
                ),
            )
        if chapter == "07_fairness":
            groups = collections.Counter((r["job"], r["mode"]) for r in rows)
            check(
                chapter + ".full_groups",
                len(groups) == 20 and set(groups.values()) == {20},
                dict(groups=len(groups)),
            )
            check(
                chapter + ".classifiers_complete",
                all(
                    r.get("automatic_label")
                    in ["feminine", "masculine", "uncertain", "review"]
                    for r in rows
                ),
            )
            check(
                chapter + ".review_status_separate",
                all("reviewed_label" in r for r in rows),
            )
        if chapter == "05_watermark":
            check(
                chapter + ".decoded_bit_records",
                all(
                    len(r["source_bits"]) == 16
                    and len(r["decoded_bits"]) == 16
                    and sum(a != b for a, b in zip(r["source_bits"], r["decoded_bits"]))
                    == r["bit_errors"]
                    for r in rows
                ),
            )
            check(
                chapter + ".negative_controls",
                sum(r["condition"] == "unmarked" for r in rows) == 27,
            )
        if chapter in ["06_esd", "06_sld"]:
            mode_pair = {"base", "esd"} if chapter == "06_esd" else {"off", "strong"}
            pairs = {}
            for row in rows:
                pairs.setdefault((row["kind"], row["seed"]), set()).add(row["mode"])
            check(
                chapter + ".complete_pairs", all(x == mode_pair for x in pairs.values())
            )
        if chapter == "08_privacy":
            feature = [r for r in rows if r["stage"] == "feature"]
            custom = [r for r in rows if r["stage"] == "customization"]
            references = {r["id"] for r in feature}
            check(chapter + ".references", len(references) == 3, len(references))
            check(chapter + ".feature_variants", len(feature) == 9, len(feature))
            check(chapter + ".customization_samples", len(custom) == 27, len(custom))
            baseline = [r for r in custom if r["variant"] == "original"]
            scorable = sum(r.get("evaluation_status") == "ok" for r in baseline)
            check(
                chapter + ".baseline_evaluable",
                scorable >= 8,
                dict(scorable=scorable, total=len(baseline)),
            )
            check(
                chapter + ".serialized_linf_bound",
                all(
                    r["linf"] <= 8 / 255 + 1e-6
                    for r in feature
                    if r["variant"] == "protected"
                ),
            )
            check(
                chapter + ".saved_custom_images",
                all((folder / r["image"]).is_file() for r in custom),
            )
    write_json(
        ROOT / "results/evidence_validation.json",
        dict(all_passed=all(c["passed"] for c in checks), checks=checks),
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(c["passed"] for c in checks) and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
