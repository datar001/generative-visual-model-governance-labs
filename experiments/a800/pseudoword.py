"""Chapter 3: benign pseudo-word matching followed by actual safe generation verification."""
from common import *
import sys, csv, torch, gc

sys.path.insert(0, str(REPO / "chapters/03_compliance"))
from pseudoword_core import run as search_run


def main():
    with run("03_pseudoword") as out:
        os.environ["PSEUDOWORD_CLIP_PATH"] = model("clip")
        # 先按等查询预算搜索无害词表，再验证真实图像生成。
        result = search_run(REPO / "chapters/03_compliance", out / "search")
        if "--search-only" in sys.argv:
            return
        with (out / "search/summary.csv").open(encoding="utf-8-sig") as f:
            candidates = [r for r in csv.DictReader(f) if r["budget"] == "64"]
        pipe = load_sd()
        rows = []
        tests = [
            dict(mode=r["mode"], search_seed=int(r["seed"]), prompt=r["best_candidate"])
            for r in candidates
        ]
        tests.append(
            dict(
                mode="natural_reference",
                search_seed=None,
                prompt="a red ceramic cup on a wooden table",
            )
        )
        # 保留六个候选及自然语言参照的全部生成结果。
        for i, item in enumerate(tests):
            image, flag = generate(pipe, item["prompt"], 2026)
            name = f"case_{i:02d}.png"
            image.save(out / name)
            save_row(
                out,
                rows,
                dict(
                    **item,
                    id=f"case_{i:02d}",
                    image=name,
                    generation_seed=2026,
                    output_flag=flag,
                ),
            )
        write_json(
            out / "scope.json",
            dict(
                target="benign red cup",
                classroom_excluded_words=["red", "cup"],
                real_safety_bypass_tested=False,
                generated_cases=len(rows),
                llm_api_calls=0,
                meaning="Tests whether a high surrogate score carries to actual image generation; failure is retained.",
            ),
        )


if __name__ == "__main__":
    main()
