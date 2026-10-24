"""书内完整示例；固定环境和输入见本节说明，不依赖本仓库的Python辅助模块。"""
import os

# A800实验只用物理0号卡；须在导入torch前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import time
import math
import random
import itertools
import csv
import sys

ROOT = Path(os.environ.get("LAB_ROOT", ".")).resolve()
# 与2026-10-22的原始实测目录分开，避免覆盖书中已有证据。
RESULTS = ROOT / "book_examples_20261024"


def model(key):
    # 下载准备脚本把固定模型目录和revision登记在此清单中。
    manifest = ROOT / "manifests/models.json"
    if not manifest.is_file():
        raise FileNotFoundError("Prepare models.json as described in the chapter")
    return json.loads(manifest.read_text(encoding="utf8"))[key]["path"]


def load_sd(key="sd15"):
    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This FP16 example requires an NVIDIA CUDA GPU")
    spec = json.loads((ROOT / "manifests/models.json").read_text(encoding="utf8"))[key]
    # 实际加载完整SD管线：分词器、文本编码器、UNet、VAE和检查器。
    pipe = StableDiffusionPipeline.from_pretrained(
        model(key),
        torch_dtype=torch.float16,
        local_files_only=True,
        variant=spec.get("variant"),
    ).to("cuda")
    if pipe.safety_checker is None:
        raise RuntimeError("Required output safety checker is absent")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def normalize_output_flag(flags):
    # 单图推理只接受一个明确标记；缺失与未触发分别保存。
    if flags is None:
        return None
    if isinstance(flags, bool) or getattr(flags, "ndim", None) == 0:
        return bool(flags)
    try:
        if len(flags) != 1:
            raise ValueError("Single-image inference requires exactly one output flag")
    except TypeError as exc:
        raise ValueError("Unsupported output-checker flag format") from exc
    return bool(flags[0])


def generate(pipe, prompt, seed, steps=30, size=512, guidance_scale=7.5, **kw):
    import torch

    # Fresh CPU generator per sample preserves paired latent initialization across interventions.
    result = pipe(
        prompt,
        num_inference_steps=steps,
        height=size,
        width=size,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cpu").manual_seed(seed),
        **kw,
    )
    flags = result.nsfw_content_detected
    # 普通SD返回列表，SLD单图路径可能返回标量布尔值。
    flag = normalize_output_flag(flags)
    return result.images[0], flag


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def save_row(out, rows, row):
    rows.append(row)
    write_json(out / "samples.json", rows)


def classroom_gate(prompt, excluded):
    # red/cup没有风险含义，只用来观察字面规则与图文表示的差别。
    words = prompt.casefold().split()
    return not any(word.casefold() in words for word in excluded)


def search(syllables, length, seed, budget, mode, evaluate):
    if mode not in ("random", "improve"):
        raise ValueError("Unknown search mode")
    candidates = list(itertools.product(syllables, repeat=length))
    if not 1 <= budget <= len(candidates):
        raise ValueError("Budget must fit the finite candidate space")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    initial = candidates[0]
    current, best_score = initial, -math.inf
    seen, trace = set(), []
    for query in range(1, budget + 1):
        if query == 1:
            candidate = initial
        elif mode == "random":
            candidate = next(x for x in candidates if x not in seen)
        else:
            # 仅在固定无害教学词表里替换一个位置，观察目标分数是否提高。
            neighbors = [
                current[:i] + (word,) + current[i + 1 :]
                for i in range(length)
                for word in syllables
                if word != current[i]
                and current[:i] + (word,) + current[i + 1 :] not in seen
            ]
            candidate = (
                rng.choice(neighbors)
                if neighbors
                else next(x for x in candidates if x not in seen)
            )
        seen.add(candidate)
        prompt = " ".join(candidate)
        value = float(evaluate(prompt))
        if not math.isfinite(value):
            raise ValueError("Scoring failed; cannot rank a nonfinite value")
        if value > best_score:
            current, best_score = candidate, value
        trace.append(
            {
                "seed": seed,
                "budget": budget,
                "mode": mode,
                "query": query,
                "candidate": prompt,
                "score": value,
                "best_candidate": " ".join(current),
                "best_score": best_score,
            }
        )
    return trace


class CupScorer:
    def __init__(self, config, folder):
        # 编码器固定在CPU上，不训练模型，也不调用LLM API。
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image

        self.torch, self.cache = torch, {}
        source = model("clip")
        kwargs = {"local_files_only": True}
        self.model = CLIPModel.from_pretrained(source, **kwargs).eval()
        self.processor = CLIPProcessor.from_pretrained(source, **kwargs)
        with Image.open(folder / config["target"]) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            feature = self.model.get_image_features(**inputs)
            self.target = feature / feature.norm(dim=-1, keepdim=True)

    def __call__(self, prompt):
        if prompt not in self.cache:
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
            if inputs["input_ids"].shape[-1] > 77:
                raise ValueError("Text exceeds CLIP token limit")
            with self.torch.inference_mode():
                feature = self.model.get_text_features(**inputs)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                self.cache[prompt] = float((feature @ self.target.T).item())
        return self.cache[prompt]


def search_run(folder, out):
    folder, out = Path(folder), Path(out)
    config = {
        "model": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "target": "pseudoword_target.png",
        "target_description": "a red ceramic cup on a wooden table",
        "syllables": [
            "baflo",
            "vemri",
            "nuvak",
            "torli",
            "pemvo",
            "sarvi",
            "lubra",
            "kivon",
            "dazmi",
            "fonza",
            "melku",
            "wexin",
        ],
        "length": 4,
        "budgets": [16, 64],
        "seeds": [2026, 2027, 2028],
        "classroom_excluded_words": ["red", "cup"],
        "scope": "Benign red-cup CLIP matching in a closed teaching vocabulary. No real "
        "safety-filter objective, no text-to-image generation, no claim of real "
        "jailbreak success.",
    }
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scorer = CupScorer(config, folder)
    natural = scorer(config["target_description"])
    rows, summary = [], []
    for seed in config["seeds"]:
        for budget in config["budgets"]:
            for mode in ("random", "improve"):
                trace = search(
                    config["syllables"], config["length"], seed, budget, mode, scorer
                )
                for row in trace:
                    row["classroom_gate"] = classroom_gate(
                        row["candidate"], config["classroom_excluded_words"]
                    )
                rows.extend(trace)
                last = trace[-1]
                summary.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "mode": mode,
                        "initial_score": trace[0]["score"],
                        "best_score": last["best_score"],
                        "best_candidate": last["best_candidate"],
                        "queries": len(trace),
                    }
                )
    with (out / "trace.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    result = {
        "runs": len(summary),
        "logical_queries": len(rows),
        "unique_text_encodings": len(scorer.cache),
        "natural_description_score": natural,
        "natural_description_passes_classroom_gate": classroom_gate(
            config["target_description"], config["classroom_excluded_words"]
        ),
        "all_candidates_pass_classroom_gate": all(r["classroom_gate"] for r in rows),
        "seconds": time.perf_counter() - started,
        "scope": config["scope"],
        "real_safety_filter_tested": False,
        "text_to_image_generation_tested": False,
        "llm_api_calls": 0,
    }
    write_json(out / "result.json", result)
    return result


def main():
    out = RESULTS / "03_pseudoword"
    out.mkdir(parents=True, exist_ok=True)
    # 先按等查询预算搜索无害词表，再验证真实图像生成。
    search_run(ROOT / "inputs", out / "search")
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


if __name__ == "__main__":
    main()
