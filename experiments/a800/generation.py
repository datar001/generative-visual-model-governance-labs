"""Chapter 2: paired prompts/seeds plus synthetic adult faces for later controlled examples."""
from common import *
import gc


def main():
    with run("02_generation") as out:
        # 加载固定SD1.5；两个颜色条件共享同一模型与采样参数。
        pipe = load_sd()
        rows = []
        prompts = [
            ("red", "a red ceramic cup on a wooden table"),
            ("blue", "a blue ceramic cup on a wooden table"),
            ("landscape", "a landscape photograph of a forest and a lake in daylight"),
            (
                "face",
                "a studio headshot photograph of a fictional adult woman aged 35, front facing, neutral expression, fully clothed, plain background",
            ),
        ]
        # 按预设提示和三个种子完整生成，不按画面效果筛选样本。
        for kind, prompt in prompts:
            for seed in [2026, 2027, 2028]:
                name = f"{kind}_{seed}.png"
                image, flag = generate(pipe, prompt, seed)
                image.save(out / name)
                save_row(
                    out,
                    rows,
                    dict(
                        id=name[:-4],
                        kind=kind,
                        prompt=prompt,
                        seed=seed,
                        image=name,
                        output_flag=flag,
                        steps=30,
                        size=512,
                    ),
                )
        assert len(rows) == 12
        del pipe
        gc.collect()


if __name__ == "__main__":
    main()
