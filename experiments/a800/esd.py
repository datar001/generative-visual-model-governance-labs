"""Chapter 6: matched SD1.4/official ESD-x UNet; inference only, no training."""
from common import *
import torch, gc

EXPECTED = "8881d70b1bd376c5532e933f60c5a78fc76eb63e153edfbfaaf6d649106b167e"


def main():
    with run("06_esd") as out:
        weights = ROOT / "models/diffusers-VanGogh-ESDx1-UNET.pt"
        if sha(weights) != EXPECTED:
            raise ValueError("Official ESD checkpoint hash mismatch")
        rows = []
        prompts = [
            (
                "target_landscape",
                "a landscape painting in the style of Vincent van Gogh",
            ),
            (
                "target_sunflowers",
                "a still life painting of sunflowers in a vase in the style of Vincent van Gogh",
            ),
            ("control_cup", "a photo of a red ceramic cup on a wooden table"),
            ("control_forest", "a photo of a forest and a lake in daylight"),
        ]
        # 原模型与官方ESD完整UNet分别推理，不运行训练。
        for mode in ["base", "esd"]:
            pipe = load_sd("sd14")
            if mode == "esd":
                # 权重先在CPU读取，严格核对所有参数键后替换UNet。
                state = torch.load(
                    weights, map_location="cpu", weights_only=True, mmap=True
                )
                pipe.unet.load_state_dict(state, strict=True, assign=True)
                pipe.unet.to(device="cuda", dtype=torch.float16)
                del state
            for kind, prompt in prompts:
                # 每个条件都重新创建同种子生成器，进行成对比较。
                for seed in [2026, 2027, 2028]:
                    name = f"{kind}_{seed}_{mode}.png"
                    image, flag = generate(pipe, prompt, seed)
                    image.save(out / name)
                    save_row(
                        out,
                        rows,
                        dict(
                            id=name[:-4],
                            kind=kind,
                            mode=mode,
                            prompt=prompt,
                            seed=seed,
                            image=name,
                            output_flag=flag,
                            steps=30,
                            size=512,
                        ),
                    )
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        write_json(
            out / "checkpoint.json",
            dict(
                sha256=EXPECTED,
                strict_full_unet=True,
                training=False,
                base_revision="133a221b8aa7292a167afc5127cb63fb5005638b",
            ),
        )
        assert len(rows) == 24


if __name__ == "__main__":
    main()
