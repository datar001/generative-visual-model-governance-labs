import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/02_generation'
out.mkdir(parents=True, exist_ok=True)
# 加载完整SD 1.5管线，包含文本编码器、UNet、VAE和输出检查器。
pipe = StableDiffusionPipeline.from_pretrained(
    models['sd15']['path'], torch_dtype=torch.float16,
    local_files_only=True,
).to('cuda')
prompts = {
    'red': 'a red ceramic cup on a wooden table',
    'blue': 'a blue ceramic cup on a wooden table',
    'landscape': 'a landscape photograph of a forest and a lake in daylight',
    'face': 'a studio headshot photograph of a fictional adult woman aged 35, '
            'front facing, neutral expression, fully clothed, plain background',
}
for kind, prompt in prompts.items():
    for seed in [2026, 2027, 2028]:
        # 每次重建同种子生成器，红蓝提示采用相同初始噪声。
        result = pipe(
            prompt, generator=torch.Generator('cpu').manual_seed(seed),
            num_inference_steps=30, guidance_scale=7.5, height=512, width=512,
        )
        result.images[0].save(out / f'{kind}_{seed}.png')
        print(kind, seed, 'output_checker:', result.nsfw_content_detected)
