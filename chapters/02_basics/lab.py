from pathlib import Path; import json; import torch
from diffusers import StableDiffusionPipeline
# 1. 创建输出目录，保存生成图与实验参数。
out = Path('results/generation')
out.mkdir(parents=True, exist_ok=True)
model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
# 2. 优先使用显卡；GTX 16 系列改用单精度以兼容。
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
if device == 'cuda' and 'GTX 16' in torch.cuda.get_device_name(): dtype = torch.float32
# 加载模型；权重文件版本与计算精度分别设置。
pipe = StableDiffusionPipeline.from_pretrained(
    model, torch_dtype=dtype, variant='fp16', use_safetensors=True)
# 分片注意力和 CPU 卸载用于降低显存占用。
if device == 'cuda':
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)
# 3. 两条提示词仅改变颜色，便于观察条件控制。
prompts = ['a red ceramic cup on a wooden table',
           'a blue ceramic cup on a wooden table']
for i, prompt in enumerate(prompts):
    # 每次重置同一种子，控制随机起点。
    rng = torch.Generator(device=device).manual_seed(2026)
    # 采样 30 步；引导强度 7.5，输出为 512 乘 512 像素。
    result = pipe(prompt, generator=rng, num_inference_steps=30,
                  guidance_scale=7.5, height=512, width=512)
    result.images[0].save(out / f'cup_{i}.png')
    print(prompt, 'flag:', result.nsfw_content_detected)
# 4. 保存参数；安全检查标记不等于人工合规结论。
(out / 'settings.json').write_text(json.dumps({
    'model': model, 'seed': 2026, 'steps': 30,
    'guidance_scale': 7.5, 'device': device, 'dtype': str(dtype),
    'image_size': [512, 512], 'scheduler': type(pipe.scheduler).__name__,
    'resolved_revision': getattr(pipe.unet.config, '_commit_hash', None)}, indent=2))
