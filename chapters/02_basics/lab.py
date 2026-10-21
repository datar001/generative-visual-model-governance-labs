from pathlib import Path; import json; import torch
from diffusers import StableDiffusionPipeline
out = Path('results/generation')
out.mkdir(parents=True, exist_ok=True)
model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
if device == 'cuda' and 'GTX 16' in torch.cuda.get_device_name(): dtype = torch.float32
pipe = StableDiffusionPipeline.from_pretrained(
    model, torch_dtype=dtype, variant='fp16', use_safetensors=True)
if device == 'cuda':
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)
prompts = ['a red ceramic cup on a wooden table',
           'a blue ceramic cup on a wooden table']
for i, prompt in enumerate(prompts):
    rng = torch.Generator(device=device).manual_seed(2026)
    result = pipe(prompt, generator=rng, num_inference_steps=30,
                  guidance_scale=7.5, height=512, width=512)
    result.images[0].save(out / f'cup_{i}.png')
    print(prompt, 'flag:', result.nsfw_content_detected)
(out / 'settings.json').write_text(json.dumps({
    'model': model, 'seed': 2026, 'steps': 30,
    'guidance_scale': 7.5, 'device': device, 'dtype': str(dtype),
    'image_size': [512, 512], 'scheduler': type(pipe.scheduler).__name__,
    'resolved_revision': getattr(pipe.unet.config, '_commit_hash', None)}, indent=2))
