import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/06_esd'
out.mkdir(parents=True, exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(
    models['sd14']['path'], torch_dtype=torch.float16, local_files_only=True,
).to('cuda')
prompts = ['a landscape painting in the style of Vincent van Gogh',
           'a photo of a red ceramic cup on a wooden table']
for mode in ['base', 'esd']:
    if mode == 'esd':
        # 加载官方已训练的完整UNet，不在课堂上训练ESD。
        state = torch.load(root / 'models/diffusers-VanGogh-ESDx1-UNET.pt',
                           map_location='cpu', weights_only=True, mmap=True)
        pipe.unet.load_state_dict(state, strict=True, assign=True)
        pipe.unet.to('cuda', dtype=torch.float16)
    for i, prompt in enumerate(prompts):
        result = pipe(prompt, generator=torch.Generator('cpu').manual_seed(2026),
                      num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
        result.images[0].save(out / f'{mode}_{i}.png')
        print(mode, i, 'output_checker:', result.nsfw_content_detected)
