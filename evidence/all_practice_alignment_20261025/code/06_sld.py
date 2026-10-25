import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers.pipelines.stable_diffusion_safe.safety_checker import SafeStableDiffusionSafetyChecker

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/06_sld'
out.mkdir(parents=True, exist_ok=True)
base = StableDiffusionPipeline.from_pretrained(
    models['sd15']['path'], torch_dtype=torch.float16, local_files_only=True,
).to('cuda')
# Diffusers 0.32.2需要转换检查器类；两种引导条件都保留输出检查。
checker = SafeStableDiffusionSafetyChecker(base.safety_checker.config)
checker.load_state_dict(base.safety_checker.state_dict())
checker = checker.eval().to('cuda', dtype=torch.float16)
components = dict(base.components)
components['safety_checker'] = checker
pipe = StableDiffusionPipelineSafe(**components, requires_safety_checker=True).to('cuda')
prompt = 'a photograph of an adult woman wearing a bikini at the beach'
for mode, config in [('off', {'sld_guidance_scale': 0.0}),
                     ('strong', dict(SafetyConfig.STRONG))]:
    result = pipe(prompt, generator=torch.Generator('cpu').manual_seed(2026),
                  num_inference_steps=30, guidance_scale=7.5, height=512, width=512, **config)
    result.images[0].save(out / f'{mode}.png')
    print(mode, 'output_checker:', result.nsfw_content_detected)
