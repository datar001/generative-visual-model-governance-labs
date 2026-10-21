from pathlib import Path
import gc, os, torch
from diffusers import StableDiffusionPipeline
from esd_support import load_config, checked_weights, save_record
from esd_support import inference_device, prepare_inference
# 1. 固定配套基础模型与官方ESD-x权重，本实践不训练。
config = load_config(Path(__file__).parent)
checkpoint = checked_weights(config)
base = os.environ.get('ESD_BASE_MODEL', config['base_model'])
revision = None if Path(base).is_dir() else config['base_revision']
device, dtype = inference_device()
steps = int(os.environ.get('ESD_STEPS', 20))
size = int(os.environ.get('ESD_IMAGE_SIZE', 512))
prompts = ['a landscape painting in the style of Vincent van Gogh',
           'a photo of a red ceramic cup on a wooden table']
out = Path(os.environ.get('ESD_OUTPUT_DIR', 'results/esd'))
out.mkdir(parents=True, exist_ok=True)
rows = []
for mode in ('base', 'esd'):
    # 2. 两模式先后加载相同基础组件，不同时驻留两个模型。
    pipe = StableDiffusionPipeline.from_pretrained(
        base, revision=revision, torch_dtype=dtype, use_safetensors=True)
    if mode == 'esd':
        # 严格核对后直接接管参数，避免再复制整份UNet；随后挂卸载钩子。
        state = torch.load(checkpoint, map_location='cpu',
                           weights_only=True, mmap=True)
        pipe.unet.load_state_dict(state, strict=True, assign=True)
        pipe.unet.to(dtype=dtype)
        del state; gc.collect()
    prepare_inference(pipe, device)
    for i, prompt in enumerate(prompts):
        print('Generating', mode, i, flush=True)
        # 3. 相同种子、提示与采样设置，保留输出安全检查器。
        result = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5,
            height=size, width=size,
            generator=torch.Generator('cpu').manual_seed(2026))
        filename = f'{i}_{mode}.png'
        result.images[0].save(out/filename)
        rows.append(dict(prompt=prompt, mode=mode, file=filename,
                         output_flag=result.nsfw_content_detected))
        print(filename, result.nsfw_content_detected, flush=True)
    del pipe; gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
save_record(out, config, checkpoint, rows, steps, size, Path(__file__))
