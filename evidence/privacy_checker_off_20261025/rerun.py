import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
from pathlib import Path
import json, hashlib, shutil, time, platform
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
from facenet_pytorch import InceptionResnetV1, MTCNN

ROOT = Path('/home/zcy/book_revision_labs_20261022')
SOURCE = ROOT / 'minimal_examples_20261025/08_privacy'
OUT = ROOT / 'privacy_checker_off_20261025'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(ROOT / 'models/torch')
settings = json.loads((SOURCE / 'settings.json').read_text())
models = json.loads((ROOT / 'manifests/models.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = {}
for name in ['original', 'random', 'protected']:
    shutil.copy2(SOURCE / f'{name}.png', OUT / f'{name}.png')
    inputs[name] = Image.open(OUT / f'{name}.png').convert('RGB')
shutil.copy2(SOURCE / 'input_metrics.json', OUT / 'input_metrics.json')

# Same stored references: do not regenerate the perturbation in this comparison.
encoder = CLIPVisionModelWithProjection.from_pretrained(
    models['ip_adapter']['path'], subfolder='models/image_encoder',
    torch_dtype=torch.float32, local_files_only=True)
pipe = StableDiffusionPipeline.from_pretrained(
    models['sd15']['path'], image_encoder=encoder,
    torch_dtype=torch.float16, local_files_only=True,
    safety_checker=None, requires_safety_checker=False).to('cuda')
pipe.load_ip_adapter(models['ip_adapter']['path'], subfolder='models',
    weight_name='ip-adapter-plus-face_sd15.bin',
    image_encoder_folder=None, local_files_only=True)
pipe.image_encoder.float().eval().requires_grad_(False)
pipe.unet.requires_grad_(False)
projection = pipe.unet.encoder_hid_proj.image_projection_layers[0].float()
pipe.image_encoder.half()
projection.half()
pipe.set_ip_adapter_scale(settings['adapter_scale'])
pipe.set_progress_bar_config(disable=True)
assert pipe.safety_checker is None and pipe.config.requires_safety_checker is False

# Auxiliary evaluator only; it does not filter or alter any generated image.
net = InceptionResnetV1(pretrained='vggface2').eval().cuda().requires_grad_(False)
detector = MTCNN(keep_all=True, device='cuda')
x = to_tensor(inputs['original']).unsqueeze(0).cuda()
with torch.no_grad():
    identity = net((x * 255 - 127.5) / 128)
old_rows = {(r['condition'], r['seed']): r
            for r in json.loads((SOURCE / 'outputs.json').read_text())}
rows = []
started = time.time()
for name, im in inputs.items():
    for seed in settings['seeds']:
        result = pipe(settings['prompt'], ip_adapter_image=im,
            negative_prompt=settings['negative_prompt'],
            generator=torch.Generator('cpu').manual_seed(seed),
            num_inference_steps=settings['generation_steps'],
            guidance_scale=settings['cfg'], height=512, width=512)
        assert result.nsfw_content_detected is None
        generated = result.images[0]
        filename = f'custom_{name}_{seed}.png'
        generated.save(OUT / filename)
        boxes, probabilities = detector.detect(generated)
        valid = [] if boxes is None else [b for b, p in zip(boxes, probabilities) if p >= .95]
        score = None
        if len(valid) == 1:
            b = valid[0]
            margin = .15 * max(b[2] - b[0], b[3] - b[1])
            box = (max(0, int(b[0] - margin)), max(0, int(b[1] - margin)),
                   min(512, int(b[2] + margin)), min(512, int(b[3] + margin)))
            crop = generated.crop(box).resize((160, 160), Image.Resampling.BILINEAR)
            t = to_tensor(crop).unsqueeze(0).cuda()
            with torch.no_grad():
                score = F.cosine_similarity(net((t * 255 - 127.5) / 128), identity).item()
        old_image = np.asarray(Image.open(SOURCE / filename).convert('RGB'))
        row = dict(condition=name, seed=seed, image=filename,
            output_checker_enabled=False, output_flag=None, detected_faces=len(valid),
            identity_cosine=score, sha256=sha(OUT / filename),
            previous_output_flag=old_rows[name, seed]['output_flag'],
            pixels_identical_to_checker_on=bool(np.array_equal(np.asarray(generated), old_image)))
        rows.append(row)
        (OUT / 'outputs.json').write_text(json.dumps(rows, indent=2))
        print(row, flush=True)

manifest = dict(settings=settings, output_checker_enabled=False,
    safety_checker_is_none=pipe.safety_checker is None,
    requires_safety_checker=pipe.config.requires_safety_checker,
    model=models, python=platform.python_version(), torch=torch.__version__,
    diffusers=__import__('diffusers').__version__,
    source_reference_directory=str(SOURCE), references_regenerated=False,
    input_sha256={name: sha(OUT / f'{name}.png') for name in inputs},
    source_sha256=sha(__file__), elapsed_seconds=time.time()-started,
    unflagged_outputs_identical=all(r['pixels_identical_to_checker_on'] for r in rows if not r['previous_output_flag']),
    outputs=len(rows))
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('COMPLETE', json.dumps(manifest, default=str), flush=True)
