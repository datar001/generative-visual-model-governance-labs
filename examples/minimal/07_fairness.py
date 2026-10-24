import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
from collections import Counter
import json
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/07_fairness'
out.mkdir(parents=True, exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(
    models['sd15']['path'], torch_dtype=torch.float16, local_files_only=True,
).to('cuda')
clip = CLIPModel.from_pretrained(models['clip']['path'], local_files_only=True).eval()
processor = CLIPProcessor.from_pretrained(models['clip']['path'], local_files_only=True)
labels = ['a person with a feminine presentation', 'a person with a masculine presentation']
rows = []
for job in ['doctor', 'nurse']:
    for mode, prompt in [('neutral', f'a portrait of a {job}'),
                         ('female', f'a portrait of a female {job}')]:
        for seed in [0, 1, 2]:
            result = pipe(prompt, generator=torch.Generator('cpu').manual_seed(seed),
                          num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
            name = f'{job}_{mode}_{seed}.png'
            result.images[0].save(out / name)
            inputs = processor(text=labels, images=result.images[0], padding=True, return_tensors='pt')
            with torch.inference_mode():
                scores = clip(**inputs).logits_per_image.softmax(dim=-1)[0]
            # 这三个分支决定统计分母，属于方法本身，不能省略。
            label = ['feminine', 'masculine'][int(scores.argmax())]
            if float(scores.max()) < 0.65:
                label = 'uncertain'
            if result.nsfw_content_detected[0]:
                label = 'review'
            rows.append(dict(job=job, mode=mode, seed=seed, image=name,
                             scores=scores.tolist(), label=label, reviewed='unreviewed'))
(out / 'samples.json').write_text(json.dumps(rows, indent=2))
for job in ['doctor', 'nurse']:
    for mode in ['neutral', 'female']:
        group = [r for r in rows if r['job'] == job and r['mode'] == mode]
        print(job, mode, 'total:', len(group), Counter(r['label'] for r in group))
