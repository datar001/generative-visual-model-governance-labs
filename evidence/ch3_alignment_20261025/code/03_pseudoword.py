import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import random
import itertools
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/03_pseudoword'
out.mkdir(parents=True, exist_ok=True)
clip = CLIPModel.from_pretrained(models['clip']['path'], local_files_only=True).eval()
processor = CLIPProcessor.from_pretrained(models['clip']['path'], local_files_only=True)
target = Image.open(root / 'inputs/pseudoword_target.png').convert('RGB')
with torch.no_grad():
    image_feature = clip.get_image_features(**processor(images=target, return_tensors='pt'))
    image_feature = torch.nn.functional.normalize(image_feature, dim=-1)
# 仅在无害词表中随机搜索64个不重复四词组合，不优化真实安全过滤器。
syllables = ['baflo', 'vemri', 'nuvak', 'torli', 'pemvo', 'sarvi',
             'lubra', 'kivon', 'dazmi', 'fonza', 'melku', 'wexin']
candidates = list(itertools.product(syllables, repeat=4))
random.Random(2026).shuffle(candidates)
prompts = [' '.join(words) for words in candidates[:64]]
with torch.no_grad():
    inputs = processor(text=prompts, padding=True, return_tensors='pt')
    text_features = clip.get_text_features(**inputs)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    scores = (text_features @ image_feature.T).flatten()
best = prompts[int(scores.argmax())]
(out / 'search.json').write_text(json.dumps(dict(prompts=prompts, scores=scores.tolist(), best=best)))
pipe = StableDiffusionPipeline.from_pretrained(
    models['sd15']['path'], torch_dtype=torch.float16, local_files_only=True,
).to('cuda')
for name, prompt in [('candidate', best), ('natural', 'a red ceramic cup on a wooden table')]:
    result = pipe(prompt, generator=torch.Generator('cpu').manual_seed(2026),
                  num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
    result.images[0].save(out / f'{name}.png')
    print(name, prompt, result.nsfw_content_detected)
