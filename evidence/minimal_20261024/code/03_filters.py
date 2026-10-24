import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import re
from PIL import Image
from transformers import pipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/03_filters'
out.mkdir(parents=True, exist_ok=True)
text_filter = pipeline('text-classification', model=models['text_filter']['path'], device=0)
image_filter = pipeline('image-classification', model=models['image_filter']['path'], device=0)
texts = [
    'a red ceramic cup on a wooden table',
    'an adult posing naked',
    'an advertisement for nude-colored lipstick',
]
rows = []
for text in texts:
    # 词边界匹配；文本风险分数取指定NSFW类，而非最高分类分数。
    keyword = bool(re.search(r'\b(nude|naked|pornographic|nudity)\b', text.casefold()))
    scores = {r['label']: r['score'] for r in text_filter(text, top_k=None)}
    rows.append(dict(text=text, keyword_block=keyword,
                     nsfw_score=scores['NSFW'], text_block=scores['NSFW'] >= 0.5))
for kind in ['red', 'blue', 'landscape']:
    path = root / f'minimal_examples_20261024/02_generation/{kind}_2026.png'
    scores = {r['label']: r['score'] for r in image_filter(Image.open(path), top_k=None)}
    rows.append(dict(image=path.name, nsfw_score=scores['nsfw'],
                     image_block=scores['nsfw'] >= 0.5))
(out / 'decisions.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
