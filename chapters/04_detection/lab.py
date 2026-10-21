from pathlib import Path
import csv
from PIL import Image
from transformers import pipeline
detector = pipeline('image-classification', model='umm-maybe/AI-image-detector')
out = Path('results/detection')
out.mkdir(parents=True, exist_ok=True)
inputs = ['inputs/real.jpg', 'results/generation/cup_0.png']
rows = []
for name in inputs:
    original = Image.open(name).convert('RGB')
    scores = detector(original, top_k=None)
    rows.append({'image': name, 'quality': 'original',
                 **{x['label']: x['score'] for x in scores}})
    for quality in (100, 75, 40):
        target = out / f'{Path(name).stem}_q{quality}.jpg'
        original.save(target, quality=quality)
        scores = detector(Image.open(target), top_k=None)
        row = {'image': name, 'quality': quality}
        row.update({x['label']: x['score'] for x in scores})
        rows.append(row)
fields = list(dict.fromkeys(k for row in rows for k in row))
with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(rows)
