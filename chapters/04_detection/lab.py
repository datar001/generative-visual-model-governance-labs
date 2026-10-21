from pathlib import Path
import csv
from PIL import Image
from transformers import pipeline
# 1. 加载鉴伪分类器，保留模型原始类别名称。
detector = pipeline('image-classification', model='umm-maybe/AI-image-detector')
out = Path('results/detection')
out.mkdir(parents=True, exist_ok=True)
# 输入为授权真实图与生成图，不能据两张图评估总体性能。
inputs = ['inputs/real.jpg', 'results/generation/cup_0.png']
rows = []
for name in inputs:
    original = Image.open(name).convert('RGB')
    # 2. 先记录未重新压缩的输入分数，作为对照基线。
    scores = detector(original, top_k=None)
    rows.append({'image': name, 'quality': 'original',
                 **{x['label']: x['score'] for x in scores}})
    # 3. 比较三档 JPEG 压缩；质量 100 也会重新编码。
    for quality in (100, 75, 40):
        target = out / f'{Path(name).stem}_q{quality}.jpg'
        original.save(target, quality=quality)
        scores = detector(Image.open(target), top_k=None)
        row = {'image': name, 'quality': quality}
        row.update({x['label']: x['score'] for x in scores})
        rows.append(row)
# 4. 保存各类别分数；分类分数不是图像来源的证明。
fields = list(dict.fromkeys(k for row in rows for k in row))
with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(rows)
