from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import models, transforms
from cnnspot_support import load_weights, save_manifest
# 1. 空间域 CNN 二分类：ResNet-50 的最后一层只输出一个 logit。
model = models.resnet50(weights=None, num_classes=1)
weights = load_weights()
state = torch.load(weights, map_location='cpu', weights_only=True)
model.load_state_dict(state['model'], strict=True)
model.eval()
# 2. 沿用官方演示的全图预处理，不缩放、不裁剪。
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
fixtures = Path(__file__).parent / 'detection_fixtures'
inputs = [('real_coffee.png', 'real'), ('generated_cup.png', 'fake')]
out = Path('results/detection')
out.mkdir(parents=True, exist_ok=True)
rows = []
for name, source in inputs:
    original = Image.open(fixtures / name).convert('RGB')
    # 3. 原图单列，质量 100 仍属于 JPEG 重新编码。
    for quality in ('original', 100, 75, 40):
        image = original
        if quality != 'original':
            target = out / f'{Path(name).stem}_q{quality}.jpg'
            original.save(target, quality=quality)
            image = Image.open(target).convert('RGB')
        with torch.inference_mode():
            logit = model(preprocess(image).unsqueeze(0)).item()
            score = torch.sigmoid(torch.tensor(logit)).item()
        # 4. sigmoid 分数越大越倾向生成类；0.5 仅作固定教学阈值。
        rows.append(dict(image=name, source=source, quality=quality,
                         fake_score=score,
                         prediction='fake' if score >= .5 else 'real'))
with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
# 5. 保存版本、权重与输入哈希，分数不能代替来源事实。
save_manifest(out, weights, fixtures, inputs, Path(__file__))
print(rows)
