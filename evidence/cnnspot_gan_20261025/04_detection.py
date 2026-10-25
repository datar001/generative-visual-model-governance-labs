import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import torch
from PIL import Image
from torchvision import models, transforms

root = Path(os.environ['LAB_ROOT'])
out = root / 'minimal_examples_20261024/04_detection'
out.mkdir(parents=True, exist_ok=True)
# CNNSpot：空间域ResNet-50二分类，单个logit经sigmoid得到生成类分数。
net = models.resnet50(weights=None, num_classes=1).eval().to('cuda')
state = torch.load(root / 'models/blur_jpg_prob0.5.pth', map_location='cpu', weights_only=True)
net.load_state_dict(state['model'])
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
photos = root / 'inputs/real_photos'
real = json.loads((photos / 'manifest.json').read_text())[0]['file']
inputs = [('real', photos / real),
          ('progan', root / 'inputs/gan/progan_2026.png'),
          ('sd15', root / 'minimal_examples_20261024/02_generation/red_2026.png')]
rows = []
for source, path in inputs:
    image = Image.open(path).convert('RGB')
    with torch.inference_mode():
        score = net(preprocess(image).unsqueeze(0).to('cuda')).sigmoid().item()
    rows.append(dict(source=source, file=path.name, fake_score=score, predicted_fake=score >= 0.5))
(out / 'scores.json').write_text(json.dumps(rows, indent=2))
print(rows)
