from pathlib import Path
import io; import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms.functional import to_tensor, to_pil_image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 1. 冻结人脸编码器；输入应为已授权、对齐的单人脸。
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model.requires_grad_(False)
x = to_tensor(Image.open('inputs/face.png').convert('RGB').resize((160,160)))
x = x.unsqueeze(0).to(device)
# 将像素从 [0,1] 映射到编码器所用的 [-1,1]。
def encode(t): return model((t-.5)/.5)
with torch.no_grad(): reference = encode(x)
torch.manual_seed(2026)
# 2. 将每个像素的扰动限制在 8/255 内。
epsilon = 8/255; delta = torch.empty_like(x).uniform_(-epsilon,epsilon)
for step in range(40):
    delta = delta.detach().requires_grad_(True)
    # 计算与原图特征的相似度，并沿梯度方向降低它。
    score = F.cosine_similarity(encode((x+delta).clamp(0,1)),reference).mean()
    gradient, = torch.autograd.grad(score,delta)
    # 每步投影回扰动范围，再保证像素仍位于 [0,1]。
    delta = (delta-(1/255)*gradient.sign()).clamp(-epsilon,epsilon)
    delta = (x+delta).clamp(0,1)-x
protected = to_pil_image((x+delta).detach().cpu()[0])
out = Path('results/privacy'); out.mkdir(parents=True, exist_ok=True)
to_pil_image(x.cpu()[0]).save(out/'original.png'); protected.save(out/'protected.png')
# 3. 重新压缩后再编码，检查防护效果是否衰减。
buffer = io.BytesIO(); protected.save(buffer,format='JPEG',quality=75); buffer.seek(0)
compressed = to_tensor(Image.open(buffer).convert('RGB')).unsqueeze(0).to(device)
# 相似度降低仅针对当前编码器，不等于所有模型都受保护。
with torch.no_grad():
    print('protected',float(F.cosine_similarity(encode((x+delta).clamp(0,1)),reference)))
    print('jpeg75',float(F.cosine_similarity(encode(compressed),reference)))
Image.open(buffer).save(out/'protected_q75.jpg')
