import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from facenet_pytorch import InceptionResnetV1
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

root = Path(os.environ['LAB_ROOT'])
os.environ['TORCH_HOME'] = str(root / 'models/torch')
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'minimal_examples_20261024/08_privacy'
out.mkdir(parents=True, exist_ok=True)
# 已裁剪的合成成人头像，配套仓库提供；不处理未授权真实人脸。
image = Image.open(root / 'inputs/face_reference.png').convert('RGB')
x = to_tensor(image).unsqueeze(0).to('cuda')
net = InceptionResnetV1(pretrained='vggface2').eval().to('cuda').requires_grad_(False)
def encode(pixels):
    return net((pixels * 255 - 127.5) / 128)
with torch.no_grad():
    reference = encode(x)
torch.manual_seed(2026)
epsilon = 8 / 255
delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
# 沿梯度降低FaceNet相似度，每步投影回像素扰动预算。
for step in range(40):
    delta = delta.detach().requires_grad_(True)
    loss = F.cosine_similarity(encode((x + delta).clamp(0, 1)), reference).mean()
    gradient, = torch.autograd.grad(loss, delta)
    delta = (delta - gradient.sign() / 255).clamp(-epsilon, epsilon)
    delta = (x + delta).clamp(0, 1) - x
pixels = ((x + delta).detach().cpu()[0] * 255).round().clamp(0, 255).byte()
protected = to_pil_image(pixels)
image.save(out / 'original.png')
protected.save(out / 'protected.png')
with torch.no_grad():
    reloaded = to_tensor(Image.open(out / 'protected.png')).unsqueeze(0).to('cuda')
    score = F.cosine_similarity(encode(reloaded), reference).item()
(out / 'feature_score.json').write_text(json.dumps(dict(protected_input_cosine=score)))
# 实际加载Plus Face适配器，比较原图和保护图作为参考时的定制结果。
encoder = CLIPVisionModelWithProjection.from_pretrained(
    models['ip_adapter']['path'], subfolder='models/image_encoder',
    torch_dtype=torch.float16, local_files_only=True,
)
pipe = StableDiffusionPipeline.from_pretrained(
    models['sd15']['path'], torch_dtype=torch.float16, local_files_only=True,
    image_encoder=encoder,
).to('cuda')
pipe.load_ip_adapter(models['ip_adapter']['path'], subfolder='models',
                     weight_name='ip-adapter-plus-face_sd15.bin',
                     image_encoder_folder=None, local_files_only=True)
pipe.set_ip_adapter_scale(0.3)
prompt = ('a close-up frontal headshot photograph of an adult woman, neutral expression, '
          'face centered and large in the frame, fully clothed, soft daylight, blurred garden background')
negative = 'low quality, blurry, distorted face, deformed eyes, cartoon, illustration'
for name, reference_image in [('original', image), ('protected', protected)]:
    result = pipe(prompt, ip_adapter_image=reference_image, negative_prompt=negative,
                  generator=torch.Generator('cpu').manual_seed(2026),
                  num_inference_steps=30, guidance_scale=5.0, height=512, width=512)
    result.images[0].save(out / f'custom_{name}.png')
    print(name, 'output_checker:', result.nsfw_content_detected)
