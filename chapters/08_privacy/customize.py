from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection
device = 'cuda'
encoder = CLIPVisionModelWithProjection.from_pretrained(
    'h94/IP-Adapter', subfolder='models/image_encoder', torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=encoder,
    torch_dtype=torch.float16)
pipe.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
                     weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors')
pipe.set_ip_adapter_scale(0.7)
pipe.enable_model_cpu_offload()
out = Path('results/privacy')
for name in ('original.png', 'protected.png', 'protected_q75.jpg'):
    image = Image.open(out/name).convert('RGB')
    result = pipe('a portrait photo of a person in a garden',
                  ip_adapter_image=image, num_inference_steps=30,
                  generator=torch.Generator(device).manual_seed(2026))
    result.images[0].save(out/f'custom_{Path(name).stem}.png')
