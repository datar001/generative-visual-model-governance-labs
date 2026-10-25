import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('PRIVACY_GPU', '0')
from pathlib import Path
import json, time, hashlib, gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline
from facenet_pytorch import InceptionResnetV1, MTCNN

ROOT = Path('/home/zcy/book_revision_labs_20261022')
OUT = ROOT / 'privacy_opt_20261025/probe3'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(ROOT / 'models/torch')
spec = json.loads((ROOT / 'manifests/models.json').read_text())
torch.manual_seed(2026)
torch.backends.cudnn.benchmark = False
image = Image.open(ROOT / 'inputs/face_reference.png').convert('RGB')
image.save(OUT / 'original.png')
old = Image.open(ROOT / 'all_chapters_alignment_20261025/minimal_examples_20261024/08_privacy/protected.png').convert('RGB')
old.save(OUT / 'facenet8.png')
encoder = CLIPVisionModelWithProjection.from_pretrained(spec['ip_adapter']['path'], subfolder='models/image_encoder', torch_dtype=torch.float32, local_files_only=True)
pipe = StableDiffusionPipeline.from_pretrained(spec['sd15']['path'], image_encoder=encoder, torch_dtype=torch.float16, local_files_only=True).to('cuda')
pipe.image_encoder = pipe.image_encoder.float().eval().requires_grad_(False)
pipe.load_ip_adapter(spec['ip_adapter']['path'], subfolder='models', weight_name='ip-adapter-plus-face_sd15.bin', image_encoder_folder=None, local_files_only=True)
pipe.unet.requires_grad_(False)
projection = pipe.unet.encoder_hid_proj.image_projection_layers[0].float().eval()
processor = pipe.feature_extractor
x = to_tensor(image).unsqueeze(0).cuda()
mean = x.new_tensor(processor.image_mean).view(1,3,1,1)
std = x.new_tensor(processor.image_std).view(1,3,1,1)

def exact_pixels(t):
    im = to_pil_image((t.detach().cpu()[0] * 255).round().clamp(0,255).byte())
    return processor(images=im, return_tensors='pt').pixel_values.cuda()

def condition(t, grad=False):
    exact = exact_pixels(t)
    if grad:
        surrogate = (F.interpolate(t, size=(224,224), mode='bicubic', align_corners=False, antialias=True) - mean) / std
        exact = exact + (surrogate - surrogate.detach())
    hidden = pipe.image_encoder(exact, output_hidden_states=True).hidden_states[-2]
    return projection(hidden)

with torch.no_grad():
    clean = condition(x).detach()
    alternate = Image.open(ROOT/'results/08_privacy/face_2027/original.png').convert('RGB')
    target = condition(to_tensor(alternate).unsqueeze(0).cuda()).detach()
alternate.save(OUT/'target.png')
variants = ['original','facenet8']
metrics = []

for method, budget, steps in [('untargeted',3,120)]:
    name = f'{method}{budget}'
    torch.manual_seed(2026)
    eps = budget/255
    delta = torch.empty_like(x).uniform_(-eps,eps)
    momentum = torch.zeros_like(x)
    trace=[]; started=time.time()
    for step in range(steps):
        adv=(x+delta).clamp(0,1).detach().requires_grad_(True)
        cond=condition(adv,grad=True)
        loss = F.cosine_similarity(cond.flatten(1), clean.flatten(1)).mean() if method=='untargeted' else F.mse_loss(cond,target)
        gradient, = torch.autograd.grad(loss, adv)
        assert torch.isfinite(gradient).all()
        momentum = .9*momentum + gradient/gradient.abs().mean().clamp_min(1e-12)
        alpha=(.5 if step < steps*0.75 else .25)/255
        delta=(delta-alpha*momentum.sign()).clamp(-eps,eps)
        delta=(x+delta).clamp(0,1)-x
        if step%20==0 or step==steps-1:
            row=dict(step=step+1,loss=float(loss),cosine=float(F.cosine_similarity(cond.flatten(1),clean.flatten(1))),seconds=round(time.time()-started,2))
            trace.append(row);print(name,row,flush=True)
    protected=to_pil_image(((x+delta).detach().cpu()[0]*255).round().clamp(0,255).byte())
    protected.save(OUT/f'{name}.png'); variants.append(name)
    (OUT/f'{name}_trajectory.json').write_text(json.dumps(trace,indent=2))

torch.manual_seed(2026)
random = (x + torch.empty_like(x).uniform_(-4/255,4/255)).clamp(0,1)
to_pil_image((random.cpu()[0]*255).round().byte()).save(OUT/'random4.png'); variants.append('random4')
with torch.no_grad():
    for name in variants:
        im=Image.open(OUT/f'{name}.png').convert('RGB'); t=to_tensor(im).unsqueeze(0).cuda()
        cond=condition(t)
        mse=float((t-x).square().mean())
        metrics.append(dict(variant=name,linf_255=round(float((t-x).abs().max())*255,4),psnr_db=None if mse==0 else -10*np.log10(mse),conditioning_cosine=float(F.cosine_similarity(cond.flatten(1),clean.flatten(1)))))
(OUT/'input_metrics.json').write_text(json.dumps(metrics,indent=2))
pipe.image_encoder.half(); projection.half(); pipe.set_ip_adapter_scale(.3); pipe.set_progress_bar_config(disable=True)
with torch.no_grad():
    clean_h,_=pipe.encode_image(image,'cuda',1,True)
    clean_half=projection(clean_h).float().flatten(1)
    half_rows=[]
    for name in variants:
        hidden,_=pipe.encode_image(Image.open(OUT/f'{name}.png').convert('RGB'),'cuda',1,True)
        token=projection(hidden).float().flatten(1)
        half_rows.append(dict(variant=name,conditioning_cosine=float(F.cosine_similarity(token,clean_half))))
(OUT/'fp16_conditioning.json').write_text(json.dumps(half_rows,indent=2))
net=InceptionResnetV1(pretrained='vggface2').eval().cuda().requires_grad_(False)
detector=MTCNN(keep_all=True,device='cuda')
def face_feature(im, detect=True):
    if detect:
        boxes,probs=detector.detect(im)
        valid=[] if boxes is None else [b for b,p in zip(boxes,probs) if p>=.95]
        if len(valid)!=1:return None,len(valid)
        b=valid[0]; margin=.15*max(b[2]-b[0],b[3]-b[1])
        box=(max(0,int(b[0]-margin)),max(0,int(b[1]-margin)),min(im.width,int(b[2]+margin)),min(im.height,int(b[3]+margin)))
        im=im.crop(box).resize((160,160),Image.Resampling.BILINEAR)
    t=to_tensor(im).unsqueeze(0).cuda()
    with torch.no_grad():return net((t*255-127.5)/128),1
reference,_=face_feature(image,False)
prompt='a close-up frontal headshot photograph of an adult woman, neutral expression, face centered and large in the frame, fully clothed, soft daylight, blurred garden background'
negative='low quality, blurry, distorted face, deformed eyes, cartoon, illustration'
rows=[]
for name in variants:
    im=Image.open(OUT/f'{name}.png').convert('RGB')
    for seed in [2026,2027,2028]:
        result=pipe(prompt,ip_adapter_image=im,negative_prompt=negative,generator=torch.Generator('cpu').manual_seed(seed),num_inference_steps=30,guidance_scale=5.,height=512,width=512)
        flag=bool(result.nsfw_content_detected[0]); generated=result.images[0]
        filename=f'custom_{name}_{seed}.png';generated.save(OUT/filename)
        feature,count=face_feature(generated)
        sim=None if feature is None or flag else float(F.cosine_similarity(feature,reference))
        row=dict(variant=name,seed=seed,output_flag=flag,detected_faces=count,output_identity_cosine=sim,image=filename)
        rows.append(row);(OUT/'outputs.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
(OUT/'settings.json').write_text(json.dumps(dict(prompt=prompt,negative_prompt=negative,seeds=[2026,2027,2028],adapter_scale=.3,cfg=5,steps=30,models=spec,preprocessing='exact PIL/CLIP forward, differentiable bicubic straight-through backward',optimized='actual post-Resampler IP conditioning',evaluation='FaceNet not used in optimized variants'),indent=2))
print('COMPLETE',OUT,flush=True)
