import os
os.environ['LAB_ROOT']='/home/zcy/book_revision_labs_20261022'
from pathlib import Path
import time, json, hashlib, io, math
base=Path('/home/zcy/book_revision_labs_20261022')
source=base/'src/08_privacy_optimized_20261025.py'
prefix=source.read_text(encoding='utf-8-sig').split('with torch.no_grad():\n    reference = condition(x).detach()')[0]
exec(compile(prefix,str(source),'exec'),globals())
OUT=base/'privacy_opt_20261025/holdout'
OUT.mkdir(parents=True,exist_ok=True)
all_inputs={};input_rows=[]
# These references/prompts/seeds were not used to choose the 3/255 configuration.
for ident in ['face_2027','face_2028']:
    image=Image.open(base/f'results/08_privacy/{ident}/original.png').convert('RGB')
    folder=OUT/ident;folder.mkdir(exist_ok=True)
    x=to_tensor(image).unsqueeze(0).cuda()
    with torch.no_grad():reference=condition(x).detach()
    torch.manual_seed(2026);eps=3/255
    delta=torch.empty_like(x).uniform_(-eps,eps);momentum=torch.zeros_like(x);trace=[]
    for step in range(120):
        adv=(x+delta).clamp(0,1).detach().requires_grad_(True)
        loss=F.cosine_similarity(condition(adv,True),reference).mean()
        grad,=torch.autograd.grad(loss,adv)
        momentum=.9*momentum+grad/grad.abs().mean().clamp_min(1e-12)
        delta=(delta-(.5 if step<90 else .25)/255*momentum.sign()).clamp(-eps,eps)
        delta=(x+delta).clamp(0,1)-x
        trace.append(float(loss.detach()))
    protected=png_image(x+delta)
    torch.manual_seed(2026)
    random_image=png_image((x+torch.empty_like(x).uniform_(-eps,eps)).clamp(0,1))
    image.save(folder/'original.png');protected.save(folder/'protected.png');random_image.save(folder/'random.png')
    protected.save(folder/'protected_q75.jpg',quality=75)
    (folder/'trajectory.json').write_text(json.dumps(trace))
    all_inputs[ident]={v:Image.open(folder/(v+('.jpg' if v.endswith('q75') else '.png'))).convert('RGB') for v in ['original','random','protected','protected_q75']}
    print('OPTIMIZED',ident,flush=True)
pipe.image_encoder.half();projection.half();pipe.set_progress_bar_config(disable=True)
net=InceptionResnetV1(pretrained='vggface2').eval().cuda().requires_grad_(False)
detector=MTCNN(keep_all=True,device='cuda')
references={}
with torch.no_grad():
    for ident,inputs in all_inputs.items():
        x=to_tensor(inputs['original']).unsqueeze(0).cuda()
        references[ident]=net((x*255-127.5)/128)
        h,_=pipe.encode_image(inputs['original'],'cuda',1,True)
        clean=projection(h).float().flatten(1)
        for name,im in inputs.items():
            t=to_tensor(im).unsqueeze(0).cuda();mse=float((t-x).square().mean())
            h,_=pipe.encode_image(im,'cuda',1,True)
            input_rows.append(dict(id=ident,condition=name,linf_255=float((t-x).abs().max())*255,psnr_db=None if mse==0 else -10*math.log10(mse),conditioning_cosine=float(F.cosine_similarity(projection(h).float().flatten(1),clean))))
(OUT/'input_metrics.json').write_text(json.dumps(input_rows,indent=2))
prompts={
 'studio':'a close-up frontal headshot photograph of an adult woman, neutral expression, face centered and large in the frame, fully clothed, studio lighting, plain blue background',
 'watercolor':'a watercolor portrait of an adult woman, face centered, fully clothed, simple pale background',
}
negative='low quality, blurry, distorted face, deformed eyes, cartoon, illustration'
rows=[]
for scale in [.3,.5]:
    pipe.set_ip_adapter_scale(scale)
    for ident,inputs in all_inputs.items():
        for pkey,prompt in prompts.items():
            if scale==.5 and pkey!='studio':continue
            for name,im in inputs.items():
                if scale==.5 and name not in ['original','protected']:continue
                for seed in [2030,2031]:
                    result=pipe(prompt,ip_adapter_image=im,negative_prompt=negative,generator=torch.Generator('cpu').manual_seed(seed),num_inference_steps=30,guidance_scale=5.,height=512,width=512)
                    generated=result.images[0];flag=bool(result.nsfw_content_detected[0])
                    filename=f'{ident}/{pkey}_{scale}_{name}_{seed}.png';generated.save(OUT/filename)
                    boxes,probs=detector.detect(generated)
                    valid=[] if boxes is None else [b for b,p in zip(boxes,probs) if p>=.95]
                    sim=None
                    if len(valid)==1 and not flag:
                        b=valid[0];m=.15*max(b[2]-b[0],b[3]-b[1])
                        box=(max(0,int(b[0]-m)),max(0,int(b[1]-m)),min(512,int(b[2]+m)),min(512,int(b[3]+m)))
                        crop=generated.crop(box).resize((160,160),Image.Resampling.BILINEAR)
                        t=to_tensor(crop).unsqueeze(0).cuda()
                        with torch.no_grad():sim=float(F.cosine_similarity(net((t*255-127.5)/128),references[ident]))
                    row=dict(id=ident,prompt=pkey,adapter_scale=scale,condition=name,seed=seed,output_flag=flag,detected_faces=len(valid),identity_cosine=sim,image=filename)
                    rows.append(row);(OUT/'outputs.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
(OUT/'settings.json').write_text(json.dumps(dict(prompts=prompts,negative_prompt=negative,seeds=[2030,2031],steps=30,cfg=5.,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),optimization_budget=3/255,optimization_steps=120,selection='held-out from configuration choice'),indent=2))
print('COMPLETE',len(rows),flush=True)
