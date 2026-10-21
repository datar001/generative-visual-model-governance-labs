import argparse, importlib.metadata as md, json, platform, sys
PROFILES={
 'standard':{},
 'watermark':{'numpy':'1.26.4','opencv-python':'4.10.0.84','Pillow':'10.4.0'},
 'generation':{'torch':'2.5.1','torchvision':'0.20.1','diffusers':'0.32.2','transformers':'4.46.3','accelerate':'1.1.1','huggingface_hub':'0.26.5','numpy':'1.26.4','Pillow':'10.4.0'},
 'privacy':{'torch':'2.2.2','torchvision':'0.17.2','facenet-pytorch':'2.6.0','Pillow':'10.2.0','numpy':'1.26.4'}}
p=argparse.ArgumentParser();p.add_argument('--profile',choices=PROFILES,default='standard')
p.add_argument('--with-torch',action='store_true');a=p.parse_args()
rows=[];ok=sys.version_info[:2]==(3,10)
for name,expected in PROFILES[a.profile].items():
 try:actual=md.version(name)
 except md.PackageNotFoundError:actual=None
 match=actual is not None and actual.split('+')[0]==expected
 rows.append({'package':name,'expected':expected,'installed':actual,'matches_reference':match})
 ok=ok and match
result={'python':platform.python_version(),'profile':a.profile,'reference_python':'3.10','packages':rows,'matches_reference':ok}
if a.with_torch:
 try:
  import torch
  result['cuda_available']=torch.cuda.is_available()
  result['gpu']=torch.cuda.get_device_name() if torch.cuda.is_available() else None
 except ImportError:result['torch_import']='unavailable'
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(0 if ok else 1)
