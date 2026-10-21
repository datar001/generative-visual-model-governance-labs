"""ESD官方完整UNet权重的来源校验和推理证据记录。"""
from pathlib import Path
import hashlib,json,os,sys

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load_config(folder):
    return json.loads((folder/'esd_pretrained.json').read_text(encoding='utf8'))

def checked_weights(config):
    p=Path(os.environ.get('ESD_WEIGHTS','results/esd_weights/'+config['filename']))
    if not p.is_file():raise FileNotFoundError('先运行 esd_support.py --download，或设置 ESD_WEIGHTS。')
    if digest(p)!=config['sha256']:raise ValueError('ESD权重SHA-256不符，停止加载。')
    return p

def inference_device():
    import torch
    requested=os.environ.get('ESD_DEVICE','auto')
    if requested not in ('auto','cpu','cuda'):raise ValueError('ESD_DEVICE必须为auto/cpu/cuda')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
    dtype=torch.float16 if device=='cuda' else torch.float32
    if device=='cuda' and 'GTX 16' in torch.cuda.get_device_name():dtype=torch.float32
    return device,dtype

def prepare_inference(pipe,device):
    if device=='cuda':
        pipe.enable_attention_slicing('max')
        pipe.enable_sequential_cpu_offload()
    else:pipe.to(device)

def save_record(out,config,checkpoint,rows,steps,size,script):
    import torch,diffusers,transformers
    (out/'observations.json').write_text(json.dumps(dict(method='ESD-x pretrained inference; no training',
        config=config,checkpoint_sha256=digest(checkpoint),code_sha256=digest(script),
        python=sys.version,torch=torch.__version__,diffusers=diffusers.__version__,
        transformers=transformers.__version__,steps=steps,size=size,seed=2026,
        guidance_scale=7.5,device=inference_device()[0],
        offload='sequential' if inference_device()[0]=='cuda' else None,results=rows),ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':
    if sys.argv[1:]!=['--download']:raise SystemExit('用法：python esd_support.py --download')
    import requests
    config=load_config(Path(__file__).parent)
    p=Path(os.environ.get('ESD_WEIGHTS','results/esd_weights/'+config['filename']))
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        temp=p.with_suffix('.download')
        with requests.get(config['url'],stream=True,timeout=(30,90)) as response:
            response.raise_for_status()
            with temp.open('wb') as f:
                for chunk in response.iter_content(4*1024*1024):f.write(chunk)
        if digest(temp)!=config['sha256']:
            temp.unlink();raise ValueError('ESD下载校验失败。')
        temp.replace(p)
    print(checked_weights(config))
