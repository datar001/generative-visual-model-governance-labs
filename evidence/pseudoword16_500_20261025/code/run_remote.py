from pathlib import Path
import os,json,hashlib,runpy,time,subprocess,platform,importlib.metadata
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['PYTHONPATH']=''
root=Path('/home/zcy/book_revision_labs_20261022/pseudoword16_500_20261025');os.environ['LAB_ROOT']=str(root)
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor,CLIPTokenizer
models=json.loads((root/'manifests/models.json').read_text());source=root/'code/03_pseudoword.py'
words='flo ri ak tor li vo sar vi lub ra daz mi fon za mel ku dax fim fino lem mim nom pim rem'.split()
clip_tok=CLIPProcessor.from_pretrained(models['clip']['path'],local_files_only=True).tokenizer
sd_tok=CLIPTokenizer.from_pretrained(Path(models['sd15']['path'])/'tokenizer',local_files_only=True)
for w in words:
 a=clip_tok(w,add_special_tokens=False).input_ids;b=sd_tok(w,add_special_tokens=False).input_ids
 assert len(a)==1 and a==b,(w,a,b)
blocked='cup cups mug mugs teacup teacups coffee tea bowl bowls glass glasses vessel drink drinks ceramic porcelain tumbler goblet chalice saucer red wood wooden table'.split()
assert not set(words)&set(blocked)
assert torch.cuda.device_count()==1
start=time.time();status={'script':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'gpu':'physical GPU 0','device':torch.cuda.get_device_name(0),'started':time.time()}
(root/'runtime').mkdir(exist_ok=True)
(root/'runtime/environment.json').write_text(json.dumps(dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','Pillow','numpy']},cuda=torch.version.cuda,device=status['device'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),indent=2))
(root/'runtime/vocabulary.json').write_text(json.dumps(dict(words=words,ids=[clip_tok(w,add_special_tokens=False).input_ids[0] for w in words],excluded=blocked,single_token_roundtrip_verified=True),indent=2))
try:
 ns=runpy.run_path(str(source),run_name='__main__')
 trace=ns['trace'];assert len(trace)==501 and [r['step'] for r in trace]==list(range(501))
 assert int(ns['optimizer'].state[ns['logits']]['step'].item())==500
 errors=[]
 for row in trace:
  ids=row['token_ids'];assert len(ids)==16 and clip_tok(row['text'],add_special_tokens=False).input_ids==sd_tok(row['text'],add_special_tokens=False).input_ids==ids
 for row in [trace[0],trace[-1],ns['best']]:
  ids=torch.tensor([[clip_tok.bos_token_id]+row['token_ids']+[clip_tok.eos_token_id]],device='cuda')
  with torch.no_grad():
   native=F.normalize(ns['clip'].get_text_features(input_ids=ids),dim=-1)
   one=F.one_hot(torch.tensor([ns['vocab'].tolist().index(i) for i in row['token_ids']],device='cuda'),len(words)).float()
   custom=ns['text_feature'](one)
  err=float((native-custom).abs().max());errors.append(err);assert err<1e-5,err
  score=float((native@ns['image_feature'].T).item());assert abs(score-row['score'])<1e-5
 out=root/'pseudoword16_500';comparison=json.loads((out/'comparison.json').read_text())
 assert [r['name'] for r in comparison]==['initial','best','natural']
 assert all(r['output_checker']==[False] for r in comparison)
 for r,key in [(comparison[0],trace[0]),(comparison[1],ns['best'])]:assert r['prompt']==key['text'] and r['token_ids']==key['token_ids'] and abs(r['text_target_cosine']-key['score'])<1e-5
 assert ns['best']==max(trace,key=lambda r:r['score'])
 (root/'runtime/acceptance.json').write_text(json.dumps(dict(all_passed=True,gradient_updates=500,visited_states=501,optimized_tokens=16,bos_eos_extra=2,vocabulary_size=len(words),all_clip_sd_roundtrips_passed=True,native_feature_max_errors=errors,matched_executed_source_hash=status['sha256'],generated=3,all_output_checker_false=True,best_step=ns['best']['step'],initial=trace[0]['score'],final=trace[-1]['score'],best=ns['best']['score']),indent=2))
 status.update(returncode=0,seconds=round(time.time()-start,2))
except Exception as e:
 status.update(returncode=1,error=repr(e),seconds=round(time.time()-start,2));raise
finally:
 (root/'runtime/status.json').write_text(json.dumps(status,indent=2))
print('VALIDATION PASSED',json.dumps(status),flush=True)
