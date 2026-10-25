from pathlib import Path
import os,json,time,hashlib,runpy,platform,importlib.metadata
os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='')
root=Path('/home/zcy/book_revision_labs_20261022/pseudoword_l14_20261025');os.environ['LAB_ROOT']=str(root)
model=Path('/home/zcy/book_revision_labs_20261022/models/clip_l14_32bd642');sd=Path('/home/zcy/book_revision_labs_20261022/models/sd15_reused')
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import CLIPTokenizer
source=root/'code/03_pseudoword.py';start=time.time();status=dict(script=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gpu='physical GPU 0',device=torch.cuda.get_device_name(0))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
try:
 assert torch.cuda.device_count()==1
 assert sha(model/'model.safetensors')=='a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb'
 with safe_open(model/'model.safetensors',framework='pt',device='cpu') as full,safe_open(sd/'text_encoder/model.safetensors',framework='pt',device='cpu') as target:
  fk={k for k in full.keys() if k.startswith('text_model.') and not k.endswith('.position_ids')};sk={k for k in target.keys() if not k.endswith('.position_ids')};assert fk==sk
  checks=[]
  for key in sorted(fk):
   a=full.get_tensor(key);b=target.get_tensor(key);err=float((a.float()-b.float()).abs().max());checks.append(dict(key=key,shape=list(a.shape),clip_dtype=str(a.dtype),sd_dtype=str(b.dtype),exact_equal=torch.equal(a.float(),b.float()),max_abs_difference=err))
 alignment=dict(repo='openai/clip-vit-large-patch14',revision='32bd64288804d66eefd0ccbe215aa642df71cc41',parameter_tensors=len(checks),equal_tensors=sum(x['exact_equal'] for x in checks),max_abs_difference=max(x['max_abs_difference'] for x in checks),all_equal=all(x['exact_equal'] for x in checks),checks=checks,comparison_precision='original weights converted to float32',sd_generation_precision='float16',text_projection_is_not_sd_condition=True)
 (root/'runtime/text_encoder_alignment.json').write_text(json.dumps(alignment,indent=2));print('WEIGHT ALIGNMENT',json.dumps({k:v for k,v in alignment.items() if k!='checks'}),flush=True)
 assert alignment['all_equal'],'L14 text weights differ from SD1.5; stopping before optimization'
 manifests=json.loads((root/'manifests/models.json').read_text());manifests['clip_l14']=dict(repo=alignment['repo'],revision=alignment['revision'],path=str(model),model_sha256=sha(model/'model.safetensors'),text_encoder_exact_equal_to_sd15=True);(root/'manifests/models.json').write_text(json.dumps(manifests,indent=2))
 (root/'runtime/environment.json').write_text(json.dumps(dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','Pillow','numpy']},cuda=torch.version.cuda,cuda_visible_devices='0'),indent=2))
 ns=runpy.run_path(str(source),run_name='__main__');tok=ns['tok'];sd_tok=ns['pipe'].tokenizer;trace=ns['trace'];out=ns['out'];best=ns['best']
 assert tok.get_vocab()==sd_tok.get_vocab();expected=[i for i in range(tok.vocab_size) if i not in tok.all_special_ids and ns['allowed'](tok.decode([i]))];assert ns['vocab_ids']==expected
 assert len(trace)==501 and [r['step'] for r in trace]==list(range(501));assert int(ns['optimizer'].state[ns['logits']]['step'].item())==500
 assert best==max((r for r in trace if r['allowed']),key=lambda r:r['score']);pool=set(expected)
 assert all(len(r['token_ids'])==16 and set(r['token_ids'])<=pool for r in trace)
 errors=[]
 for row in [trace[0],trace[-1],best]:
  ids=torch.tensor([[tok.bos_token_id]+row['token_ids']+[tok.eos_token_id]],device='cuda')
  with torch.no_grad():
   native=F.normalize(ns['clip'].get_text_features(input_ids=ids),dim=-1)
   one=F.one_hot(torch.tensor([expected.index(i) for i in row['token_ids']],device='cuda'),len(expected)).float();custom=ns['text_feature'](one)
  error=float((native-custom).abs().max());errors.append(error);assert error<1e-5
  assert abs(float((native@ns['image_feature'].T).item())-row['score'])<1e-5
 rows=json.loads((out/'comparison.json').read_text());assert [r['name'] for r in rows]==['initial_ids','best_ids','best_text','natural'];assert [len(r['token_ids']) for r in rows[:2]]==[16,16];assert len(rows[2]['token_ids'])<=75
 assert rows[0]['token_ids']==trace[0]['token_ids'] and rows[1]['token_ids']==best['token_ids'];assert rows[2]['token_ids']==sd_tok(best['text'],add_special_tokens=False).input_ids
 olddir=Path('/home/zcy/book_revision_labs_20261022/pseudoword_fullvocab_20261025/pseudoword_fullvocab');oldrows=json.loads((olddir/'comparison.json').read_text());assert rows[0]['token_ids']==oldrows[0]['token_ids']
 from PIL import Image
 pixels_equal={name:Image.open(out/(name+'.png')).tobytes()==Image.open(olddir/(name+'.png')).tobytes() for name in ['initial_ids','natural']};assert all(pixels_equal.values())
 rescored=[]
 with torch.no_grad():
  for row in oldrows:
   ids=torch.tensor([[tok.bos_token_id]+row['token_ids']+[tok.eos_token_id]],device='cuda');text=F.normalize(ns['clip'].get_text_features(input_ids=ids),dim=-1)
   pixels=ns['processor'](images=Image.open(olddir/(row['name']+'.png')).convert('RGB'),return_tensors='pt').pixel_values.to('cuda');image=F.normalize(ns['clip'].get_image_features(pixels),dim=-1)
   rescored.append(dict(origin='B32 optimization',case=row['name'],metric_encoder='CLIP ViT-L/14',text_target_cosine=float((text@ns['image_feature'].T).item()),image_target_cosine=float((image@ns['image_feature'].T).item()),source_original_ids=row['token_ids']))
 for row in rows:rescored.append(dict(origin='L14 optimization',case=row['name'],metric_encoder='CLIP ViT-L/14',text_target_cosine=row['text_target_cosine'],image_target_cosine=row['image_target_cosine'],source_original_ids=row['token_ids']))
 (root/'runtime/cross_encoder_rescore.json').write_text(json.dumps(rescored,indent=2))
 accept=dict(all_passed=True,encoder='CLIP ViT-L/14',text_weight_tensors=len(checks),text_weight_all_exact=True,source_sha256=status['sha256'],total_vocab=tok.vocab_size,allowed_vocab=len(expected),specials_excluded=len(tok.all_special_ids),blacklist_excluded=tok.vocab_size-len(tok.all_special_ids)-len(expected),non_end_word_tokens_retained=sum(not tok.convert_ids_to_tokens(i).endswith('</w>') for i in expected),updates=500,states=501,optimized_id_slots=16,valid_states=sum(r['allowed'] for r in trace),roundtrip_states=sum(tok(r['text'],add_special_tokens=False).input_ids==r['token_ids'] for r in trace),native_feature_max_errors=errors,best_step=best['step'],initial=trace[0]['score'],final=trace[-1]['score'],best=best['score'],generated=4,token_counts=[len(r['token_ids']) for r in rows],output_checker_flags={r['name']:r['output_checker'] for r in rows},same_initial_discrete_ids_as_b32=True,baseline_pixels_equal=pixels_equal,old_b32_results_rescored_with_l14=True)
 (root/'runtime/acceptance.json').write_text(json.dumps(accept,indent=2));status.update(returncode=0,seconds=round(time.time()-start,2))
except Exception as e:status.update(returncode=1,error=repr(e),seconds=round(time.time()-start,2));raise
finally:(root/'runtime/status.json').write_text(json.dumps(status,indent=2))
print('VALIDATION PASSED',json.dumps(accept),flush=True)
