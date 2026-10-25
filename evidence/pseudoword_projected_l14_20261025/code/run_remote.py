from pathlib import Path
import os,json,time,hashlib,runpy,platform,importlib.metadata,sys
os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='')
root=Path('/home/zcy/book_revision_labs_20261022/pseudoword_projected_l14_20261025');os.environ['LAB_ROOT']=str(root)
import torch
from safetensors import safe_open
from PIL import Image
from reference_parity import capture_adamw_steps,reference_trajectory,compare_trajectories,check_native_score_gradient
start=time.perf_counter();torch.set_num_threads(4)
source=root/'code/03_pseudoword.py';status=dict(script=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gpu='physical GPU 0',device=torch.cuda.get_device_name(0))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 assert torch.cuda.device_count()==1
 manifest=json.loads((root/'manifests/models.json').read_text());l14=Path(manifest['clip_l14']['path']);sd=Path(manifest['sd15']['path'])
 assert sha(l14/'model.safetensors')=='a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb'
 assert sha(root/'inputs/pseudoword_target.png')=='a987ed93305d449f4d8ef202935d745a25c53063bb10d34841ab4636fca148a3'
 with safe_open(l14/'model.safetensors',framework='pt',device='cpu') as full,safe_open(sd/'text_encoder/model.safetensors',framework='pt',device='cpu') as target:
  fk={k for k in full.keys() if k.startswith('text_model.') and not k.endswith('.position_ids')};sk={k for k in target.keys() if not k.endswith('.position_ids')};assert fk==sk;checks=[]
  for k in sorted(fk):
   a=full.get_tensor(k);b=target.get_tensor(k);checks.append(dict(key=k,shape=list(a.shape),equal=torch.equal(a.float(),b.float()),max_abs_difference=float((a.float()-b.float()).abs().max())))
 alignment=dict(text_tensors=len(checks),all_equal=all(r['equal'] for r in checks),max_abs_difference=max(r['max_abs_difference'] for r in checks),checks=checks);assert alignment['all_equal'] and alignment['text_tensors']==196
 (root/'runtime/text_encoder_alignment.json').write_text(json.dumps(alignment,indent=2))
 (root/'runtime/environment.json').write_text(json.dumps(dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','Pillow','numpy']},cuda=torch.version.cuda,cuda_visible_devices='0'),indent=2))
 print('WEIGHT ALIGNMENT PASSED: 196 tensors exactly equal',flush=True)
 with capture_adamw_steps(5) as captured:
  ns=runpy.run_path(str(source),run_name='__main__')
 trace=ns['trace'];out=ns['out'];tok=ns['tok'];best=ns['best'];final=ns['final'];rows=json.loads((out/'comparison.json').read_text())
 for record,row in zip(captured,trace):record.update(projected_ids=row['token_ids'],score=row['score'])
 ref=reference_trajectory(ns['table'],captured[0]['parameter_before'],ns['text_similarity'],ns['vocab_ids'],steps=5)
 parity=compare_trajectories(captured,ref);(root/'runtime/original_core_parity.json').write_text(json.dumps(parity,indent=2));assert parity['all_passed'],parity
 native={name:check_native_score_gradient(ns['clip'],tok,ns['image_feature'],row['token_ids'],ns['text_similarity']) for name,row in [('initial',trace[0]),('best',best),('final',final)]}
 (root/'runtime/native_score_gradient.json').write_text(json.dumps(native,indent=2));assert all(r['all_passed'] for r in native.values()),native
 assert len(trace)==500 and [r['step'] for r in trace]==list(range(500)) and int(ns['optimizer'].state[ns['prompt_embeds']]['step'].item())==500
 assert best==max((r for r in trace if r['allowed']),key=lambda r:r['score'])
 expected=[i for i in range(tok.vocab_size) if i not in tok.all_special_ids and ns['allowed'](tok.decode([i]))];assert expected==ns['vocab_ids'];pool=set(expected)
 assert all(len(r['token_ids'])==16 and set(r['token_ids'])<=pool for r in trace+[final]);assert ns['pipe'].tokenizer.get_vocab()==tok.get_vocab()
 assert [r['name'] for r in rows]==['initial_ids','best_ids','best_text','natural'] and len(rows[2]['token_ids'])<=75
 assert rows[0]['token_ids']==trace[0]['token_ids'] and rows[1]['token_ids']==best['token_ids'] and rows[2]['token_ids']==rows[1]['decoded_retokenized_ids']
 torch.save(dict(initial_parameter=captured[0]['parameter_before'].cpu(),final_parameter=ns['prompt_embeds'].detach().cpu(),optimizer=ns['optimizer'].state_dict()),root/'runtime/optimizer_state.pt')
 oldroot=Path('/home/zcy/book_revision_labs_20261022/pseudoword_l14_20261025');oldrows=json.loads((oldroot/'pseudoword_l14/comparison.json').read_text());rescored=[]
 with torch.no_grad():
  for row in oldrows:
   ids=torch.tensor([[tok.bos_token_id]+row['token_ids']+[tok.eos_token_id]],device='cuda');txt=ns['clip'].get_text_features(input_ids=ids);txt=txt/txt.norm(dim=-1,keepdim=True)
   px=ns['processor'](images=Image.open(oldroot/'pseudoword_l14'/(row['name']+'.png')).convert('RGB'),return_tensors='pt').pixel_values.to('cuda');img=ns['clip'].get_image_features(px);img=img/img.norm(dim=-1,keepdim=True)
   rescored.append(dict(origin='historical logits ST-softmax',case=row['name'],text_target_cosine=float((txt@ns['image_feature'].T).item()),image_target_cosine=float((img@ns['image_feature'].T).item())))
 for row in rows:rescored.append(dict(origin='original projected embedding core',case=row['name'],text_target_cosine=row['text_target_cosine'],image_target_cosine=row['image_target_cosine']))
 (root/'runtime/cross_method_rescore.json').write_text(json.dumps(rescored,indent=2))
 natural_equal=Image.open(out/'natural.png').tobytes()==Image.open(oldroot/'pseudoword_l14/natural.png').tobytes()
 acceptance=dict(all_passed=True,source_sha256=status['sha256'],algorithm='original continuous projected embedding AdamW core',upstream_commit='21eb35f16697529388a43bf1504089eccc6bc432',original_nn_reference_steps=5,core_parity_passed=parity['all_passed'],text_weight_tensors=196,text_weights_equal=True,encoder='CLIP ViT-L/14',total_vocab=tok.vocab_size,allowed_vocab=len(expected),specials_excluded=len(tok.all_special_ids),blacklist_excluded=tok.vocab_size-len(tok.all_special_ids)-len(expected),optimized_id_slots=16,optimized_continuous_parameters=ns['prompt_embeds'].numel(),updates=500,training_projection_states=500,final_after_update_separately_recorded=True,valid_states=sum(r['allowed'] for r in trace),roundtrip_states=sum(tok(r['text'],add_special_tokens=False).input_ids==r['token_ids'] for r in trace),best_step=best['step'],initial=trace[0]['score'],best=best['score'],final=final['score'],optimization_seconds=ns['optimization_seconds'],timing_note='CUDA-synchronized 500-update loop; includes first-five-step verification snapshots, excludes SD loading/generation and post-run parity',token_counts=[len(r['token_ids']) for r in rows],output_checker_flags={r['name']:r['output_checker'] for r in rows},same_natural_baseline_pixels_as_previous_l14=natural_equal,sd_generation=dict(seed=2026,steps=30,guidance_scale=7.5,height=512,width=512))
 (root/'runtime/acceptance.json').write_text(json.dumps(acceptance,indent=2));status.update(returncode=0,seconds=round(time.perf_counter()-start,2))
except Exception as exc:status.update(returncode=1,error=repr(exc),seconds=round(time.perf_counter()-start,2));raise
finally:(root/'runtime/status.json').write_text(json.dumps(status,indent=2))
print('VALIDATION PASSED',json.dumps(acceptance),flush=True)
