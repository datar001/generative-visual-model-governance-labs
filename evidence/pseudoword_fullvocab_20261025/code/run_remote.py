from pathlib import Path
import os,json,hashlib,runpy,time,platform,importlib.metadata,unicodedata
os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='')
root=Path('/home/zcy/book_revision_labs_20261022/pseudoword_fullvocab_20261025');os.environ['LAB_ROOT']=str(root)
import torch
import torch.nn.functional as F
source=root/'code/03_pseudoword.py';start=time.time();status=dict(script=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gpu='physical GPU 0',device=torch.cuda.get_device_name(0))
assert torch.cuda.device_count()==1
(root/'runtime/environment.json').write_text(json.dumps(dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','Pillow','numpy']},cuda=torch.version.cuda,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),indent=2))
try:
 ns=runpy.run_path(str(source),run_name='__main__');tok=ns['tok'];sd=ns['pipe'].tokenizer;trace=ns['trace'];out=ns['out']
 assert tok.get_vocab()==sd.get_vocab()
 expected=[i for i in range(tok.vocab_size) if i not in tok.all_special_ids and ns['allowed'](tok.decode([i]))]
 assert ns['vocab_ids']==expected and len(expected)>45000
 assert len(trace)==501 and [r['step'] for r in trace]==list(range(501))
 assert int(ns['optimizer'].state[ns['logits']]['step'].item())==500
 valid=[r for r in trace if r['allowed']];assert ns['best']==max(valid,key=lambda r:r['score'])
 for row in trace:
  assert len(row['token_ids'])==16 and set(row['token_ids'])<=set(expected)
  assert row['allowed']==(not any(w in unicodedata.normalize('NFKC',row['text']).casefold() for w in ns['blacklist']))
 assert trace[0]['text'].strip()==ns['initial_text'] and trace[0]['allowed']
 errors=[]
 for row in [trace[0],trace[-1],ns['best']]:
  ids=torch.tensor([[tok.bos_token_id]+row['token_ids']+[tok.eos_token_id]],device='cuda')
  with torch.no_grad():
   feat=F.normalize(ns['clip'].get_text_features(input_ids=ids),dim=-1)
   one=F.one_hot(torch.tensor([expected.index(i) for i in row['token_ids']],device='cuda'),len(expected)).float()
   custom=ns['text_feature'](one)
  error=float((feat-custom).abs().max());errors.append(error);assert error<1e-5
  assert abs(float((feat@ns['image_feature'].T).item())-row['score'])<1e-5
 rows=json.loads((out/'comparison.json').read_text());assert [r['name'] for r in rows]==['initial_ids','best_ids','best_text','natural']
 assert rows[0]['token_ids']==trace[0]['token_ids'] and rows[1]['token_ids']==ns['best']['token_ids']
 assert rows[2]['token_ids']==sd(ns['best']['text'],add_special_tokens=False).input_ids
 for r in rows[:3]:assert ns['allowed'](r['text'])
 old=json.loads(Path('/home/zcy/book_revision_labs_20261022/pseudoword16_500_20261025/pseudoword16_500/search.json').read_text())
 assert trace[0]['token_ids']==old['trace'][0]['token_ids'] and abs(trace[0]['score']-old['trace'][0]['score'])<1e-5
 acceptance=dict(all_passed=True,total_vocab=tok.vocab_size,allowed_vocab=len(expected),specials_excluded=len(tok.all_special_ids),blacklist_ids_excluded=tok.vocab_size-len(tok.all_special_ids)-len(expected),non_end_word_tokens_retained=sum(not tok.convert_ids_to_tokens(i).endswith('</w>') for i in expected),gradient_updates=500,visited_states=501,optimized_id_slots=16,valid_decoded_states=len(valid),invalid_decoded_states=501-len(valid),roundtrip_equal_states=sum(tok(r['text'],add_special_tokens=False).input_ids==r['token_ids'] for r in trace),native_feature_max_errors=errors,best_step=ns['best']['step'],initial=trace[0]['score'],final=trace[-1]['score'],best=ns['best']['score'],unfiltered_max=max(r['score'] for r in trace),same_initial_discrete_ids_as_24token=True,generated=4,output_checker_flags={r['name']:r['output_checker'] for r in rows},source_sha256=status['sha256'])
 (root/'runtime/acceptance.json').write_text(json.dumps(acceptance,indent=2));status.update(returncode=0,seconds=round(time.time()-start,2))
except Exception as e:
 status.update(returncode=1,error=repr(e),seconds=round(time.time()-start,2));raise
finally:(root/'runtime/status.json').write_text(json.dumps(status,indent=2))
print('VALIDATION PASSED',json.dumps(acceptance),flush=True)
