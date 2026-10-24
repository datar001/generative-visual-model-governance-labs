from pathlib import Path
import json,hashlib,zipfile,os,math
import numpy as np
from PIL import Image
r=Path('/home/zcy/book_revision_labs_20261022');source=r/'src/minimal_20261024';review=r/'minimal_review_20261024';out=r/'minimal_examples_20261024'
checks=[]
def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
status=json.loads((review/'status.json').read_text())
check('ten_script_exit_codes',len(status)==10 and all(x['returncode']==0 for x in status))
for item in status:
 check('executed_source_hash_'+item['script'],hashlib.sha256((source/item['script']).read_bytes()).hexdigest()==item['sha256'])
for p in sorted((out/'02_generation').glob('*.png')):
 q=r/'book_examples_20261024/02_generation'/p.name
 check('generation_pixels_'+p.stem,np.array_equal(np.array(Image.open(p)),np.array(Image.open(q))))
rows=json.loads((out/'03_filters/decisions.json').read_text())
check('three_text_three_image',len(rows)==6 and sum('text' in x for x in rows)==3)
check('filter_probability_range',all(0<=x['nsfw_score']<=1 for x in rows))
pseudo=json.loads((out/'03_pseudoword/search.json').read_text())
check('unique_benign_64_candidates',len(pseudo['prompts'])==len(set(pseudo['prompts']))==64 and all(not {'red','cup'} & set(p.split()) for p in pseudo['prompts']))
check('finite_clip_scores',all(math.isfinite(x) for x in pseudo['scores']))
detection=json.loads((out/'04_detection/scores.json').read_text())
check('two_cnnspot_scores',len(detection)==2 and all(0<=x['fake_score']<=1 for x in detection))
water=json.loads((out/'05_watermark/decoded.json').read_text())
check('watermark_png_and_jpeg',len(water)==3 and all(x['bit_accuracy']==1 for x in water[:2]),water)
fair=json.loads((out/'07_fairness/samples.json').read_text())
check('fairness_12_complete_records',len(fair)==12 and len({x['image'] for x in fair})==12)
check('fairness_unreviewed_retained',all(x['reviewed']=='unreviewed' for x in fair))
check('fairness_legal_labels',all(x['label'] in ['feminine','masculine','uncertain','review'] for x in fair))
x=np.array(Image.open(out/'08_privacy/original.png')).astype(int);y=np.array(Image.open(out/'08_privacy/protected.png')).astype(int)
linf=int(np.max(np.abs(x-y)))
check('privacy_saved_png_budget',linf<=8,dict(maximum_integer_pixel_difference=linf))
privacy=json.loads((out/'08_privacy/feature_score.json').read_text())
check('privacy_input_feature_finite',math.isfinite(privacy['protected_input_cosine']) and -1<=privacy['protected_input_cosine']<=1,privacy)
counts={k:len(list((out/k).glob('*.png'))) for k in ['02_generation','03_pseudoword','06_sld','06_esd','07_fairness','08_privacy']}
check('actual_generation_file_counts',counts==dict(zip(['02_generation','03_pseudoword','06_sld','06_esd','07_fairness','08_privacy'],[12,2,2,4,12,4])),counts)
result=dict(all_passed=all(x['passed'] for x in checks),checks=checks,gpu='physical GPU 0 only',generation_attempts=34,notes='Minimal examples are separately executed introductory subsets, not a rerun of the 488-attempt full study. Privacy measures only input feature cosine; no output face-similarity metric is claimed here.')
(review/'acceptance.json').write_text(json.dumps(result,indent=2))
with zipfile.ZipFile(review/'minimal_runtime_20261024.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(review.glob('*.json')):z.write(p,'runtime/'+p.name)
 for p in sorted(review.glob('*.log')):z.write(p,'runtime/'+p.name)
 for p in sorted(out.rglob('*')):
  if p.is_file():z.write(p,'outputs/'+p.relative_to(out).as_posix())
 for p in sorted(source.glob('*.py')):z.write(p,'code/'+p.name)
print(json.dumps(dict(all_passed=result['all_passed'],checks=len(checks),status=status,counts=counts,privacy=privacy,linf=linf),indent=2))
