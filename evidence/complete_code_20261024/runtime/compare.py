from pathlib import Path
import json,hashlib,csv,math
root=Path('/home/zcy/book_revision_labs_20261022');old=root/'results';new=root/'book_examples_20261024';work=root/'code_review_20261024'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,passed,detail):checks.append(dict(name=name,passed=bool(passed),detail=detail))
status=json.loads((work/'status.json').read_text())
check('ten_complete_files_run',len(status)==10 and all(x['status']=='complete' for x in status),status)
expected={'02_generation':12,'03_pseudoword':7,'04_detection':60,'05_watermark':216,'06_sld':18,'06_esd':24,'07_fairness':400,'08_privacy':45,'03_filters':49}
for chapter,n in expected.items():
 a=json.loads((old/chapter/'samples.json').read_text());b=json.loads((new/chapter/'samples.json').read_text())
 check(chapter+'.rows',len(a)==len(b)==n,dict(old=len(a),new=len(b),expected=n))
 mismatches=[];matched=0
 for i,(x,y) in enumerate(zip(a,b)):
  # Scope/audit paths intentionally differ; core values must retain their meaning.
  for key in ('id','stage','kind','condition','variant','seed','mode','image','output_flag','automatic_label','reviewed_label','prediction','evaluation_status','decoded_bits','bit_accuracy','fake_score','cosine_to_original','psnr_db','linf','scores','keyword','text'):
   if key not in x or key not in y:continue
   v,w=x[key],y[key]
   if isinstance(v,float) and isinstance(w,(float,int)):same=math.isclose(v,w,abs_tol=1e-6,rel_tol=1e-6)
   else:same=v==w
   if not same:mismatches.append(dict(row=i,key=key,old=v,new=w))
  if 'image' in x and isinstance(x['image'],str) and x['image']==y.get('image'):
   p=old/chapter/x['image'];q=new/chapter/y['image']
   if p.is_file() and q.is_file():
    matched+=1
    if sha(p)!=sha(q):mismatches.append(dict(row=i,key='image_sha256',file=x['image']))
 check(chapter+'.reproduces_original',not mismatches,dict(compared_images=matched,mismatches=mismatches))
# Exact search/query trajectories and chapter-1 risk registers must also agree.
for filename in ('trace.csv','summary.csv'):
 check('03_pseudoword.'+filename,sha(old/'03_pseudoword/search'/filename)==sha(new/'03_pseudoword/search'/filename),filename)
check('01_governance.csv',sha(old/'01_governance/risk_register.csv')==sha(new/'01_governance/risk_register.csv'),'same three scenarios, seven fields')
result=dict(all_passed=all(c['passed'] for c in checks),checks=checks)
(work/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))