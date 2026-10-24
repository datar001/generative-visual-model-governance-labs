from pathlib import Path
import subprocess,os,time,json,hashlib,sys
root=Path('/home/zcy/book_revision_labs_20261022')
work=root/'code_review_20261024';(work/'logs').mkdir(parents=True,exist_ok=True);(work/'isolated_workdir').mkdir(exist_ok=True)
source=root/'src/book_complete_20261024'
orders=['01_governance','02_generation','03_pseudoword','04_detection','05_watermark','06_sld','06_esd','07_fairness','08_privacy','03_filters']
env=os.environ.copy();env.update(LAB_ROOT=str(root),CUDA_VISIBLE_DEVICES='0',TORCH_HOME=str(root/'models/torch'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONPATH='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
records=[]
for name in orders:
 p=source/(name+'.py');start=time.time()
 rec=dict(example=name,status='running',script_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cuda_visible_devices='0',isolated_workdir=True)
 records.append(rec);(work/'status.json').write_text(json.dumps(records,indent=2))
 with (work/'logs'/(name+'.log')).open('w') as log:
  result=subprocess.run([str(root/'env/generation/bin/python'),str(p)],cwd=work/'isolated_workdir',env=env,stdout=log,stderr=subprocess.STDOUT)
 rec.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,elapsed_seconds=time.time()-start)
 (work/'status.json').write_text(json.dumps(records,indent=2))
 print(name,rec['status'],round(rec['elapsed_seconds'],2),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
