from pathlib import Path
import os, subprocess, json, time, hashlib
root=Path('/home/zcy/book_revision_labs_20261022')
folder=root/'minimal_review_20261024';folder.mkdir(exist_ok=True)
work=folder/'empty_workdir';work.mkdir(exist_ok=True)
source=root/'src/minimal_20261024'
os.environ.update(CUDA_VISIBLE_DEVICES='0',LAB_ROOT=str(root),TORCH_HOME=str(root/'models/torch'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONPATH='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
order=['01_governance','02_generation','03_filters','03_pseudoword','04_detection','05_watermark','06_sld','06_esd','07_fairness','08_privacy']
records=[]
for name in order:
    p=source/(name+'.py');start=time.time()
    with (folder/(name+'.log')).open('w') as f:
        r=subprocess.run([str(root/'env/generation/bin/python'),str(p)],cwd=work,env=os.environ,stdout=f,stderr=subprocess.STDOUT)
    records.append(dict(script=name+'.py',returncode=r.returncode,seconds=round(time.time()-start,2),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (folder/'status.json').write_text(json.dumps(records,indent=2))
    print(name,r.returncode,flush=True)
print('finished',flush=True)
