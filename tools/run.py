import argparse, ast, datetime, hashlib, importlib.metadata as md
import json, os, pathlib, subprocess, sys, uuid
ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = {
 '01': {'lab': ('01_governance.py', [])},
 '02': {'lab': ('02_generation.py', [])},
 '03': {'lab': ('03_filters.py', []), 'pseudoword': ('03_pseudoword.py', [])},
 '04': {'lab': ('04_detection.py', [])},
 '05': {'lab': ('05_watermark.py', [])},
 '06': {'sld': ('06_sld.py', []), 'esd': ('06_esd.py', [])},
 '07': {'lab': ('07_fairness.py', [])},
 '08': {'features': ('08_privacy.py', ['--features-only']),
        'customize': ('08_privacy.py', [])}}

def main():
 p=argparse.ArgumentParser(description='Run one chapter example from the repository root.')
 p.add_argument('chapter',choices=CONFIG);p.add_argument('--case')
 p.add_argument('--dry-run',action='store_true')
 args=p.parse_args();cases=CONFIG[args.chapter]
 case=args.case or next(iter(cases))
 if case not in cases:p.error('Cases for this chapter: '+', '.join(cases))
 filename,extra=cases[case]
 target=ROOT/'examples/minimal'/filename
 ast.parse(target.read_text(encoding='utf8'))
 command=[sys.executable,str(target),*extra]
 if args.dry_run:
  print(json.dumps({'chapter':args.chapter,'case':case,'script':str(target),
                    'working_directory':str(ROOT),'command':command},ensure_ascii=False,indent=2))
  return 0
 if not os.environ.get('LAB_ROOT'):
  p.error('Set LAB_ROOT to your experiment directory; see examples/minimal/README.md.')
 run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
 out=ROOT/'results/runs'/run_id;out.mkdir(parents=True)
 packages={}
 for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','opencv-python','Pillow','facenet-pytorch'):
  try:packages[name]=md.version(name)
  except md.PackageNotFoundError:packages[name]=None
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
 record={'run_id':run_id,'chapter':args.chapter,'case':case,'command':command,
  'python':sys.version,'packages':packages,'code_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
  'git_commit':commit.stdout.strip() if commit.returncode==0 else None,
  'status':'running','outputs_root':os.environ['LAB_ROOT'],'validation':'Process success does not prove model or learning outcomes.'}
 manifest=out/'run.json';manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
 print('Run log:',out,flush=True)
 try:
  with (out/'stdout.txt').open('w',encoding='utf8') as stdout,(out/'stderr.txt').open('w',encoding='utf8') as stderr:
   result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
  code=result.returncode
 except KeyboardInterrupt:
  code=130
 finally:
  record['exit_code']=locals().get('code',1)
  record['status']='completed' if record['exit_code']==0 else 'failed'
  manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
 print('Exit:',record['exit_code'],'; inspect the outputs and log before drawing conclusions.')
 return record['exit_code']
if __name__=='__main__':raise SystemExit(main())
