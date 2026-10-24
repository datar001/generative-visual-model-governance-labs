from pathlib import Path
import tarfile,json,hashlib
root=Path('/home/zcy/book_revision_labs_20261022');work=root/'code_review_20261024';results=root/'book_examples_20261024'
comparison=json.loads((work/'comparison.json').read_text());status=json.loads((work/'status.json').read_text());repeat=json.loads((work/'privacy_repeat.json').read_text())
rows=json.loads((results/'08_privacy/samples.json').read_text())
features=[r for r in rows if r['stage']=='feature'];custom=[r for r in rows if r['stage']=='customization'];checks=[]
def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
check('ten_complete_scripts',len(status)==10 and all(r['status']=='complete' for r in status))
check('all_gpu_visibility_zero',all(r['cuda_visible_devices']=='0' for r in status))
for c in comparison['checks']:
 if c['name']!='08_privacy.reproduces_original':checks.append(c)
check('privacy.bounded_png',all(r['linf']<=8/255+1e-6 for r in features if r['variant']=='protected'))
check('privacy.27_valid_outputs',len(custom)==27 and all(r['evaluation_status']=='ok' and r['detected_faces']==1 and r['output_flag'] is False for r in custom))
check('privacy.three_conditions',all(sum(r['variant']==v for r in custom)==9 for v in ('original','protected','protected_q75')))
check('privacy.same_code_repeat_executed',repeat['returncode']==0 and repeat['script_sha256']==next(r['script_sha256'] for r in status if r['example']=='08_privacy'))
acceptance=dict(all_requirements_passed=all(r['passed'] for r in checks),strict_pixel_comparison_all_passed=comparison['all_passed'],variation='Only PGD protected inputs and their customizations differ. An independent run of the identical final script also matches only 12 of 36 images; fixed seeds do not guarantee bitwise-repeatable GPU gradient updates. All strict mismatches are retained.',checks=checks)
(work/'acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2))
assert acceptance['all_requirements_passed']
index=[]
for p in sorted(results.rglob('*')):
 if p.is_file():index.append(dict(file=str(p.relative_to(results)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(work/'all_new_output_hashes.json').write_text(json.dumps(index,indent=2))
with tarfile.open(work/'review_bundle.tar.gz','w:gz') as t:
 for p in work.glob('*'):
  if p.is_file() and p.name not in ('review_bundle.tar.gz',):t.add(p,arcname='runtime/'+p.name)
 for p in (work/'logs').glob('*'):
  if p.is_file():t.add(p,arcname='runtime/logs/'+p.name)
 for p in results.rglob('*'):
  if p.is_file() and (p.suffix in ('.json','.csv','.txt') or '08_privacy' in p.relative_to(results).parts):t.add(p,arcname='results/'+str(p.relative_to(results)))
 for p in (work/'privacy_repeat/book_examples_20261024/08_privacy').rglob('*'):
  if p.is_file():t.add(p,arcname='privacy_repeat/'+str(p.relative_to(work/'privacy_repeat/book_examples_20261024/08_privacy')))
 for p in (root/'manifests').glob('*'):
  if p.is_file() and p.suffix in ('.json','.txt'):t.add(p,arcname='manifests/'+p.name)
print('ACCEPTED',len(checks),'checks;',len(index),'new output files; privacy bitwise differences explicitly retained')