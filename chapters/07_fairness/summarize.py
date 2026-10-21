import argparse, csv, collections, json, pathlib
LABELS={'feminine','masculine','uncertain','no_face','multiple','failed','unreviewed'}
def summarize(rows):
 groups={}
 for row in rows:
  key=(row['job'],row['mode'])
  counts=groups.setdefault(key,collections.Counter())
  label=row.get('reviewed_label','').strip() or 'unreviewed'
  if label not in LABELS:raise ValueError('Unknown reviewed_label: '+label)
  counts['total']+=1;counts[label]+=1
 result=[]
 for (job,mode),c in sorted(groups.items()):
  valid=c['feminine']+c['masculine']
  result.append({'job':job,'mode':mode,**{x:c[x] for x in ['total',*sorted(LABELS)]},
   'decidable':valid,'feminine_fraction_all':c['feminine']/c['total'],
   'feminine_fraction_decidable':c['feminine']/valid if valid else None})
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('csv_path');p.add_argument('--output',default='results/fairness/review_summary.json')
 a=p.parse_args()
 with open(a.csv_path,encoding='utf-8-sig',newline='') as f:result=summarize(csv.DictReader(f))
 target=pathlib.Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print('Groups:',len(result),'; saved:',target)
