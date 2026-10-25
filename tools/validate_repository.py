import ast, json, pathlib, re
from render_exercises import render_markdown
ROOT=pathlib.Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'content/curriculum.json').read_text(encoding='utf8'))
assert len(data)==8
questions=answers=0
for c in data:
 folder=ROOT/'chapters'/c['slug']
 assert len(c['questions'])==12
 assert len({q[0] for q in c['questions']})==12
 assert all(q[0] not in {'基础知识','原理分析','实践设计'} for q in c['questions'])
 qtext=(folder/'exercises.md').read_text(encoding='utf8')
 atext=(folder/'answers.md').read_text(encoding='utf8')
 assert qtext==render_markdown(c)
 assert atext==render_markdown(c,True)
 for i,(title,q,a) in enumerate(c['questions'],1):
  ident=f"{c['chapter']}-{i:02}"
  assert f'## {ident} ' in qtext and f'## {ident} ' in atext
  assert q in qtext and a in atext and len(a)>25
  questions+=1;answers+=1
 for lab in c['labs']:
  assert all(lab.get(x) for x in ('why','goals','learn','setup','steps','expected','measured','analysis'))
  assert len(lab['goals'])>=3 and len(lab['learn'])>=3
scripts=[]
for p in ROOT.rglob('*.py'):
 if any(part.startswith('.venv') or part=='.git' for part in p.parts):continue
 ast.parse(p.read_text(encoding='utf8'));scripts.append(str(p.relative_to(ROOT)))
for p in ROOT.rglob('*.md'):
 if '.git' in p.parts:continue
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
  if re.match(r'^(https?://|#|mailto:)',target):continue
  assert (p.parent/target.split('#')[0]).exists(),(str(p),target)
print(json.dumps({'chapters':8,'practices':sum(len(c['labs']) for c in data),
 'questions':questions,'answers':answers,'python_files':len(scripts),
 'local_markdown_links':'ok'},ensure_ascii=False,indent=2))
