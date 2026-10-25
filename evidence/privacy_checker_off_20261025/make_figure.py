from pathlib import Path
import json, statistics, hashlib, shutil
from PIL import Image, ImageDraw, ImageFont
S=Path(__file__).resolve().parent
R=S/'raw';rows=json.loads((R/'outputs.json').read_text())
manifest=json.loads((R/'manifest.json').read_text())
old=S.parent/'privacy_opt_20261025/formal'
assert len(rows)==9 and manifest['unflagged_outputs_identical']
for name,sha in manifest['input_sha256'].items():
    assert hashlib.sha256((old/f'{name}.png').read_bytes()).hexdigest()==sha
assert all(r['output_flag'] is None and r['detected_faces']==1 for r in rows)
summary={}
for name in ['original','random','protected']:
    rs=[r for r in rows if r['condition']==name]
    values=[r['identity_cosine'] for r in rs if r['identity_cosine'] is not None]
    summary[name]=dict(total=len(rs),scorable=len(values),identity_cosine_mean=statistics.mean(values))
(S/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
font=lambda size,bold=False:ImageFont.truetype('C:/Windows/Fonts/'+('msyhbd.ttc' if bold else 'msyh.ttc'),size)
TILE=360;LEFT=128;GAP=20;TOP=120;ROWH=410
W=LEFT+3*TILE+2*GAP+25;H=TOP+4*ROWH+90
canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
def centered(text,x,y,w,f,fill='black'):
    box=d.textbbox((0,0),text,font=f);d.text((x+(w-box[2]+box[0])/2,y),text,font=f,fill=fill)
centered('第八章：关闭输出检查器后的配对定制对照',0,8,W,font(30,True))
centered('同一组已保存参考图｜SD 1.5｜适配器0.3｜CFG 5｜30步',0,53,W,font(20),fill='#555555')
names=[('original','原图'),('random','随机噪声（3/255）'),('protected','优化保护（3/255）')]
for j,(name,label) in enumerate(names):centered(label,LEFT+j*(TILE+GAP),86,TILE,font(25,True))
for i,seed in enumerate([None,2026,2027,2028]):
    y=TOP+i*ROWH
    label=['参考输入'] if seed is None else ['定制输出',f'种子{seed}']
    for k,text in enumerate(label):centered(text,0,y+145+k*32,LEFT,font(22,True))
    for j,(name,_) in enumerate(names):
        filename=f'{name}.png' if seed is None else f'custom_{name}_{seed}.png'
        im=Image.open(R/filename).convert('RGB').resize((TILE,TILE),Image.Resampling.LANCZOS)
        x=LEFT+j*(TILE+GAP);canvas.paste(im,(x,y))
        if seed is None:
            cap='原始发布图' if name=='original' else ('同预算随机加噪' if name=='random' else '针对适配器优化')
        else:
            score=next(r['identity_cosine'] for r in rows if r['condition']==name and r['seed']==seed)
            cap=f'输出身份余弦：{score:.3f}'
        centered(cap,x,y+367,TILE,font(21))
centered('九张定制图均为实际生成结果，未使用输出替换或人工遮挡。',0,H-75,W,font(21))
centered('身份余弦比较输出与原始参考；仅作辅助评价，不是保护成功率。',0,H-40,W,font(21),fill='#555555')
canvas.save(S/'comparison_checker_off.png')
print(json.dumps(summary,ensure_ascii=False,indent=2))
