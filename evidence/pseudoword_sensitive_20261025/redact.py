from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
S=Path(__file__).resolve().parent
R=S/'raw/pseudoword_projected_l14';O=S/'publication';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
boxes={'target.png':[(167,226,346,355),(233,425,306,512)],
       'best_text.png':[(112,276,412,460)]}
manifest=[]
for filename,rectangles in boxes.items():
    source=Image.open(R/filename).convert('RGB');before=np.asarray(source)
    assert source.size==(512,512)
    output=source.copy();draw=ImageDraw.Draw(output);mask=np.zeros((512,512),dtype=bool)
    for x0,y0,x1,y1 in rectangles:
        draw.rectangle((x0,y0,x1-1,y1-1),fill=(0,0,0));mask[y0:y1,x0:x1]=True
    after=np.asarray(output)
    assert np.array_equal(before[~mask],after[~mask])
    assert (after[mask]==0).all()
    destination=O/(Path(filename).stem+'_redacted.png');output.save(destination)
    manifest.append(dict(source=filename,original_sha256=sha(R/filename),published=destination.name,
        published_sha256=sha(destination),rectangles_xyxy_exclusive=rectangles,
        unchanged_outside_rectangles=True,opaque_black_inside_rectangles=True))
search=json.loads((R/'search.json').read_text(encoding='utf8'))
comparison=json.loads((R/'comparison.json').read_text(encoding='utf8'))
best=next(r for r in comparison if r['name']=='best_text')
assert best['text']==search['best']['text']
(O/'best_text.txt').write_text(best['text'],encoding='utf8')
(O/'best_text_escaped.json').write_text(json.dumps({'best_text':best['text']},ensure_ascii=True,indent=2),encoding='utf8')
(O/'redaction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

# Native publication figure; the experiment pixels change only inside the masks above.
W=1120;H=776;canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
cn=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
bold=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',27)
mono=ImageFont.truetype('D:/anaconda/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf',22)
symbol=ImageFont.truetype('C:/Windows/Fonts/seguisym.ttf',22)
def center(text,y,font):
    b=d.textbbox((0,0),text,font=font);d.text(((W-b[2]+b[0])/2,y),text,font=font,fill='black')
center('第三章伪词实践：目标图与 Best_text 对应输出',10,bold)
d.text((30,57),'Best_text（原文）',font=cn,fill='black')
x=30;y=92
for ch in best['text']:
    f=symbol if ord(ch)==0x1f5d3 else mono
    d.text((x,y),ch,font=f,fill='black');x+=d.textlength(ch,font=f)
assert x<W-20
for j,(filename,label) in enumerate([('target_redacted.png','目标图（展示遮挡版）'),('best_text_redacted.png','Best_text 文本通路生成图（展示遮挡版）')]):
    x=30+j*548;canvas.paste(Image.open(O/filename),(x,166))
    b=d.textbbox((0,0),label,font=cn);d.text((x+(512-b[2]+b[0])/2,132),label,font=cn,fill='black')
center('黑块为出版展示后处理；遮挡外像素与原图一致。',690,cn)
center(f"最优更新前状态：195  |  文本—目标余弦：{best['text_target_cosine']:.4f}  |  生成图—目标余弦：{best['image_target_cosine']:.4f}",727,cn)
canvas.save(O/'target_best_text_comparison.png')
print(json.dumps(dict(best_step=search['best']['step'],best_score=search['best']['score'],best_text_score=best['text_target_cosine'],image_score=best['image_target_cosine'],redaction_checks=manifest),ensure_ascii=True,indent=2))
