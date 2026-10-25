from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S=Path(__file__).resolve().parent
raw=S.parent/'sld_sweep_20261025'
m=json.loads((raw/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for row in m['outputs']:
    src=raw/row['file'];assert sha(src)==row['sha256']
    im=Image.open(src).convert('RGB');before=np.array(im)
    boxes=[] if row['mode']=='max' else [[145,280,440,512]]
    mask=np.zeros(before.shape[:2],dtype=bool)
    draw=ImageDraw.Draw(im)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle((x0,y0,x1-1,y1-1),fill=(0,0,0))
        mask[y0:y1,x0:x1]=True
    after=np.array(im)
    assert np.array_equal(before[~mask],after[~mask])
    assert np.all(after[mask]==0)
    dest=S/(row['mode']+'_redacted.png');im.save(dest)
    records.append(dict(mode=row['mode'],source_sha256=sha(src),display_sha256=sha(dest),boxes_xyxy_exclusive=boxes,unchanged_outside_boxes=True,opaque_black_inside_boxes=True))

fig,axes=plt.subplots(1,5,figsize=(18,4.6),dpi=180)
fig.subplots_adjust(left=.012,right=.988,bottom=.17,top=.83,wspace=.035)
for ax,row in zip(axes,m['outputs']):
    ax.imshow(Image.open(S/(row['mode']+'_redacted.png')),interpolation='none')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(row['mode'].upper(),fontsize=15,fontweight='bold',pad=10)
    c=row['config']
    ax.set_xlabel(f"SLD scale: {c['sld_guidance_scale']:g}\nWarm-up: {c.get('sld_warmup_steps','-')}",fontsize=10,labelpad=8)
fig.suptitle('SLD preset comparison | Same prompt and seed',fontsize=18,y=.98)
fig.text(.5,.91,'SD 1.5  |  seed 2026  |  30 steps  |  CFG 7.5  |  output checker disabled',ha='center',fontsize=11,color='#444444')
fig.text(.5,.035,'Black rectangles are editorial redactions, not model outputs. Presets change multiple parameters.',ha='center',fontsize=10,color='#555555')
fig.savefig(S/'sld_progression_redacted.png',dpi=180,facecolor='white')
(S/'redaction_manifest.json').write_text(json.dumps(dict(method='Opaque black rectangles only; no generative image editing',coordinate_convention='x0,y0 inclusive; x1,y1 exclusive',images=records),indent=2))
print('Five display assets saved; original pixels outside rectangles verified unchanged.')
