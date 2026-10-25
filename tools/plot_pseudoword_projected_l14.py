from pathlib import Path
import os,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folder=Path(os.environ['LAB_ROOT'])/'pseudoword_projected_l14'
data=json.loads((folder/'search.json').read_text());trace=data['trace'];final=data['final_after_update'];best=data['best']
x=[r['step'] for r in trace];y=[r['score'] for r in trace]
valid=[r['score'] if r['allowed'] else -np.inf for r in trace]
fig,ax=plt.subplots(figsize=(6.3,2.7),layout='constrained')
ax.plot(x,y,color='#7192ab',alpha=.7,linewidth=.7,label='Pre-update projection')
ax.plot(x,np.maximum.accumulate(valid),color='#b63832',linewidth=1.6,label='Best eligible projection')
ax.scatter([best['step']],[best['score']],s=28,color='#b63832',zorder=3)
ax.annotate(f"Best at {best['step']}: {best['score']:.4f}",(best['step'],best['score']),
    xytext=(-125,-40),textcoords='offset points',fontsize=9,
    arrowprops={'arrowstyle':'->','color':'#b63832'})
ax.scatter([final['step']],[final['score']],s=30,facecolors='none',edgecolors='#214a72',zorder=4,label='After update 500 (separate)')
ax.set(xlabel='Completed AdamW updates',ylabel='Text-target CLIP-L/14 cosine',xlim=(0,505))
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=7.5,frameon=False)
fig.savefig(folder/'optimization_curve.pdf');fig.savefig(folder/'optimization_curve.png',dpi=200)
plt.close(fig)
