from pathlib import Path
import os,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folder=Path(os.environ['LAB_ROOT'])/'pseudoword_l14'
trace=json.loads((folder/'search.json').read_text())['trace']
x=[r['step'] for r in trace];y=[r['score'] for r in trace]
best=max((r for r in trace if r['allowed']),key=lambda r:r['score'])
valid=[r['score'] if r['allowed'] else -np.inf for r in trace]
fig,ax=plt.subplots(figsize=(6.3,2.7),layout='constrained')
ax.plot(x,y,color='#7192ab',alpha=.7,linewidth=.7,label='Current discrete candidate')
ax.plot(x,np.maximum.accumulate(valid),color='#b63832',linewidth=1.6,label='Best visited candidate')
ax.scatter([best['step']],[best['score']],s=28,color='#b63832',zorder=3)
ax.annotate(f"Best at {best['step']}: {best['score']:.4f}",(best['step'],best['score']),
    xytext=(-125,-40),textcoords='offset points',fontsize=9,
    arrowprops={'arrowstyle':'->','color':'#b63832'})
ax.set(xlabel='Adam updates',ylabel='Text-target CLIP-L/14 cosine',xlim=(0,500))
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8,frameon=False)
fig.savefig(folder/'optimization_curve.pdf');fig.savefig(folder/'optimization_curve.png',dpi=200)
plt.close(fig)
