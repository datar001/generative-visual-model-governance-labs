import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot the benign cup-matching trace.')
parser.add_argument('--results', type=Path, default=Path('results/pseudoword_cup'))
parser.add_argument('--output', type=Path)
parser.add_argument('--pdf', type=Path)
args = parser.parse_args()
with (args.results/'trace.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
result = json.loads((args.results/'result.json').read_text(encoding='utf-8'))
fig, ax = plt.subplots(figsize=(6.8,3.0))
for mode, color, label in [('random','#3b63a7','Random'),('improve','#c4772e','Local search')]:
    means, lows, highs = [], [], []
    for query in range(1,65):
        values = [float(x['best_score']) for x in rows
                  if x['mode']==mode and x['budget']=='64' and int(x['query'])==query]
        means.append(statistics.mean(values));lows.append(min(values));highs.append(max(values))
    ax.plot(range(1,65), means, color=color, label=label, lw=1.7)
    ax.fill_between(range(1,65), lows, highs, color=color, alpha=.12)
ax.axhline(result['natural_description_score'], color='#555555', ls='--', lw=1,
           label='Natural description')
ax.set(xlabel='Candidate queries', ylabel='Best CLIP cosine', xlim=(1,64), ylim=(.18,.34))
ax.grid(alpha=.18);ax.legend(loc='center right', frameon=False, fontsize=8)
fig.tight_layout()
output = args.output or args.results/'convergence.png'
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=170)
if args.pdf:
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf)
plt.close(fig)
