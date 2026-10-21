from pathlib import Path
import csv, json
from collections import Counter
cases = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))
allowed = {'authenticity', 'source', 'compliance', 'fairness', 'privacy'}
out = Path('results/governance')
out.mkdir(parents=True, exist_ok=True)
rows = []
counts = Counter()
for item in cases:
    risks = item['risks']
    if not risks or set(risks)-allowed:
        raise ValueError('Unknown or empty risk labels: '+item['id'])
    counts.update(risks)
    rows.append({
        'id': item['id'], 'scenario': item['scenario'],
        'risks': ';'.join(risks), 'evidence': item['evidence'],
        'control': item['control'], 'owner': item['owner'],
        'review_status': item.get('review_status', 'pending')})
with (out/'risk_register.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'cases': len(rows), 'risk_mentions': dict(counts)},
                 ensure_ascii=False, indent=2))
