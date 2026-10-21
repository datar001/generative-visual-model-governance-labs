from pathlib import Path
import csv, json
from collections import Counter
# 读取教学情境；风险标签由人工给定，不作自动判定。
cases = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))
# 限定五类风险，防止标签拼写错误影响统计。
allowed = {'authenticity', 'source', 'compliance', 'fairness', 'privacy'}
out = Path('results/governance')
out.mkdir(parents=True, exist_ok=True)
rows = []
counts = Counter()
# 逐条核对标签，并保留证据、措施和责任人。
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
# 用带编码标记的 CSV 保存清单，便于表格软件读取。
with (out/'risk_register.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
# 风险出现次数可大于情境数，同一情境可有多类风险。
print(json.dumps({'cases': len(rows), 'risk_mentions': dict(counts)},
                 ensure_ascii=False, indent=2))
