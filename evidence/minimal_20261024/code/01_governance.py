from pathlib import Path
from collections import Counter
import csv
import os

out = Path(os.environ['LAB_ROOT']) / 'minimal_examples_20261024/01_governance'
out.mkdir(parents=True, exist_ok=True)
# 风险标签由人工填写，同一情境可有多个标签。
rows = [
    ['产品广告图', 'authenticity;source', '核对规格与来源', '标明示意用途', '产品审核人', 'pending'],
    ['新闻说明图', 'authenticity;compliance', '核实事实与语境', '合成标识与复核', '编辑', 'pending'],
    ['授权肖像定制', 'privacy;fairness', '核实授权与呈现', '用途核验与复核', '实验负责人', 'pending'],
]
with (out / 'risk_register.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['scenario', 'risks', 'evidence', 'control', 'owner', 'review_status'])
    writer.writerows(rows)
counts = Counter(risk for row in rows for risk in row[1].split(';'))
print('情境数:', len(rows), '风险出现次数:', dict(counts))
