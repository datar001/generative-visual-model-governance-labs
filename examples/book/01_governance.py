"""书内完整示例；固定环境和输入见本节说明，不依赖本仓库的Python辅助模块。"""
import os

# A800实验只用物理0号卡；须在导入torch前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import json
import csv
from collections import Counter

ROOT = Path(os.environ.get("LAB_ROOT", ".")).resolve()
# 与2026-10-22的原始实测目录分开，避免覆盖书中已有证据。
RESULTS = ROOT / "book_examples_20261024"


# 读取教学情境；风险标签由人工给定，不作自动判定。
cases = [
    {
        "id": "demo-ad",
        "scenario": "生成产品广告图",
        "risks": ["authenticity", "source"],
        "evidence": "待核对产品规格与素材来源",
        "control": "核实实物能力并标明示意用途",
        "owner": "产品与内容审核人",
        "review_status": "pending",
    },
    {
        "id": "demo-news",
        "scenario": "新闻说明配图",
        "risks": ["authenticity", "compliance"],
        "evidence": "待核实事件事实与传播语境",
        "control": "事实核查、合成标识及人工复核",
        "owner": "编辑与审核人",
        "review_status": "pending",
    },
    {
        "id": "demo-portrait",
        "scenario": "授权肖像定制演示",
        "risks": ["privacy", "fairness"],
        "evidence": "待核实授权范围与样本呈现",
        "control": "按用途核验授权、复核群体呈现",
        "owner": "实验与权益负责人",
        "review_status": "pending",
    },
]
# 限定五类风险，防止标签拼写错误影响统计。
allowed = {"authenticity", "source", "compliance", "fairness", "privacy"}
out = RESULTS / "01_governance"
out.mkdir(parents=True, exist_ok=True)
rows = []
counts = Counter()
# 逐条核对标签，并保留证据、措施和责任人。
for item in cases:
    risks = item["risks"]
    if not risks or set(risks) - allowed:
        raise ValueError("Unknown or empty risk labels: " + item["id"])
    counts.update(risks)
    rows.append(
        {
            "id": item["id"],
            "scenario": item["scenario"],
            "risks": ";".join(risks),
            "evidence": item["evidence"],
            "control": item["control"],
            "owner": item["owner"],
            "review_status": item.get("review_status", "pending"),
        }
    )
# 用带编码标记的 CSV 保存清单，便于表格软件读取。
with (out / "risk_register.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
# 风险出现次数可大于情境数，同一情境可有多类风险。
print(
    json.dumps(
        {"cases": len(rows), "risk_mentions": dict(counts)},
        ensure_ascii=False,
        indent=2,
    )
)
