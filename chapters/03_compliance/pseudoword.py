from pathlib import Path
import json
from pseudoword_core import run

# 1. 使用已提供的红杯图与固定伪词音节，设置种子和查询预算。
folder = Path(__file__).resolve().parent
out = Path('results/pseudoword_cup')

# 2. 固定本地CLIP，比较等预算随机搜索与逐步改进搜索。
# 每次候选查询均写入日志，缓存命中也计入逻辑预算。
result = run(folder, out)

# 3. 分开报告图文分数、课堂词表结果和实际验证范围。
# 本例没有运行真实安全过滤或文生图，分数提高不代表越狱成功。
print(json.dumps(result, ensure_ascii=False, indent=2))
