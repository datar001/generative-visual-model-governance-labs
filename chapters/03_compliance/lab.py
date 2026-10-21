from pathlib import Path
import argparse
import json
from filter_core import compare

# 1. 同一套配置固定词表、模型修订号与阈值。
folder = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Compare keyword, text and image filters.')
parser.add_argument('--keyword-only', action='store_true')
parser.add_argument('--images-manifest', type=Path)
parser.add_argument('--device', type=int, default=-1)
parser.add_argument('--out', type=Path, default=Path('results/filter'))
args = parser.parse_args()

# 2. 独立审核风险与安全文本，保留两个过滤器的不同判断。
# 图像另表记录，不把风险提示与任意安全图片强行配对。
summary = compare(folder, args.out, keyword_only=args.keyword_only,
                  image_manifest=args.images_manifest, device=args.device)

# 3. 保存逐样本分数、命中词、失败状态和模型配置。
# 模型未加载或推理失败记为不可用，不能当作未触发。
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(1 if summary['failures'] else 0)
