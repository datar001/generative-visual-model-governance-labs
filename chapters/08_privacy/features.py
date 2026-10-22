"""统一入口：调用已实测实现，完整参数、图像与记录保存到 LAB_ROOT。"""
from pathlib import Path
import sys
# 同时支持教学仓库和随书工程，避免维护两份实验逻辑。
candidates=[p/relative for p in Path(__file__).resolve().parents
            for relative in ('experiments/a800','practice/a800')]
folder=next((p for p in candidates if (p/'common.py').is_file()),None)
if folder is None:raise SystemExit('缺少配套实验实现，请使用完整教学仓库或随书工程。')
sys.path.insert(0,str(folder))
# 先单独验证扰动和特征；另一个入口再完成真实适配器生成。
sys.argv.append('--features-only')
from privacy import main
if __name__=="__main__":main()
