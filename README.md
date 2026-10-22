# 生成式视觉大模型风险与治理：教学实践与习题

远端：https://github.com/datar001/generative-visual-model-governance-labs

本书独立教学配套仓库。8章各有实践说明、代码、12道多类型习题和逐题参考答案，共96题。章节实践采用“为什么做 → 目标 → 学习收获 → 环境 → 操作 → 实际输出 → 实测证据 → 结果分析”的结构。

## 按章学习

| 章 | 材料 | 实践 | 习题 / 答案 |
|---|---|---|---|
| 1 | [风险与治理概述](chapters/01_overview/README.md) | 1个小节 | [12题](chapters/01_overview/exercises.md) / [参考答案](chapters/01_overview/answers.md) |
| 2 | [生成式视觉大模型基础](chapters/02_basics/README.md) | 1个小节 | [12题](chapters/02_basics/exercises.md) / [参考答案](chapters/02_basics/answers.md) |
| 3 | [内容合规性漏洞](chapters/03_compliance/README.md) | 2个小节 | [12题](chapters/03_compliance/exercises.md) / [参考答案](chapters/03_compliance/answers.md) |
| 4 | [生成图像鉴伪](chapters/04_detection/README.md) | 1个小节 | [12题](chapters/04_detection/exercises.md) / [参考答案](chapters/04_detection/answers.md) |
| 5 | [图像水印](chapters/05_watermark/README.md) | 1个小节 | [12题](chapters/05_watermark/exercises.md) / [参考答案](chapters/05_watermark/answers.md) |
| 6 | [安全防御](chapters/06_defense/README.md) | 2个小节 | [12题](chapters/06_defense/exercises.md) / [参考答案](chapters/06_defense/answers.md) |
| 7 | [公平性去偏](chapters/07_fairness/README.md) | 1个小节 | [12题](chapters/07_fairness/exercises.md) / [参考答案](chapters/07_fairness/answers.md) |
| 8 | [肖像隐私保护](chapters/08_privacy/README.md) | 1个小节 | [12题](chapters/08_privacy/exercises.md) / [参考答案](chapters/08_privacy/answers.md) |

## 从哪里开始

不需要显卡即可先完成第1章。仓库根目录执行：

```powershell
python tools/run.py 01 --dry-run
python tools/run.py 01
```

准备其他实验前，阅读[环境配置](docs/environment.md)，再进入各章README。第5章水印也可独立用CPU运行。第2章生成结果可供后续章节使用，但各章输入依赖已分别列明。

## 2026-10-22 A800实测

八章10个实践已完成。主实验含488次模型生成、18条文本及31张图像审核、60次CNNSpot推理、216条水印条件记录。全部检查器替换、不确定与未复核状态保留。第8章正式27张定制图均为单人脸可判读；构图预实验和六组正常输入参数诊断单独归档。

- [完整运行与环境说明](experiments/a800/README.md)
- [实际统计与轻量证据](evidence/a800_20261022/README.md)
- [54项完整证据检查](evidence/a800_20261022/evidence_validation.json)
- [结果汇总](evidence/a800_20261022/analysis_summary.json)

本轮21项离线回归测试通过。全部原始PNG与运行资料另有完整结果包，远端原始实验保存在 `/home/zcy/book_revision_labs_20261022`。仓库保存代码、固定样例、完整联系表与逐样本分数；模型权重和整书PDF不进入Git。

第三章伪词是无害红杯目标类比，已追加实际生成验证，不报告真实风险绕过率；图像过滤没有风险阳性对照。SLD使用非露骨成人场景，不估计色情抑制率；公平性自动外观标签不冒充真实身份或人工审核；隐私代理分数降低不等于通用防护成功。

[实验记录与提交物](docs/experiment_record.md) · [维护与同步](docs/maintenance.md) · [接口依据](docs/sources.md)

远端当前为私有，学生在线访问需作者授予权限或决定公开。历史2026-10-21证据保留在对应有日期目录中。
