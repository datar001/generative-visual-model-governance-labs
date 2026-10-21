# 生成式视觉大模型风险与治理：教学实践与习题

远端：https://github.com/datar001/generative-visual-model-governance-labs

本书独立教学配套仓库。8章各有实践说明、代码、12道多类型习题和逐题参考答案，共96题。章节实践采用“为什么做 → 目标 → 学习收获 → 环境 → 操作 → 预期效果 → 实测证据 → 结果分析”的结构。

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

## 实测状态

历史真实验证：第2章红杯/蓝杯生成、第5章在红杯图上的水印压缩对照。记录和图像在[evidence](evidence/README.md)。本轮第1章离线执行、11个程序入口检查、18项离线测试和第5章水印复测通过，详见docs/validation_20261021.json。第3章已完成8条文本与2张安全图的过滤器实测，详见[evidence](evidence/filter_comparison_20261021/README.md)；风险图像阳性对照未测。第4章CNNSpot已完成两图四条件共8组实测，含生成图漏检记录，见[第4章说明](chapters/04_detection/README.md)。第6章ESD已完成官方权重匹配及CPU低分辨率单步流程检查，见[记录](evidence/esd_pretrained_20261021/README.md)，默认课堂效果未测；SLD、第7、8章模型效果实验未完成实测，不把预期输出当作测得结论。

[实验记录与提交物](docs/experiment_record.md) · [维护与同步](docs/maintenance.md) · [接口依据](docs/sources.md)

本仓库不分发模型权重、授权人脸、令牌或整书PDF。初始远端为私有，学生访问需作者授权；是否公开由作者决定。


第3章另增无害红杯目标的伪词搜索教学类比：本机CPU已完成12组、480次候选查询，约26秒、无LLM API调用。见[实际轨迹与统计](evidence/pseudoword_cup_20261021/README.md)。这组实验没有测试真实安全过滤或文生图，不能据此给出真实越狱成功率。
