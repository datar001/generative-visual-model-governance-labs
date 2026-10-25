# 生成式视觉大模型风险与治理：教学实践与习题

远端：https://github.com/datar001/generative-visual-model-governance-labs

本书独立教学配套仓库。8章各有实践说明、代码、12道多类型习题和逐题参考答案，共96题。章节实践采用“为什么做 → 目标 → 学习收获 → 环境 → 操作 → 实际输出 → 实测证据 → 结果分析”的结构。

## 当前第三章：CLIP全词表减黑名单，16个ID优化500步

[113行最小代码](examples/minimal/03_pseudoword.py)从CLIP 49,408个ID出发，去掉2个特殊ID和536个显式黑名单命中项，保留48,870个候选（含BPE碎片和非英语词元）。500次Adam更新记录501条状态，0卡实跑最佳在408步，代理余弦0.3324。ID条件图像余弦0.7087、解码文本重编码通路0.6531，仍未生成红杯；自然语言参照0.9444。[完整实测及核验](evidence/pseudoword_fullvocab_20261025/README.md) · [章节说明](chapters/03_compliance/README.md)。最佳16个ID重编码为15个，两个生成通路和分数分别保留。旧24词元及64候选属于有日期历史实验。

## 此前全章最小实现与结果对应（2026-10-25）

八章十个实践的目标、操作、代码和结果现已逐项对齐，所有书内图表直接来自当前最小代码的新实跑。未实现的JPEG、PSNR、CLIP评估、人工复核或输出身份评分不再混入案例。十份代码保持简短，统一按完整文件打印。

A800物理0号卡重跑10份文件，共34次生成；123项对应核验通过。[当前最小运行入口](examples/minimal/README.md) · [全部原始证据](evidence/all_practice_alignment_20261025/README.md)。完整大样本实验继续在仓库扩展入口保留。

## 此前第三章单独对齐记录（2026-10-25）

第三章正文现在直接展示最小代码实际输出：过滤器3条文本+3张图；伪词64候选单次搜索+2张生成图。18/31及480/7的大规模对照只保留为仓库扩展。[新实跑证据](evidence/ch3_alignment_20261025/README.md)保留作者当前代码注释及源文件哈希。

## 书内最小完整实现（2026-10-24）

书中改用10份[最小完整实现](examples/minimal/README.md)，第二章从126行减为32行，仍包含导入、SD加载、采样和保存；无需隐藏的项目辅助函数。通用校验、完整函数、批量对照与详细日志留在[完整单文件实现](examples/book/README.md)和[审计实验](experiments/a800/README.md)。

十份最小代码均在A800 0号卡实跑完成，共34次生成；35项输出核验通过，第二章12张图与完整实现逐像素相同。最小案例和书中完整量化实验的分母分开说明，详见[本轮证据](evidence/minimal_20261024/README.md)。

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

先阅读[当前最小实现](examples/minimal/README.md)。第1章只用标准库，不需要显卡；第5章水印也可在CPU运行。第2章的12张输出供后续章节复用，各章README已写明当前代码、实际产物及扩展入口。

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
