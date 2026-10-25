# 实验结果与来源索引

当前源码入口为[随书最小实现](../examples/minimal/README.md)。本目录保存实际执行记录与出版展示资产；不同日期、算法和检查器配置分别解释，代码同步不自动构成新实测。

## 当前书内案例与独立对照

| 章节 | 证据 | 说明 |
|---|---|---|
| 3 | [指定人像目标与Best_text](pseudoword_sensitive_20261025/README.md) | 既有step195记录，目标及文本通路图已遮挡；当前短黑名单源文件未在本次整理中重跑。 |
| 4 | [CNNSpot三图](cnnspot_gan_20261025/README.md) | 真实照片、ProGAN和SD1.5红杯，同权重同阈值。 |
| 6 | [SLD五档](sld_sweep_20261025/README.md) | 输出检查器关闭，固定提示/种子；仅分发敏感区域遮挡版。 |
| 7 | [中性职业提示30张](fairness_neutral_20261025/README.md) | 5职业×6种子，作者逐图人工标签与原始生成来源。 |
| 8 | [IP-Adapter条件保护](privacy_ip_adapter_20261025/README.md) | 正式9张、探索对照和40张扩展；主例输出检查器开启。 |
| 8 | [检查器关闭配对](privacy_checker_off_20261025/README.md) | 复用同一组参考PNG，单独9张；有便携运行入口。 |

## 历史完整实测

- [全章最小实现对齐](all_practice_alignment_20261025/README.md)、[最初最小实现](minimal_20261024/README.md)、[完整单文件实现](complete_code_20261024/README.md)：只证明记录中对应源码版本。
- [A800批量实验](a800_20261022/README.md)：488次生成等历史口径，不是当前书内样本数。
- 第3章：[原投影嵌入红杯](pseudoword_projected_l14_20261025/README.md)、[ST-L14](pseudoword_l14_20261025/README.md)、[B32全词表](pseudoword_fullvocab_20261025/README.md)、[早期对齐](ch3_alignment_20261025/README.md)、[过滤器比较](filter_comparison_20261021/README.md)。
- 第4章：[CNNSpot八组早期结果](detection_20261021/README.md)。
- 第6章：[ESD预训练CPU最小流程](esd_pretrained_20261021/README.md)。
- [早期红杯/蓝杯生成](generation/)与[DCT水印](watermark/)，含固定输入、参数及逐位解码；不覆盖后续版本。

敏感展示图中的黑色补丁是出版编辑，不参与原始指标计算。未遮挡的作者实验原图不随本次代码发布；模型权重、整书PDF及独立审计备份也不进入Git。各目录的SHA-256清单用于核查文件内容，具体实验边界以对应README和manifest为准。
