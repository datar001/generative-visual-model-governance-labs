# 第三章：指定人像目标与Best_text展示（2026-10-25）

本目录整理用户指定的既有实验 `/home/zcy/book_revision_labs_20261022/redcup_debug_projected_l14/pseudoword_projected_l14/`，没有重新搜索或生成图片。原始实验目标实际为人像，不能根据目录名当作红杯。

当前书内采用本组人像目标，展示UTF-8 Best_text、其实际文本通路输出和目标图。两图仅敏感区域叠加不透明黑块，遮挡外逐像素不变；原图只在作者本地审计目录保存，不加入公开展示资产。

- [Best_text原文](best_text.txt)与[精确Unicode转义](best_text_escaped.json)。含U+FFFD替换字符和U+1F5D3日历符号，不能删掉后当作同一输入。
- [本组参数、提交ID和指标](case_summary.json)。最优更新前索引195，原始ID代理余弦0.3305；文本重编码后0.2740，其生成图—目标图余弦0.5747。原始与重编码均为16IDs，但序列不相同。
- [精确遮挡矩形与原图哈希](redaction_manifest.json)。图像指标是原实验遮挡前的值，未对黑块图重新评分。

![目标与文本通路输出](target_best_text_comparison.png)

目标与输出的色彩、姿态、背景明显不同，不称精确重建。输出记录的checker为null，相关调试脚本显式设为None；本图不证明绕过输出检查器，也不提供成功率。Best_text仍有erotic等风险关联片段，显式黑名单不等于穷尽关联语义。

原实验另有best_ids等输出，本页只展示用户要求的best_text.png，不把另一条ID通路的图片或0.3305分数冒充文本通路结果。原实验natural仍是红杯提示，与本目标不匹配，不能作为本组同目标的自然语言对照。完整原始JSON及日志按原样存作者审计目录；此目录只分发遮挡版与所需的对应关系。

[第三章当前主实践与历史实验](../../chapters/03_compliance/README.md)。像素核验和图像哈希见[validation.json](validation.json)。

当前随书135行代码已缩短黑名单并关闭输出检查器；此页step195来自既有记录，不能写成对当前源文件的全流程重跑。原始优化目标不随仓库分发；展示遮挡图不能替代原图作为复现实验输入。
