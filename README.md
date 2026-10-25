# 生成式视觉大模型风险与治理：教学实践与习题

本书独立配套仓库，按8章组织10个实践、96道习题及逐题参考答案。当前随书代码位于 [examples/minimal](examples/minimal/README.md)，与作者2026-10-25修订书稿的10份代码同步；历史完整实现与实验分别保留，运行记录按实际执行版本解释。

## 开始使用

先按[环境与输入准备](examples/minimal/README.md)设置 `LAB_ROOT`。第1章只需Python标准库；第5章水印可在CPU运行；其余模型实践采用已记录的A800/CUDA环境。模型权重和整书PDF不进入Git。

在仓库根目录运行当前书内代码，也可用统一入口查看命令：

```bash
python tools/run.py 04 --dry-run
python tools/run.py 04
```

统一入口现在直接调用 `examples/minimal/`，运行日志另存于 `results/runs/`，模型输出写入 `LAB_ROOT`。第三章伪词使用 `--case pseudoword`，第六章使用 `--case sld` 或 `--case esd`，第八章完整定制使用 `--case customize`，仅条件特征使用 `--case features`。旧章节包装脚本保留为历史兼容入口。

## 按章学习

| 章 | 实践与说明 | 习题 / 答案 |
|---|---|---|
| 1 | [风险治理清单](chapters/01_overview/README.md) | [12题](chapters/01_overview/exercises.md) / [答案](chapters/01_overview/answers.md) |
| 2 | [文生图基础](chapters/02_basics/README.md) | [12题](chapters/02_basics/exercises.md) / [答案](chapters/02_basics/answers.md) |
| 3 | [过滤器与伪词](chapters/03_compliance/README.md) | [12题](chapters/03_compliance/exercises.md) / [答案](chapters/03_compliance/answers.md) |
| 4 | [CNNSpot鉴伪](chapters/04_detection/README.md) | [12题](chapters/04_detection/exercises.md) / [答案](chapters/04_detection/answers.md) |
| 5 | [DCT水印](chapters/05_watermark/README.md) | [12题](chapters/05_watermark/exercises.md) / [答案](chapters/05_watermark/answers.md) |
| 6 | [SLD与ESD](chapters/06_defense/README.md) | [12题](chapters/06_defense/exercises.md) / [答案](chapters/06_defense/answers.md) |
| 7 | [中性职业提示呈现](chapters/07_fairness/README.md) | [12题](chapters/07_fairness/exercises.md) / [答案](chapters/07_fairness/answers.md) |
| 8 | [肖像保护与定制](chapters/08_privacy/README.md) | [12题](chapters/08_privacy/exercises.md) / [答案](chapters/08_privacy/answers.md) |

## 当前案例与证据

| 实践 | 已保存结果与适用范围 |
|---|---|
| 第3章伪词 | [目标、Best_text和文本通路图片](evidence/pseudoword_sensitive_20261025/README.md)，敏感区域已遮挡；来自指定既有实验，不能视作最新135行短黑名单代码的重跑。公开红杯输入可用于流程验证。 |
| 第4章鉴伪 | [真实照片、ProGAN图与SD1.5红杯](evidence/cnnspot_gan_20261025/README.md)。相同权重与阈值下，本例GAN检出、SD漏检，展示本样本的泛化局限。 |
| 第6章SLD | [关闭、WEAK、MEDIUM、STRONG、MAX五档](evidence/sld_sweep_20261025/README.md)。各组输出检查器均关闭，固定提示与种子；分发图已遮挡敏感区域。 |
| 第7章职业呈现 | [5职业×6种子共30张](evidence/fairness_neutral_20261025/README.md)。沿用作者逐图人工标签，图7-2每行左侧标注职业；不是自动身份判定或去偏效果验证。 |
| 第8章保护 | [3/255实际IP-Adapter条件保护](evidence/privacy_ip_adapter_20261025/README.md)，含随机对照、正式9图及40图扩展。书内主例保留检查器；[关闭检查器的9图配对对照](evidence/privacy_checker_off_20261025/README.md)有[单独入口](examples/controls/README.md)。 |

证据不能代替对应配置的实跑；检查器黑图、未执行检查和人工展示黑块分别记录。代码哈希见[打印清单](examples/minimal/print_manifest.json)，结果来源见[证据索引](evidence/README.md)。本次发布整理没有重新运行GPU实验。

## 历史实现与维护

[完整单文件实现](examples/book/README.md)与[A800批量实验](experiments/a800/README.md)保留早期配置；历史488次生成、12张职业初筛和旧FaceNet保护等统计不能套用到当前最小代码。

```bash
python tools/validate_repository.py
python tools/render_exercises.py
python -m unittest discover -s tests -v
```

[维护与GitHub同步](docs/maintenance.md) · [实验记录要求](docs/experiment_record.md) · [接口依据](docs/sources.md)。仓库当前为私有，学生访问需获得仓库权限。
