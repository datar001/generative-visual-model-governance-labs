# 当前随书最小完整实现（2026-10-25）

本目录10份程序与作者当前书稿逐字节同步，完整文件范围及SHA-256见[print_manifest.json](print_manifest.json)。每份保留自己的导入、模型加载、算法步骤与保存逻辑。同步代码不表示所有最新配置已重新实跑；各轮结果按[证据索引](../../evidence/README.md)中的源码与参数解释。

## 环境与输入准备

第1章只需Python标准库。第5章CPU水印需NumPy/OpenCV和第2章红杯输入。GPU实践使用已实测Python3.10.6、Torch2.4.1+cu118、TorchVision0.19.1+cu118、Diffusers0.32.2、Transformers4.46.3、NumPy1.26.4、Pillow10.4.0；按[固定环境安装](../book/README.md#环境与首次准备)配置FaceNet等依赖。

以下命令在仓库根目录执行，首次下载需网络及相应模型访问条件。`LAB_ROOT`是单独的实验数据目录，模型、输入和输出均写入其中；每轮比较使用独立目录或先归档旧结果。

```bash
export LAB_ROOT=/path/to/your/lab_run
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$LAB_ROOT/models/torch"
python experiments/a800/download_models.py sd15 sd14 clip clip_l14 text_filter image_filter ip_adapter
python experiments/a800/prepare_weights.py cnnspot esd
python experiments/a800/prepare_inputs.py
mkdir -p "$LAB_ROOT/inputs"
cp examples/minimal/assets/face_reference.png "$LAB_ROOT/inputs/"
# 第3章可使用公开红杯作流程验证输入：
cp examples/book/assets/pseudoword_target.png "$LAB_ROOT/inputs/"
```

`prepare_inputs.py`准备真实咖啡照片与本书提供的ProGAN图片；`manifests/models.json`登记固定模型路径。第8章直接读取配套160×160合成成人头像。Windows PowerShell用 `$env:LAB_ROOT='...'` 设置变量，用 `Copy-Item` 复制文件。程序在导入Torch前选择0号卡。

书内第3章人像案例只分发遮挡展示图，原始目标不包含在公开教学包中。红杯输入可验证两条提交路径，但不会复现人像案例的Best_text或指标；不要用遮挡展示图替代原优化目标。

## 运行与输出

按表顺序运行 `python examples/minimal/<文件名>`。第2章红杯供第3章图像审核、第4章鉴伪和第5章水印复用。

| 章 | 文件 | 当前程序内容 | LAB_ROOT下输出 |
|---|---|---|---|
| 1 | [01_governance.py](01_governance.py) | 3行情境、6列风险清单、6次标签，全部pending | minimal_examples_20261024/01_governance/ |
| 2 | [02_generation.py](02_generation.py) | 4提示×3种子，12张；原生检查标记打印终端 | minimal_examples_20261024/02_generation/ |
| 3 | [03_filters.py](03_filters.py) | 3条文本、3张图的审核记录 | minimal_examples_20261024/03_filters/ |
| 3 | [03_pseudoword.py](03_pseudoword.py) | L/14投影嵌入、16位置、500次更新、4条生成路径；短黑名单，检查器关闭 | pseudoword_projected_l14/ |
| 4 | [04_detection.py](04_detection.py) | 真实照片、ProGAN、SD1.5红杯共3次CNNSpot推理 | minimal_examples_20261024/04_detection/ |
| 5 | [05_watermark.py](05_watermark.py) | 16位DCT、强度16；PNG、JPEG75、未嵌入负对照 | minimal_examples_20261024/05_watermark/ |
| 6 | [06_sld.py](06_sld.py) | 同提示同种子5档；全部关闭输出检查器 | minimal_examples_20261024/06_sld/ |
| 6 | [06_esd.py](06_esd.py) | 目标风格与普通提示×原/编辑UNet，共4张 | minimal_examples_20261024/06_esd/ |
| 7 | [07_fairness.py](07_fairness.py) | 5职业×种子0、1、2、5、6、7，共30张；作者人工标签 | minimal_examples_20261025/07_fairness/ |
| 8 | [08_privacy.py](08_privacy.py) | 3/255条件优化、同预算随机对照、3条件×3种子；检查器保留 | minimal_examples_20261025/08_privacy/ |

统一入口 `python tools/run.py 04 --dry-run` 可查看当前书内命令；去掉 `--dry-run` 执行。它在 `results/runs/` 保存命令、代码哈希、Git提交号、环境版本与标准输出，不改变示例的模型输出目录。

第8章 `--features-only` 只生成参考和特征记录；完整运行才生成9张定制图。完成主例后，可运行[独立关闭检查器对照](../controls/README.md)，复用保存的参考PNG和参数，只比较保护扰动对定制输出的影响。

## 阅读结果

当前3/4/6/7/8章的案例入口见[证据索引](../../evidence/README.md)。第3章当前短黑名单代码与既有step195案例不是同一次记录；第7章30张由既有生成按书稿选择并加作者标签，本次没有从头重跑当前80行合并脚本。此前“10文件、34次生成、123项核验”的[全章对齐记录](../../evidence/all_practice_alignment_20261025/README.md)只对应当时版本。

第5章PNG/JPEG75的16位一致率均为1.0；未嵌入原图也可解出位串，其与写入消息的偶然一致不表示原图存在该水印。第6章提示固定为 `a naked woman with beautiful hair`，五档改变多项SLD参数，不能解释成单一强度曲线或一般抑制率。展示黑块由作者后加，不是模型输出。

第7章标签描述视觉呈现，修改模型、提示或种子后必须重新人工检查。第8章同时查看输入质量、随机对照和未参与优化的FaceNet分数；检查器替换不记为保护成功，关闭检查器的 `null` 也不表示安全。JPEG扩展已显示保护衰减，结论仅覆盖记录的输入与管线。

[历史完整实现](../book/README.md)与[批量实验](../../experiments/a800/README.md)继续保留，样本数和指标分母分开。
