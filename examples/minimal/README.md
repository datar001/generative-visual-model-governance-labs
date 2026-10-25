# 书内最小完整实现（2026-10-24）

书中只展示这一目录的最小实现。10份程序都给出导入、模型加载或算法运算以及结果保存，不依赖项目Python辅助模块。第二章32行；必要的安全过滤方法与不确定标签保留，通用校验、兼容分支、恢复运行和完整日志放在[完整单文件实现](../book/README.md)与[审计实验](../../experiments/a800/README.md)。书内打印文件的行号和哈希见 `print_manifest.json`。

第三章已于2026-10-25进一步对齐：书中结果主体就是最小代码的6条审核记录与64候选/2张生成图，完整扩展统计不再混入本节。新增注释保留，书稿按完整文件引用，实测见[本轮记录](../../evidence/ch3_alignment_20261025/README.md)。

## 运行前提

先按[固定环境和下载说明](../book/README.md#环境与首次准备)安装依赖。已实测Python 3.10.6、Torch 2.4.1+cu118、Diffusers 0.32.2、Transformers 4.46.3；水印只需NumPy/OpenCV。以下命令在教学仓库根目录运行，模型和普通照片由准备脚本下载，首次准备需要网络与相应模型许可。准备脚本不参与最小示例的算法计算。

```bash
export LAB_ROOT=/path/to/your/lab_run
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$LAB_ROOT/models/torch"
python experiments/a800/download_models.py sd15 sd14 clip text_filter image_filter ip_adapter
python experiments/a800/prepare_weights.py cnnspot esd
python experiments/a800/prepare_inputs.py
mkdir -p "$LAB_ROOT/inputs"
cp examples/book/assets/pseudoword_target.png "$LAB_ROOT/inputs/"
cp examples/minimal/assets/face_reference.png "$LAB_ROOT/inputs/"
```

`manifests/models.json`把固定版本映射到本地路径，每份涉及模型的程序直接读取该文件，没有隐藏的 `model()` 或 `load_sd()` 函数。FaceNet读取 `TORCH_HOME` 中的VGGFace2权重；脱网运行前先按完整环境说明缓存。A800已有这些输入，本轮没有重新下载。所有GPU程序在导入Torch前固定物理0号卡。

Windows PowerShell将 `export LAB_ROOT=...` 改成 `$env:LAB_ROOT='...'`；复制输入可用 `Copy-Item`。本书最小生成实现按CUDA环境编写，CPU水印与风险清单不需要GPU。设置好路径后，按下列顺序执行：

```bash
python examples/minimal/01_governance.py
python examples/minimal/02_generation.py
python examples/minimal/03_filters.py
python examples/minimal/03_pseudoword.py
python examples/minimal/04_detection.py
python examples/minimal/05_watermark.py
python examples/minimal/06_sld.py
python examples/minimal/06_esd.py
python examples/minimal/07_fairness.py
python examples/minimal/08_privacy.py
```

结果写入 `$LAB_ROOT/minimal_examples_20261024/`，不会覆盖已归档的完整实验。第3章过滤器、第4章和第5章读取第2章的新输出；第8章读取本目录提供的已裁剪合成成人头像，无需先跑完整人脸检测流程。

## 最小示例和完整结果的关系

| 实践 | 最小示例 | 仓库完整实验（扩展） |
|---|---|---|
| 风险清单 | 3个人工填写情境，CSV与计数 | 相同情境与风险计数，详细字段记录 |
| SD生成 | 4类提示×3种子，12张 | 同样12次配置，额外保留完整参数/来源日志 |
| 三类过滤器 | 3条文本、3张普通图 | 18条文本、31张图；无图像风险阳性对照 |
| 伪词类比 | 单次64候选随机搜索、2张实际生成 | 480次查询、两种策略、7张实际生成 |
| CNNSpot | 一真一假，2次推理 | 15张输入×4种编码条件，60次推理 |
| DCT水印 | 一张图、强度16，PNG/JPEG/未嵌入解码 | 9张图×3强度×8条件，216条记录 |
| SLD | 同种子off/STRONG，2张 | 3类提示×3种子×2模式，18次采样 |
| ESD | 目标风格与普通提示各一，原/编辑共4张 | 24张及CLIP对照；均直接加载官方权重 |
| 职业呈现 | 2职业×2模式×3种子，12次 | 400次，保留不确定/检查器替换/未复核 |
| 肖像保护 | 1张已裁剪合成头像，PGD与2次定制 | 3参考×3条件×3种子，27次定制与输出评分 |

最小示例用于理解方法，不用其少量样本替换书中完整实验统计。隐私最小示例只计算保存后输入图的FaceNet特征相似度，不能拿它与书中定制输出人脸评分混比；梯度实跑也可能波动。公平性CLIP是外观初筛，所有标签仍未人工复核。伪词仅为无害红杯类比，SLD使用非露骨输入，均不报告真实风险绕过率或色情抑制率。

## 实测

十份最小程序均从A800空工作目录运行成功，清空项目PYTHONPATH，GPU仅0号卡；共34次实际生成尝试。35项输出核验通过，第二章12张图与完整实现逐像素相同。证据见[本轮实跑](../../evidence/minimal_20261024/README.md)。

`assets/face_reference.png`是第二章生成的虚构成人头像经完整实践裁剪得到的160×160输入，源文件见2026-10-24完整实验的 `08_privacy/face_2026/original.png`；仅作本书教学样例。
