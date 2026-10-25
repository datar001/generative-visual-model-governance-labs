# 书内最小完整实现：全章对应版（2026-10-25）

书内实践目标、操作、代码和结果图表现在直接对应本目录十份程序。每份都包含所需导入、实际模型加载或算法步骤与结果保存，不依赖隐藏的项目Python辅助函数。额外校验、恢复运行、完整日志和大样本对照放在[完整单文件实现](../book/README.md)与[审计实验](../../experiments/a800/README.md)。书稿按整个代码文件引用，新增注释不会再截掉末尾。

只做第1章时，标准库即可：先设置LAB_ROOT，运行01_governance.py，无需安装GPU环境或下载模型。其他章节先按[固定环境与下载说明](../book/README.md#环境与首次准备)配置依赖。

## 固定环境与输入准备

已实测Python 3.10.6、Torch 2.4.1+cu118、Diffusers 0.32.2、Transformers 4.46.3、NumPy 1.26.4、Pillow 10.4.0。CPU水印只需NumPy/OpenCV；FaceNet按完整环境说明安装并预缓存公开VGGFace2权重。下面的下载与输入准备在仓库根目录执行，首次需要网络与相应模型许可；这些准备脚本不参与最小代码的算法计算。

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

`manifests/models.json`供生成/CLIP/过滤器示例读取固定模型的本地路径；鉴伪直接加载models下的CNNSpot文件。第8章使用本目录assets/face_reference.png给出的160×160合成成人头像，不要求学生先运行检测裁剪流程。所有GPU代码在导入Torch前固定物理0号卡。

Windows PowerShell将export LAB_ROOT=...改为`$env:LAB_ROOT='...'`，复制输入可用Copy-Item。设置好环境后按顺序运行：

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

第2章先生成共同输入。第三章伪词当前输出在`$LAB_ROOT/pseudoword_fullvocab/`；其他程序沿用各自的`$LAB_ROOT/minimal_examples_20261024/`子目录。每轮用独立LAB_ROOT保存，不覆盖历史结果。

## 代码实际做了什么

| 章节 | 最小代码直接完成的内容 |
|---|---|
| 第1章 风险治理清单 | 3条教学记录；CSV固定6列；标签出现6次，全部pending。 |
| 第2章 文生图基础 | 4类提示×3种子，12张PNG；原生检查标记打印终端，不自动写参数JSON。 |
| 第3章 过滤器与伪词 | 过滤器3条文本+3张图；完整CLIP词表减黑名单后48,870候选、16个ID、500次更新、501状态和4张生成。 |
| 第4章 CNNSpot鉴伪 | 一真一假两次推理，2条分数；本例生成红杯被漏检。 |
| 第5章 DCT水印 | 1张红杯，强度16；PNG/JPEG75/未嵌入3条解码记录，一致率1/1/0.625。 |
| 第6章 SLD与ESD | SLD同提示、同种子2张；ESD一条目标风格和一条普通提示，原/编辑共4张。 |
| 第7章 职业外观初筛 | 2职业×2提示×3种子，12张；4组各3条；全部unreviewed。 |
| 第8章 肖像保护 | 1张已裁剪合成人脸，保存1个FaceNet输入分数，实际定制2张；不含JPEG和输出人脸评分。 |

代码未自动完成的事项不能写成已有结果：例如第2章只在终端打印检查标记，不写参数JSON；第4章没有CPU分支/JPEG实验；第5章没有PSNR；SLD/ESD没有CLIP测量；第7章没有完成真实人工复核；第8章没有JPEG、适配器特征或输出人脸评分。

## 红杯全词表/500步实践

`03_pseudoword.py`为113行完整实现。候选池从49,408个ID仅排除特殊ID和显式黑名单，不做词尾/语言白名单。16个位置ST-softmax与Adam更新500次，前向离散，模型冻结。最佳在408步，代理余弦0.3324，最终0.2711。保存原始ID通路与解码文本重新分词通路，最佳16IDs实际重编码为15个；四图加目标均保留。

实际生成仍没有红杯：最佳ID条件为木纹、白色标牌及红杆，图像余弦0.7087；文本重编码为0.6531；自然语言参照有红杯，余弦0.9444。条件和两类指标不能混用。黑名单为NFKC+casefold子串规则，完整候选文本也复查；本次501条全部通过。初始ID条件与旧标准文本初始图逐像素一致。

可选绘图：`python tools/plot_pseudoword_fullvocab.py`，读取LAB_ROOT/pseudoword_fullvocab/search.json。旧24-token曲线工具仍作为历史保留。

## 对照验证

此前十份程序重跑，共34次生成尝试，123项对应核验通过。第三章伪词现已更新，单独重新实跑并保存[全词表/500步新证据](../../evidence/pseudoword_fullvocab_20261025/README.md)，其他九份代码保持原样。[本轮全部原始证据](../../evidence/all_practice_alignment_20261025/README.md)保存CSV、JSON、日志、图片、环境及实际执行源码哈希。[打印清单](print_manifest.json)记录当前整个文件的范围。

书内主结果只使用本目录最小实现的样本与指标。18文本/31图、60次鉴伪、216条水印、18/24次防御、400次职业采样及27次定制，继续作为仓库完整扩展实验保留，不能与最小案例混用分母。

本例的CLIP外观标签不代表真实身份；黑名单约束的红杯匹配不是真实安全越狱；非露骨SLD样例不给出色情抑制率；FaceNet输入分数下降不等于定制输出身份保护成功。所有结论均按本次样本与实际实现限定。

assets/face_reference.png来自此前第二章虚构成人头像的已裁剪教学输入，完整来源见2026-10-24实跑证据。本轮保持该输入不变，以便比较算法行为。
