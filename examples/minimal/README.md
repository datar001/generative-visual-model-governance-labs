# 书内最小完整实现：全章对应版（2026-10-25）

书内实践目标、操作、代码和结果图表现在直接对应本目录十份程序。每份都包含所需导入、实际模型加载或算法步骤与结果保存，不依赖隐藏的项目Python辅助函数。额外校验、恢复运行、完整日志和大样本对照放在[完整单文件实现](../book/README.md)与[审计实验](../../experiments/a800/README.md)。书稿按整个代码文件引用，新增注释不会再截掉末尾。

只做第1章时，标准库即可：先设置LAB_ROOT，运行01_governance.py，无需安装GPU环境或下载模型。其他章节先按[固定环境与下载说明](../book/README.md#环境与首次准备)配置依赖。

## 固定环境与输入准备

已实测Python 3.10.6、Torch 2.4.1+cu118、Diffusers 0.32.2、Transformers 4.46.3、NumPy 1.26.4、Pillow 10.4.0。CPU水印只需NumPy/OpenCV；FaceNet按完整环境说明安装并预缓存公开VGGFace2权重。下面的下载与输入准备在仓库根目录执行，首次需要网络与相应模型许可；这些准备脚本不参与最小代码的算法计算。

```bash
export LAB_ROOT=/path/to/your/lab_run
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$LAB_ROOT/models/torch"
python experiments/a800/download_models.py sd15 sd14 clip clip_l14 text_filter image_filter ip_adapter
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

第2章先生成共同输入。第三章伪词当前输出在`$LAB_ROOT/pseudoword_projected_l14/`；其他程序沿用各自的`$LAB_ROOT/minimal_examples_20261024/`子目录。每轮用独立LAB_ROOT保存，不覆盖历史结果。

## 代码实际做了什么

| 章节 | 最小代码直接完成的内容 |
|---|---|
| 第1章 风险治理清单 | 3条教学记录；CSV固定6列；标签出现6次，全部pending。 |
| 第2章 文生图基础 | 4类提示×3种子，12张PNG；原生检查标记打印终端，不自动写参数JSON。 |
| 第3章 过滤器与伪词 | 过滤器3条文本+3张图；原投影嵌入优化核心、L/14全词表减黑名单48,870候选，16位置、500次更新和4次生成。 |
| 第4章 CNNSpot鉴伪 | 一真一假两次推理，2条分数；本例生成红杯被漏检。 |
| 第5章 DCT水印 | 1张红杯，强度16；PNG/JPEG75/未嵌入3条解码记录，一致率1/1/0.625。 |
| 第6章 SLD与ESD | SLD同提示、同种子2张；ESD一条目标风格和一条普通提示，原/编辑共4张。 |
| 第7章 职业外观初筛 | 2职业×2提示×3种子，12张；4组各3条；全部unreviewed。 |
| 第8章 肖像保护 | 1张已裁剪合成人脸，保存1个FaceNet输入分数，实际定制2张；不含JPEG和输出人脸评分。 |

代码未自动完成的事项不能写成已有结果：例如第2章只在终端打印检查标记，不写参数JSON；第4章没有CPU分支/JPEG实验；第5章没有PSNR；SLD/ESD没有CLIP测量；第7章没有完成真实人工复核；第8章没有JPEG、适配器特征或输出人脸评分。

## 红杯原投影嵌入核心/500步实践

`03_pseudoword.py`为136行完整实现，加载`clip_l14`完整图文模型。连续优化形状为1×16×768的词嵌入；将连续向量与完整词表行分别L2归一化后按余弦最近邻选ID，送入文本编码器的是未经归一化的原始词嵌入。在投影点计算目标梯度，直接赋给连续参数的grad，再用AdamW更新连续参数，学习率0.1、weight_decay=0.1；投影结果不覆盖连续参数本体。

执行500次更新，按原循环只从更新前第0--499步的500个投影点选优，其中500个完整解码文本通过黑名单；最后一次更新后的第500步另外记录，不参与选优。500条轨迹保留原始ID、解码文本和黑名单判定；`final_after_update`独立保存末次更新后的投影点。随机token embedding初始化种子2026，和旧ST轮的0/0.1 logits初始化不同。

本例使用与SD1.5的196个文本权重逐项相同的完整CLIP-L/14、16个全可优化位置、单张红杯目标、无自然语言前缀和全BPE减原黑名单。这些是本轮设置适配；原仓库的英语候选/同义词过滤、前缀模板、多图目标等未照搬，因此称原投影嵌入优化核心重实现，不称完整默认实验或论文性能复现。候选总计49,408，排除2特殊ID和536个黑名单命中后48,870，包含BPE碎片与非英语词元。黑名单仍是NFKC+casefold子串规则，非语义关联穷尽；提示可能包含普通词和名字。CLIP池化投影与SD逐位置条件虽均为768维，生成仍用SD自己的77×768条件。

最佳记录step=350（已更新350次、下一次更新前的投影），代理余弦0.3312；初始0.1620，第500次更新后另测0.2432。四次生成的L/14图像余弦依次为初始0.6651、最佳ID0.7613、最佳文本重编码0.7601、自然语言参照0.9379；最佳16IDs经SD1.5实际慢版分词器重编码为16。按SD1.5实际CLIPTokenizer（慢版）统计，仅129/500个投影状态可精确解码重编码回原ID；CLIPProcessor使用CLIPTokenizerFast，另测为138/500，最佳序列重编码为18IDs。两者词表映射、48,870候选和500条解码字符串相同，但Unicode重编码结果不同，不能把Fast统计用于SD提交路径。最佳原始序列与SD慢版重编码序列不同，即使词元数量相同也不能视为无损回读。四次输出检查标记依次为[[True], [False], [False], [False]]。被检查器替换的initial_ids保留黑图，其图像余弦描述替换后的黑图，不能解释为实际生成语义或筛前画面的质量。500次优化循环实测15.07秒，计时含前五步核验hook快照，不含SD加载、生成与额外原源码一致性核验。原红杯目标为木纹桌面上偏左的俯视单个红杯。初始ID输出被原生检查器标为True并替换成全黑图，其图像余弦0.6651只描述替换黑图，不能评价被遮蔽的原画面。最优原始ID图为木桌上的多个浅色带柄杯具、液体与木棒；最优文本重编码图还出现带盖容器与碟盘，没有目标的单个红杯和对应构图。自然语言参照有木桌上的红杯，但位置、杯柄、碟状结构及纹理不同。最优提示含pots与sake等正常词，黑名单仅显式字符串约束，未穷尽容器/饮品的语义关联。两类余弦均由参与优化的L/14计算，非独立评价。原始16IDs与解码文本重新分词分两次实际生成；不能混用两条通路的分数或图片。

运行`python tools/plot_pseudoword_projected_l14.py`读取LAB_ROOT/pseudoword_projected_l14/search.json。完整驱动额外保存原源码前五步核心一致性核验，最小代码本身不导入原仓库；旧ST-L14、B32和小词表结果按日期保留，初始化与优化器不同，不当作单变量因果对照。

## 对照验证

此前十份程序重跑，共34次生成尝试，123项对应核验通过。第三章伪词现已更新，单独重新实跑并保存[原投影嵌入核心实测](../../evidence/pseudoword_projected_l14_20261025/README.md)，其他九份代码保持原样。[本轮全部原始证据](../../evidence/all_practice_alignment_20261025/README.md)保存CSV、JSON、日志、图片、环境及实际执行源码哈希。[打印清单](print_manifest.json)记录当前整个文件的范围。

书内主结果只使用本目录最小实现的样本与指标。18文本/31图、60次鉴伪、216条水印、18/24次防御、400次职业采样及27次定制，继续作为仓库完整扩展实验保留，不能与最小案例混用分母。

本例的CLIP外观标签不代表真实身份；黑名单约束的红杯匹配不是真实安全越狱；非露骨SLD样例不给出色情抑制率；FaceNet输入分数下降不等于定制输出身份保护成功。所有结论均按本次样本与实际实现限定。

assets/face_reference.png来自此前第二章虚构成人头像的已裁剪教学输入，完整来源见2026-10-24实跑证据。本轮保持该输入不变，以便比较算法行为。
