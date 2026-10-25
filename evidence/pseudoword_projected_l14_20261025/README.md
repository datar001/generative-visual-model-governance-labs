# 原投影嵌入核心：CLIP-L/14 红杯实测（2026-10-25）

本轮替换此前logits/ST-softmax变体，按原仓库固定提交 `21eb35f16697529388a43bf1504089eccc6bc432` 实现连续嵌入优化。核心为：对连续向量和词表行归一化，找余弦最近邻，取原始未归一化token embedding，在独立投影点计算 `1 - mean(cosine)` 梯度，直接给连续参数，执行AdamW(lr=0.1, weight_decay=0.1)。连续参数不会被离散投影覆盖；对象任务不加入风格MSE或logit scale。

[实际136行代码](code/03_pseudoword.py) · [完整核验驱动](code/run_remote.py) · [未修改原源码与hash](code/original_source/manifest.json) · [原核心5步对照](runtime/original_core_parity.json) · [原生CLIP分数与梯度](runtime/native_score_gradient.json)。前5步ID、分数、梯度、连续参数、AdamW一阶/二阶状态最大绝对误差均0；初始/最佳/最终三个点与原生CLIP的分数和输入梯度误差均0。该证据不是对所有500步、所有实验条件的穷尽证明。

实验设置适配为SD1.5对应完整CLIP-L/14（196文本权重完全相同）、单一原红杯、16个全优化位置、完整BPE词表减显式黑名单。未复制原入口自然提示模板、英语掩码与Word2Vec同义词扩展；不称整套论文默认实验复现。L/14固定版本32bd64288804d66eefd0ccbe215aa642df71cc41。

[候选池及完整黑名单](pseudoword_projected_l14/vocabulary.json)：49,408减2特殊ID、536字符串命中，保留48,870。沿用cup/mug等关联词列表与NFKC+casefold子串规则，500条完整文本全通过。本次最优仍含pots、sake等容器/饮品相关普通词，规则不保证语义穷尽，也不保证全部是无意义造词。

[完整轨迹及末态](pseudoword_projected_l14/search.json)：随机token embedding初始化，种子2026；16×768=12,288个连续参数，500次更新。best仅从step0..499的500个更新前投影状态选；第500次更新后的final_after_update单独保存，不参与选优。best step350指已完成350次更新、第351次更新前，分数0.331151068；初始0.162025481，更新500次后另测0.243155405。

[实际条件与指标](pseudoword_projected_l14/comparison.json)：

|条件|内容ID数|L/14图文余弦|L/14返回图-目标余弦|输出检查器|
|---|---:|---:|---:|---|
|初始ID|16|0.162025481|0.665106058（黑图）|True|
|最佳原始ID|16|0.331151068|0.761290312|False|
|最佳文本重编码|16|0.307804763|0.760143340|False|
|自然语言参照|8|0.299474955|0.937918782|False|

初始图是原生检查器的全黑替换输出，其0.6651只描述黑图，不能评价被遮蔽画面；没有关闭检查器或换种子挑选。其余三次原生标记False。四次均SD1.5、种子2026、30步、CFG7.5、512²；自然语言参照像素与前次L/14实验相同。

![原红杯](pseudoword_projected_l14/target.png)
![初始检查器黑图](pseudoword_projected_l14/initial_ids.png)
![最优原始ID](pseudoword_projected_l14/best_ids.png)
![最优文本重编码](pseudoword_projected_l14/best_text.png)
![自然语言参照](pseudoword_projected_l14/natural.png)

最优两种条件均生成木桌上的多个浅色杯具；文本通路多了盖状容器及餐盘。没有还原偏左俯视单个红杯。自然语言参照有红杯但位置、杯柄、碟状结构、木纹不同。CLIP参与搜索与测量，两类余弦不是独立评价、不是成功概率。

按SD实际使用的慢版CLIPTokenizer统计，129/500状态可decode/reencode无损闭环；CLIP处理器快版CLIPTokenizerFast为138/500，最佳文本在Fast下重编码18IDs，在实际SD慢版下是16IDs。两者词表、候选池和原始ID解码文本相同，但Unicode预处理不同，不能混用计数。[CPU分词器独立核验](runtime/cpu_tokenizer_audit.json)。最优原始和重编码都16个，但ID不同；字节片段涉及替换字符、不可见字符，数量相同不保证条件相同。best_ids.roundtrip_equal=False；best_text字段只检查实际重编码提交ID。

[验收和计时](runtime/acceptance.json)：物理GPU0，500更新段CUDA同步耗时15.0729秒，包含前5步状态快照，排除SD加载生成及额外原源码对照；完整驱动35.43秒（导入Torch后起计）。[环境](runtime/environment.json) · [权重核验](runtime/text_encoder_alignment.json) · [优化器状态](runtime/optimizer_state.pt)。

[与旧ST变体统一L/14重评](runtime/cross_method_rescore.json)仅为历史诊断。本轮按原方式随机初始化，旧轮是固定ID初始化；不能把差异全归因于算法，单目标单种子也不支持一般优劣结论。旧实验按日期保留。

准备环境和模型后，在仓库根目录设置LAB_ROOT，复制原红杯输入并运行：

```bash
export CUDA_VISIBLE_DEVICES=0
python experiments/a800/download_models.py sd15 clip_l14
mkdir -p "$LAB_ROOT/inputs"
cp examples/book/assets/pseudoword_target.png "$LAB_ROOT/inputs/"
python examples/minimal/03_pseudoword.py
python tools/plot_pseudoword_projected_l14.py
```

输出在 `$LAB_ROOT/pseudoword_projected_l14/`。本次A800实际源文件为 `/home/zcy/book_revision_labs_20261022/pseudoword_projected_l14_20261025/code/03_pseudoword.py`。
