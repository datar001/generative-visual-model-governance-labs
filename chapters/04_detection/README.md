# 第4章 CNNSpot鉴伪：当前书内最小实践（2026-10-25）

空间域ResNet-50监督二分类、sigmoid分数和固定0.5阈值；不做JPEG实验。

## 目标与实际产物

一真一假两次推理，2条分数；本例生成红杯被漏检。

## 运行

先按[固定环境与输入准备](../../examples/minimal/README.md)配置LAB_ROOT，再在仓库根目录运行：

```bash
python examples/minimal/04_detection.py
```

第2章输出供图像审核、鉴伪与水印读取；第8章直接使用配套已裁剪合成人脸。各文件的结果目录由代码中的out定义。

## 实测与逐项核对

[当前十份代码的原始结果与核验](../../evidence/all_practice_alignment_20261025/README.md)。书中主表与图片均对应本次最小输出；扩大提示、种子、条件和评价器后，应另存并重跑。

## 习题与答案

[分类型习题](exercises.md) · [逐题参考答案](answers.md)

<details>
<summary>仓库完整实验扩展与此前说明</summary>

> 2026-10-24最新：书内采用最小完整实现；导入、实际模型加载和核心运算保留，额外校验及完整函数在Git仓库中。
> 运行 `python examples/minimal/04_detection.py`（[代码](../../examples/minimal/04_detection.py)）。
> [最小示例运行准备](../../examples/minimal/README.md) · [完整函数与批量实现](../../examples/book/README.md)。下方大样本实测对应完整实验。

# 第4章：生成图像鉴伪

完整环境和数据准备见 [全章运行说明](../../experiments/a800/README.md)。运行前设置独立的 `LAB_ROOT`，实际结果保存于其 `results/`；书中只展示固定样例，完整原图保留在实测结果包。

## 用CNNSpot学习空间域CNN鉴伪

实践目标：

- 加载CNNSpot官方公开权重，说明ResNet-50单输出二分类器与生成类分数的含义。
- 对有来源记录的真实照片和第2章生成图分别检测原始输入及三档JPEG副本，保存60组分数。
- 在固定0.5阈值下标记预测与来源是否一致，解释压缩影响和漏检现象。

本实践把第4.2节的空间域鉴伪方法落到可运行的经典CNNSpot检测器。学员先理解“像素特征—二分类分数—阈值判断”的流程，再用JPEG对照观察生成痕迹变化，认识训练范围与泛化边界。

学习收获：

- 掌握全局图像信号中的空间域CNN监督二分类：RGB图像经ResNet-50提取特征，单个logit经sigmoid转为生成类分数。
- 固定权重和预处理，通过同图对照分析JPEG影响；区分原始输入与质量100的重编码副本。
- 区分模型分数、来源标签和事实真实性；理解在ProGAN上训练的检测器不保证识别扩散模型图像。

运行：

```bash
python experiments/a800/detection.py
```

实测结果：A800完成3张真实照片与12张生成图×4条件=60次CNNSpot推理。各条件真实照片0/3误报、生成图12/12漏检，固定阈值和域外泛化边界如实展示。

## 习题和参考答案

- [分类型习题](exercises.md)
- [逐题参考答案](answers.md)

更多方法见 [拓展材料](extensions.md)。CNNSpot资源来源和许可说明保留在本章配套资料中。


</details>
