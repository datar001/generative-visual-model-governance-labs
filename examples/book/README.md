> 第8章当前主实践已更新为实际IP-Adapter条件优化，见[新版最小代码](../../examples/minimal/08_privacy.py)。本目录第8章完整独立文件保留旧FaceNet批量基线，供历史复现，不代表新版算法。

# 完整单文件实现（2026-10-24）

这里保留10份完整单文件实现，对应本日早期的284页书稿。当前书内已改为[最小完整实现](../minimal/README.md)，本目录的算法、通用函数及完整运行记录继续保留，以下连续行号描述属于早期版本。每份文件都包含自己的导入、路径、必要函数、模型加载、算法步骤、执行入口和结果保存；不导入 common.py、filter_core.py、pseudoword_core.py 或其他项目Python辅助模块。书中各段按连续行号排列，依次拼接就是同一个文件。

模型和图片是运行输入，需要按下列说明准备。代码不把下载好的权重藏在未说明的系统缓存中：`LAB_ROOT/manifests/models.json` 明确登记固定版本的本地目录；每份文件中的 `model()` 都有完整定义。SD示例直接给出 Diffusers 的 `StableDiffusionPipeline.from_pretrained`、FP16、CUDA、输出检查器检查和同种子采样。

## 为什么增加这一套文件

此前书内摘录了实验脚本中间的若干行，函数定义和导入在另一文件，读者只看书无法运行。原 `experiments/a800/` 仍保留完整审计、模型准备与原始实验；本目录用于学生按书逐段学习。两者不能混淆。

## 环境与首次准备

使用已实测的 Python 3.10、PyTorch 2.4.1+cu118、TorchVision 0.19.1+cu118，安装根目录 `requirements-a800.txt`，再按[固定环境说明](../../experiments/a800/README.md)安装 FaceNet。CPU水印只需NumPy/OpenCV；第一章只需标准库。完整生成示例使用CUDA，不把未测试的低显存版本写成已验证。

首次安装命令（先创建并激活Python 3.10环境）：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-a800.txt
python -m pip install facenet-pytorch==2.6.0 --no-deps
```

FaceNet的上游依赖声明较旧，本轮采用上述实际验证组合，不让它自动替换已经固定的Torch/Pillow版本。详细版本随证据清单保存。

以下均在教学仓库根目录运行。把 `/path/to/your/lab_run` 改成自己的实验目录，首次准备需要网络下载公开权重；已有模型且清单正确时不必重复下载。

```bash
export LAB_ROOT=/path/to/your/lab_run
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$LAB_ROOT/models/torch"
python experiments/a800/download_models.py sd15 sd14 clip text_filter image_filter ip_adapter
python experiments/a800/prepare_sld.py
python experiments/a800/prepare_weights.py cnnspot esd
python experiments/a800/prepare_inputs.py
mkdir -p "$LAB_ROOT/inputs"
cp examples/book/assets/pseudoword_target.png "$LAB_ROOT/inputs/"
```

FaceNet第一次加载使用公开的VGGFace2预训练权重。脱网前应在相同 `TORCH_HOME` 下完成下载；本轮A800固定哈希见原证据中的 `manifests/facenet_weights.json`。不随代码分发模型权重。

Windows PowerShell把 `export LAB_ROOT=...` 换成 `$env:LAB_ROOT='...'`，复制目标图可用 `Copy-Item`；A800命令按Linux shell写出。所有示例在导入torch前固定 `CUDA_VISIBLE_DEVICES=0`。

## 按顺序运行

```bash
python examples/book/01_governance.py
python examples/book/02_generation.py
python examples/book/03_pseudoword.py
python examples/book/04_detection.py
python examples/book/05_watermark.py
python examples/book/06_sld.py
python examples/book/06_esd.py
python examples/book/07_fairness.py
python examples/book/08_privacy.py
python examples/book/03_filters.py
```

第3章过滤器可以在第2章后先运行，随后第6章结果只作为扩展；放在上面顺序最后是为了复核书中包含31张图的完整审核集合。

结果写入 `$LAB_ROOT/book_examples_20261024/`，不覆盖原2026-10-22的 `results/`。第4、5、8章从这一新结果目录读取第2章输入，而不是隐式使用旧结果。第3章红杯搜索目标是单独声明的固定样本。

## 方法和结果边界

完整代码保留所有条件和分母：第2章12张、伪词7张实际生成、SLD18次、ESD24张、公平性400次、肖像定制27张，共488次生成尝试。安全检查器替换、不确定和未人工复核分别保存。图像分类器没有风险图像阳性对照；SLD使用非露骨输入；伪词是无害目标教学类比；隐私分数下降不等于通用防护成功。

本轮验证直接从空工作目录执行本目录文件，清空项目PYTHONPATH，不借用原始实验脚本的函数。实测日志和逐图对照结论见本轮证据目录；运行状态以实际结果为准。

## 本轮实跑结论

10份完整文件均运行完成，27项功能/范围核验通过。除隐私梯度更新外，与原实测逐图/逐分数一致。隐私同一文件独立复跑也出现差异，书中图表更新为本轮首轮结果并标注波动；严格不一致项没有被删除。详见[完整核验记录](../../evidence/complete_code_20261024/README.md)。
