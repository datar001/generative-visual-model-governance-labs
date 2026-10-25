> 当前书内运行入口见[最小实现](../../examples/minimal/README.md)，统一运行器已直接调用这些文件。本页下方为2026-10-22批量实验及后续历史扩展，保留各自原参数与分母。当前第3章人像/短黑名单、第6章检查器关闭五档、第7章30张人工记录均不能套用下方历史结果。

> 第8章 `privacy.py` 调用当前3/255条件优化最小代码；`privacy_facenet_baseline.py` 保留旧27图FaceNet基线。关闭检查器的配对对照另见[独立入口](../../examples/controls/README.md)。

# 全章实测与配套运行

本目录补充书中10个实践小节的完整实验。主线为一次真实运行、完整样本保存、配对比较与结果解释；研究方法的生产级复现和论文性能表不在课堂实验范围内。

## A800 运行结构

实验根目录 `/home/zcy/book_revision_labs_20261022` 下包含 `env/`、`models/`、`inputs/`、`results/`、`logs/`、`manifests/` 和 `src/repository/`。复用既有公开权重时只读，不更改其他项目。所有图像输出和逐样本记录留在本目录；代码、必要的轻量证据和书中样例同步教学仓库。

单项实验命令（独立仓库根目录，Python依赖先按仓库 requirements 安装）：

```bash
export LAB_ROOT=/path/to/your/lab_run
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TORCH_HOME="$LAB_ROOT/models/torch"
python experiments/a800/download_models.py sd15 clip clip_l14 sd14 text_filter image_filter ip_adapter
python experiments/a800/prepare_sld.py
python experiments/a800/prepare_inputs.py
python experiments/a800/prepare_weights.py cnnspot esd
python experiments/a800/generation.py
python experiments/a800/fairness.py
# ESD先按第6章说明下载并校验官方完整UNet，再执行：
python experiments/a800/esd.py
python experiments/a800/sld.py
python experiments/a800/detection.py
python experiments/a800/watermark.py
python experiments/a800/filters.py
python experiments/a800/pseudoword.py
python experiments/a800/privacy_facenet_baseline.py
python experiments/a800/measure_clip.py
python experiments/a800/analyse.py
python experiments/a800/validate_evidence.py
```

`models.json` 保存仓库及不可变revision。首次新环境运行需要下载公开模型。`prepare_sd15.py` 和 `register_caches.py` 是本次A800既有缓存的迁移辅助，学生新环境不需要使用；新环境用 `download_models.py`，遇到HEAD元数据错误可参照 `download_remaining.py` 的GET续传逻辑。

真实照片输入复制本仓库 `inputs/real_photos/` 到 `$LAB_ROOT/inputs/real_photos/`。CNNSpot权重按第4章配置放入 `$LAB_ROOT/models/blur_jpg_prob0.5.pth`，哈希不符时停止加载；不向Git提交权重。

## 实验口径

- 生成分辨率512×512、30步；除第8章正式定制使用CFG5、适配器强度0.3外，其余生成使用CFG7.5；每个条件重新创建同种子CPU生成器。A800采用FP16整模GPU推理，不沿用本机低显存卸载。
- 第一章标准库入口仍为 `chapters/01_overview/lab.py`，无显卡也能运行；`governance.py` 为本次服务器证据包装。
- 第三章当前最小伪词实践按原仓库投影嵌入核心重写：连续嵌入、余弦最近邻投影、投影点梯度直接回传和AdamW；完整L/14、16位置和红杯目标为本轮适配，历史扩展仍用原入口。搜索完成后追加实际文生图，不以CLIP高分声称真实风险绕过。
- 第四章是CNNSpot/ResNet-50空间域监督二分类，固定阈值0.5。压缩副本与原图不能当作相互独立的来源样本。
- 第五章是教学DCT重复16位水印。裁剪/缩放暴露同步局限；未嵌水印图必须保留。解码恢复消息，不恢复无水印原图。
- 第六章ESD使用官方完整UNet，不训练。SLD用非露骨成人泳装及普通提示，输出检查器保留；不报告色情抑制率。
- 第七章400样本的CLIP外观初筛与人工复核字段分离，保留不确定、检查器替换与未复核分母。二分类呈现不能代表完整公平性。
- 第八章用合成成人参考脸，FaceNet PGD保持保存后PNG像素界限8/255；JPEG会另行改变像素。实际SD1.5/IP-Adapter Plus Face的结果、适配器特征与被优化编码器分数分别核对，不把代理指标下降写成普遍身份保护成功。

最终实测数量以 `results/evidence_validation.json` 为准，设计数量不表示运行已完成。书中图表必须由本轮样本生成，并附清晰实验口径。

## 本轮固定环境安装

使用Python 3.10。先安装PyTorch 2.4.1+cu118与TorchVision 0.19.1+cu118，再安装仓库 `requirements-a800.txt`。最后单独安装 `facenet-pytorch==2.6.0 --no-deps`：上游元数据限定较早的Torch/Pillow版本，本轮采用Torch 2.4.1、Pillow 10.4的实际运行验证路径，完整版本和逐样本证据随记录保存。原有两个环境方案仍可作为拓展；不要把未运行的版本组合写成已验证。

上述批量命令调用本目录历史实现；当前书稿完整引用 examples/minimal 下的独立文件。

## 完整性与结果解释

本轮全部主实验已经实际完成。第8章先保留一轮构图不适合比较的预实验，再用正常参考图进行六组诊断，固定特写、CFG5、适配器强度0.3后重跑27个正式条件，全部单脸可判读。两个版本都归档，正式统计与预实验分开。第7章400条自动标签全部保留独立的未人工复核状态。

## 当前Git工作区

A800教学Git工作区为 `/home/zcy/book_revision_labs_20261022/github_repository`，origin对应本仓库。`src/repository`保留此前运行副本，不能当作Git工作区。实验输入、模型及各轮输出仍在实验根目录下；运行前设置 `LAB_ROOT=/home/zcy/book_revision_labs_20261022`。同步流程见[维护说明](../../docs/maintenance.md)。
