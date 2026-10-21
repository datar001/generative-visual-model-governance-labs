# 环境配置

以下固定组合用于与本书示例对齐；并不宣称所有平台都已实测。安装前确认Python为3.10。不要把不同实验环境混装。

## 第1章：标准库环境

```powershell
python -m venv .venv-standard
.venv-standard\Scripts\Activate.ps1
python tools/check_environment.py --profile standard
python tools/run.py 01
```

macOS/Linux激活方式为 `source .venv-standard/bin/activate`。若系统禁止PowerShell激活脚本，可直接调用虚拟环境中的python.exe，无须修改系统执行策略。

## 第2、4、6、7章与第8章定制阶段

```powershell
python -m venv .venv-generation
.venv-generation\Scripts\Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-generation.txt
python tools/check_environment.py --profile generation --with-torch
```

上面是CPU安装命令，适用于配置检查、CPU检测及能够接受慢速推理的示例。CUDA设备可将PyTorch安装源替换为官方兼容构建，例如cu121；先核对驱动与设备，再确认torch.cuda.is_available()。单独安装CUDA工具包不能证明PyTorch可用GPU。本书已有GTX 1660 SUPER单精度生成记录；当前代码对GTX 16系列使用float32及CPU卸载。ESD当前实现同时保留教师、学生和优化状态，本机6GB显存不足；第8章SDXL定制也需要额外资源。CPU安装环境不能直接运行代码中明确要求CUDA的ESD/定制阶段。

首次模型下载需要网络、权重存储空间和相应访问许可。模型缓存可通过HF_HOME配置，不放入本仓库。模型标识符、分支/修订号和软件版本应写入实验记录。

## 第5章：轻量CPU水印环境

```powershell
python -m venv .venv-watermark
.venv-watermark\Scripts\Activate.ps1
python -m pip install -r requirements-watermark.txt
python tools/check_environment.py --profile watermark
python tools/run.py 05
```

没有第2章杯子图时，脚本使用明确标记的程序图验证编解码；不能将其与历史杯子结果混为同一实验。也可以复制evidence/generation/cup_0.png到results/generation/cup_0.png作历史输入对照，并在报告中说明来源。

## 第8章：独立FaceNet环境

```powershell
python -m venv .venv-privacy
.venv-privacy\Scripts\Activate.ps1
python -m pip install -r requirements-privacy.txt
python tools/check_environment.py --profile privacy --with-torch
python tools/run.py 08 --case features
```

准备inputs/face.png后才运行特征阶段；FaceNet权重首次也需下载。然后退出该环境，在生成环境执行 `python tools/run.py 08 --case customize`。两阶段通过results/privacy中的图像衔接。权重与授权照片不提交Git。

## 如何检查、如何报告失败

先运行各章的 --dry-run 确认脚本入口，再按对应环境执行。run.py为每次运行保存stdout、stderr、退出码、软件版本和代码哈希；模型下载、显存与数值失败也应保留。进程退出码0仅说明脚本执行结束，学习目标和输出质量仍需按章复核。

参考：[PyTorch 2.5.1固定版本](https://pytorch.org/get-started/previous-versions/#v251)、[Diffusers 0.32.2加载接口](https://huggingface.co/docs/diffusers/v0.32.2/en/using-diffusers/loading)、[SLD固定接口](https://huggingface.co/docs/diffusers/v0.32.2/en/api/pipelines/stable_diffusion/stable_diffusion_safe)。

## 第3章：独立过滤器环境

本章不需要图像生成模型或显卡。关键词模式只用Python标准库；完整对照使用两个固定版本的英文NSFW分类模型。

```powershell
python -m venv .venv-filters
.venv-filters\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-filters.txt
python tools/check_environment.py --profile filters --with-torch
python tools/run.py 03
```

离线词表练习：`python chapters/03_compliance/lab.py --keyword-only`。模型首次需下载约六百MB权重；之后可以使用本地缓存。模型修订号、阈值和词表保存在章节filter_config.json，不与第2章图像生成安全检查器混用。模型加载或推理失败将保留unavailable并返回非零退出码，不当作审核通过。默认使用CPU。


### 第3章伪词目标匹配

沿用过滤器环境。官方CLIP模型首次约600MB下载，后续可用本地缓存；默认CPU，不调用LLM API。运行 `python tools/run.py 03 --case pseudoword`。可选安装 `python -m pip install matplotlib==3.8.4` 后执行 `python chapters/03_compliance/plot_pseudoword.py`，复绘results/pseudoword_cup中的实际轨迹。该实验目标是已提供的无害红杯图，只演示CLIP匹配搜索，不连接真实安全审核或文生图流程。
