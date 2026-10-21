# 接口核对依据

- [Diffusers 0.32.2：加载管线](https://huggingface.co/docs/diffusers/v0.32.2/en/using-diffusers/loading)
- [Diffusers 0.32.2：Safe Stable Diffusion](https://huggingface.co/docs/diffusers/v0.32.2/en/api/pipelines/stable_diffusion/stable_diffusion_safe)
- [GitHub：管理远端](https://docs.github.com/en/get-started/git-basics/managing-remote-repositories)
- [GitHub REST：创建仓库](https://docs.github.com/en/rest/repos/repos#create-a-repository-for-the-authenticated-user)

这些链接说明接口和维护方式；不构成代码已运行或模型效果已验证的证据。


## 第3章三过滤器对照（2026年10月21日核对）

参考[作者仓库query_T2I.py](https://github.com/datar001/metaphor-based-jailbreaking-attack/blob/d135d0195afe9b7963aaee02c55fae7cee8281cc/query_T2I.py)中的关键词、Text_Filter与Img_Cls接口，独立编写教学代码。文本模型旧ID michellejieli/NSFW_text_classifier已重定向至[michelleli99/NSFW_text_classifier](https://huggingface.co/michelleli99/NSFW_text_classifier)；图像模型为[Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)。当前修订号、风险类别与0.5阈值见章节filter_config.json，下载文件哈希见evidence/filter_comparison_20261021/model_snapshots.json。


## 伪词目标匹配教学类比

参考[定向攻击仓库](https://github.com/datar001/Revealing-Vulnerabilities-in-Stable-Diffusion-via-Targeted-Attacks/tree/21eb35f16697529388a43bf1504089eccc6bc432)的目标图、预算与评估组织，以及[MJA](https://github.com/datar001/metaphor-based-jailbreaking-attack/tree/d135d0195afe9b7963aaee02c55fae7cee8281cc)的方法定位。原定向方法包含替换/附加词和梯度优化，教学版使用独立的无害有限词表搜索，未宣称完整复现原攻击。编码器采用[官方CLIP](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)。
