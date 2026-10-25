# 本地维护与GitHub同步

本仓库独立于整书工程，按8章组织，避免学生必须下载完整书稿、字体或模型权重。每章README、程序、习题和答案同目录保存。

本地维护流程：
1. 修改本仓库章节材料；书稿中的对应实践与答案也应同步更新。
2. 执行 `python tools/validate_repository.py`，检查编号、题目答案对应和程序语法。
3. 对改动涉及的离线任务或模型任务做相应验证，保存新结果；不要把历史证据冒充新运行。
4. 使用 `git diff` 查看范围，然后只暂存本次修改的文件，再执行 `git commit -m "Describe the teaching update"`、`git push origin main`。
5. 查看远端提交号，确认本地HEAD与远端main一致。

本仓库的content/curriculum.json保存本轮教学结构化源数据。要批量重生成书稿，使用整书审计目录的构建脚本前必须先备份并检查变更范围；该脚本不是面向学生的日常入口，不应覆盖作者后续手工修改。

仓库初建为私有，分享给学生前需授予访问权限或由作者决定改为公开。仓库不自动附加开源许可证；对外再分发前由作者确认代码与文字的许可方式，以及第三方模型和库的使用条件。

## 仅维护习题与答案

`content/curriculum.json`中的每道题保持三元组格式：`[具体题目标题, 完整题干, 参考答案]`。第一项是题目标题，不再是固定类别；保留每章12题和“章号-两位题号”。答案须回应题干的全部子问，计算题写明公式、代入和分母，开放题提供可直接参考的方案。教学假设数据须明示，不能写成模型实测。

修改questions字段后，运行 `python tools/render_exercises.py --write` 只更新各章习题、答案Markdown。运行 `python tools/render_exercises.py` 检查是否同步；运行 `python tools/validate_repository.py` 检查全仓库。答案文件同时包含题干，方便逐题核对。

维护者可附加 `--book-root <整书book目录>` 同步章末习题与答案；工具保留“习题”标记之前的字节，并在检测到额外章节结构时中止。运行前仍须备份。整书审计目录的旧版全量教学生成器包含实践重写逻辑及固定类别假设，不适合本轮格式，不应用于日常习题同步。

2026-10-25修订覆盖8章96题，取消三类分组，补齐问题条件、推理与计算。用户已经修订的动手实践不属于本次生成范围。

## 当前本地与A800位置

- 本地Git工作区：`E:\ondrive_clone\paper\book_revision\visual-model-governance-labs`。
- GitHub origin：`https://github.com/datar001/generative-visual-model-governance-labs.git`，分支 `main`。
- A800 Git工作区：`/home/zcy/book_revision_labs_20261022/github_repository`。
- A800实验数据根目录：`/home/zcy/book_revision_labs_20261022`。其中 `src/repository` 是先前运行副本，不是Git工作区，保留用于核对历史执行。

从A800 Git目录运行当前示例时，设置 `LAB_ROOT=/home/zcy/book_revision_labs_20261022`，复用已有模型与输入，输出仍按各程序的独立目录保存。不要用Git操作覆盖实验结果。

2026-10-25整理时A800无法解析GitHub域名，因此先从本地提交并推送，再将同一Git历史通过bundle离线克隆到上述独立目录，origin仍设置为GitHub。后续网络恢复可正常fetch/pull；也可继续传递Git bundle并以快进方式同步，先检查工作区干净及提交关系。每次同步须核对本地、GitHub和A800的HEAD一致。

发布前确认 `examples/minimal/print_manifest.json` 与源码一致；仅分发已遮挡的敏感展示资产。`evidence/sld_only_20261025/`为作者未遮挡历史实验，本地保留但被Git忽略；当前读者入口使用 `evidence/sld_sweep_20261025/`。
