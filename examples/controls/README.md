# 独立对照入口

这些程序补充当前书内最小实践，运行设置与主例分开记录。

## 第8章：只关闭输出检查器

先按[第8章主例](../../chapters/08_privacy/README.md)运行当前最小代码，保留同一 `LAB_ROOT` 下的三张参考PNG、`settings.json`、`input_metrics.json` 和 `outputs.json`，再运行：

```bash
python examples/controls/08_privacy_checker_off.py
```

本程序读取 `$LAB_ROOT/minimal_examples_20261025/08_privacy/`，复用已保存的原图、随机图和保护图，不重新优化。输出单独写入 `$LAB_ROOT/privacy_checker_off_20261025/`。使用相同提示、种子、模型和生成设置，仅禁用输出检查器；FaceNet只评价，不过滤图片。`output_flag=null`表示未执行检查。

[已保存的9张配对结果及解释](../../evidence/privacy_checker_off_20261025/README.md)。该结果来自先前实测，本次整理只将脚本根目录改为环境变量 `LAB_ROOT`；执行时的原脚本仍原样保留在证据目录。
