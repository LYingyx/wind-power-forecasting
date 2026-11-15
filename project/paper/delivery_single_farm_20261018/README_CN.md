# 单风场功率预测论文工作稿（2026-10-18 重写）

当前主稿已从旧 SLAST / 多风场叙事改为“单风场多步功率预测的历史轨迹条件化天气修正”。实际入口是本目录的 `main.tex`，阅读文件为 `manuscript_cn.pdf`（`manuscript.pdf` 内容相同）。

已更新摘要、引言与贡献、方法、实验、讨论、结论及附录，纳入 15 个共享命名 baseline、本站 CBH-SL、区域 E2C、四格比较、新 FP16 条件消融、FP32 天气七臂、典型风场、场景/时段与未见站适配。图 1 是新生成框架；其余八组图为真实实验数据生成的 PDF。全部 56 站逐站结果与独立 baseline pilot 独立列入附录。

## 论文主张

1. 显式历史预测轨迹参与天气修正，完整历史记忆保留；实际摘要为均值，不是 SLAST。
2. 共享训练与在线区域输入分开评价。CBH-SL 支撑本站部署；E2C 与目前的 CBH/E2C 迁移仍依赖区域历史。
3. 迁移相对从头训练有数据效率收益，但没有稳定超过强预训练 Transformer。迁移训练天数不包含 H1 checkpoint 选择标签。

本站输入 E2C / E2N 及其匹配消融尚未运行。本次没有训练新的预测模型，也没有将待补实验写成已完成结果。作者信息和文献综述/参考文献尚未补齐，本稿不是可以直接投稿的终稿；沿用原稿暂不加入相关工作的范围。

## 文件

- `main.tex`、`preamble.tex`、`sections/`：实际使用的正文与附录。
- `data/single_farm/`：从已审查 CSV 生成的 LaTeX 表格、逐站数据和来源哈希。
- `figures/architecture_single_farm_v19.png`：通过 imagegen CLI、模型参数 `gpt-image-2.5` 生成的新框架。
- `figures/imagegen_provenance/`：三轮完整 prompt、模型/模式记录、语义检查。
- `figures/single_farm/`：八组真实结果图，使用 PDF 保留矢量文字与曲线。
- `RESULTS_PROVENANCE_CN.md`：主张、表图和原始结果的对应关系。
- `AUTHOR_NOTES_CN.md`：尚需补齐的投稿要素与实验边界。
- `BUILD_CHECK.json`：数据、引用、编译和图像一致性检查。
- `manuscript_source.zip`：独立编译所需源码、图表及数据；`arxiv_source.zip` 是同内容兼容名称，不表示已满足投稿要求。

旧主稿完整备份在 `../archive_before_single_farm_rewrite_20261018/`。当前目录中历史图及 `results_single_farm_review.tex` 是保留的旧材料，不被 `main.tex` 引用，也不纳入新的编译源码包。

## 编译与复现

标准环境安装 TeX Live 的 XeLaTeX、ctex、xeCJK、Fandol 后运行：

```bash
bash build.sh
```

本机无 `xelatex` 命令时，脚本使用 `/home/liying/codex/.deps/tex-cn` 中已安装的格式文件与宏包。可设置 `WIND_PAPER_TEX_DEPS` 指向其他本地依赖目录。标准环境不需要此路径、图像 API 或模型训练环境。

在完整项目中更新表图：

```bash
python3 tools/update_single_farm_tables.py
python3 tools/prepare_figures.py
bash build.sh
python3 tools/verify_manuscript.py
python3 tools/package_source.py
```

表图重汇总只读取 `../../runs/single_farm_review_20261018/` 的已核验数据；完整预测复算脚本在该审查目录中，不因论文排版重复执行。源码压缩包中的 CSV/LaTeX/PDF 图表已就绪，编译不需要整个实验目录。
