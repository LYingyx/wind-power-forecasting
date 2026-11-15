# 中文论文工作稿

当前版本包含引言、方法与实验结果，暂不含研究现状、相关工作、摘要和参考文献。依据已保留的 SLAST 实现撰写，作者信息留空。

## 阅读与编辑

- `manuscript_cn.pdf`：中文版；`manuscript.pdf` 为同一阅读版。
- `main.tex`：主入口；`preamble.tex`：公共设置。
- `sections/introduction.tex`：中文引言。
- `sections/method.tex`：中文方法及数学定义。
- `sections/results.tex`：实验协议、8 个配置的对比表、分协议核心消融表、配对置信区间表及结果解释。
- `figures/architecture_overview.png`：通过 imagegen CLI 请求 `gpt-image-2.5` 重绘的 v18 框架，最终为 1536×884；实际查看 NeurIPS 2025 TiRex、ICLR 2026 CoRA/STORM 方法图后，重新组织为白底、细线、低饱和分组与紧凑的图形示意。模块连线保留水平、垂直和直角路由；基准、历史、直接天气及独立目标上下文路径均核对。模型输出仅裁掉外围白边，没有使用代码绘制或修补结构。
- `RESULTS_PROVENANCE_CN.md`：表格的来源与可比范围。
- `AUTHOR_NOTES_CN.md`：实现核对与解释边界。
- `archive/english_v1/`：上一版英文 PDF、源码包和说明。

当前主对比表是最终骨干的已完成结构对比，尚非完整外部算法基准表。原实验目录在当前环境未挂载，表格使用两份已核验的本地汇总；未凭空补充外部模型结果、标准差、RMSE 或总参数量。取得原完整结果后，可另补统一协议的外部对比。

## 编译

标准环境需要 TeX Live 的 XeLaTeX、ctex、xeCJK 和 Fandol 字体：

```bash
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

本机已将缺失宏包解包至 `../.deps/tex-cn`，不修改系统安装。直接运行：

```bash
bash build.sh
```

`build.sh` 优先使用标准 XeLaTeX；本机备用路径使用项目内宏包与生成的格式文件。正文源码不包含该绝对路径。无需图像 API、Python、实验文件或 shell escape 即可在标准 TeX Live 环境编译正文。

## 源码打包

```bash
python3 tools/package_source.py
```

`arxiv_source.zip` 只包含主文件、宏定义、三个章节和实际使用的图片。中文稿请选择 XeLaTeX，不再使用原英文稿的 pdfLaTeX。arXiv 没有统一专属版式；本稿采用标准 ctexart 单栏 11 pt、letter paper、1 inch 页边距。正式提交前仍需补齐作者、摘要、引用和剩余论文内容，并检查平台生成的 PDF。

当前图像的 prompt 与修订记录汇总于 `../output/imagegen/framework_research_v18/PROMPTS.md`；实际参考来源与取舍见同目录 `STYLE_REVIEW_CN.md`，概念与实现对应见 `CONCEPT_MAPPING_CN.md`。`tools/prepare_figures.py` 仅拷贝最终图。替换前的 v17 中文 PDF、源码包和检查记录保存在 `archive/chinese_figure_v17/`；更早版本仍保留。参考论文的原图不进入正文或源码包。
