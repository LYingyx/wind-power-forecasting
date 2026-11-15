# IEEE BigDIA 中文底稿

本目录基于 `../conference-latex-template.zip` 中的 IEEE conference LaTeX template 创建。

主要文件：

- `manuscript_zh.tex`：中文底稿主文件，已切换为 `\documentclass[conference]{IEEEtran}`。
- `IEEEtran.cls`：来自官方 conference template。
- `IEEEtran.bst`：来自 `../IEEEtranBST2.zip`，后续英文版如改 BibTeX 可使用。
- `sections/`：从 `elsarticle/elsarticle_2_0/sections/` 迁移的中文章节，图表路径已调整到当前目录。

建议编译命令：

```bash
HOME=/storage/luoao/GNN/DP_NEW/dayahead/paper/.home \
XDG_CACHE_HOME=/storage/luoao/GNN/DP_NEW/dayahead/paper/.cache \
/storage/luoao/GNN/DP_NEW/dayahead/paper/.conda-tectonic/bin/tectonic \
  manuscript_zh.tex --outdir build
```

当前本地检查结果：`tectonic` 已能读取本目录下的 `IEEEtran.cls`，但离线环境缺少 Tectonic 的基础字体 bundle，编译停在 `lmroman10-regular` 字体加载。联网允许 Tectonic 拉取 bundle 后应可继续检查正文、图表和页数。
