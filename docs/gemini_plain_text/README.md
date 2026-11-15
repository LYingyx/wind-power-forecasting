# Gemini 纯文本读取备用入口

这四个 `.txt` 是 UTF-8 明文，包含原始文件正文，不是 ZIP。可在 GitHub 中打开单个文件，选择 **Raw / Download raw file** 下载，然后逐个上传到 Gemini。不要使用整个目录的 ZIP 作为普通附件。

1. [01_context_and_results.txt](01_context_and_results.txt)：当前协议、结果、计划及结果到代码的导航。
2. [02_models_and_training.txt](02_models_and_training.txt)：三轮实验的全部 `code/` Python 文件；内容完全相同的副本只显示一次，全部原路径保留。
3. [03_farm_seed_horizon_metrics.txt](03_farm_seed_horizon_metrics.txt)：逐风场、逐种子、分时效指标。
4. [04_data_selection_evaluation.txt](04_data_selection_evaluation.txt)：天气准备和审计、实际选型配置、汇总计算及零训练诊断源码。

实际训练过程见[全部 2571 项任务的训练记录](../../project/analysis/daily_protocol_review_20261115/provenance/)。三个 JSON 含实际配置、逐轮曲线、选中轮次、指标摘要与原件哈希；可以按需单独下载上传。

[MANIFEST.json](MANIFEST.json)列出各 TXT 及其来源文件的 SHA-256。TXT 是针对当前实验链条的阅读副本，完整仓库仍应通过 Gemini 的“导入代码”入口导入。原始气象/功率数据、权重和全量预测保留在研究服务器。

重新导出及检查命令见[结果到代码导航](../RESULTS_TO_CODE_CN.md)。检查能验证交付内容，不保证第三方网页解析器成功；导入后应先要求 Gemini 引用实际函数和训练记录，再让它分析机制。
