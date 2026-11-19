# Gemini 纯文本读取备用入口

**2026-11-19 最新材料：优先读取 [08 科学问题与完整实验结果](08_paper_review_20261119.txt)，需要核对实现时再读取 [09 当前模型与评估源码](09_current_model_code_20261119.txt)。** 两份均为 UTF-8 明文，可直接下载后作为文本交给 Gemini。来源校验见 [PAPER_REVIEW_MANIFEST_20261119.json](../PAPER_REVIEW_MANIFEST_20261119.json)。旧 01–07 文件保留原版本，不能代替 08/09 的最新结论。

以下历史七个 `.txt` 是 UTF-8 明文，包含原始文件正文，不是 ZIP。可在 GitHub 中打开单个文件，选择 **Raw / Download raw file** 下载，然后逐个上传到 Gemini。不要使用整个目录的 ZIP 作为普通附件。查看 2024 测试结果优先读第 7 份；第 6 份是 2023 验证结果；第 2 份只包含此前发布的实现，不能假定覆盖 v4–v8 全部源码。

1. [01_context_and_results.txt](01_context_and_results.txt)：当前协议、结果、计划及结果到代码的导航。
2. [02_models_and_training.txt](02_models_and_training.txt)：三轮已完成实验及 11-16 v2 新实验的全部 `code/` Python 文件；内容完全相同的副本只显示一次，全部原路径保留。
3. [03_farm_seed_horizon_metrics.txt](03_farm_seed_horizon_metrics.txt)：逐风场、逐种子、分时效指标。
4. [04_data_selection_evaluation.txt](04_data_selection_evaluation.txt)：天气准备和审计、实际选型配置、汇总计算及零训练诊断源码。
5. [05_current_experiment_results.txt](05_current_experiment_results.txt)：11-16 v2 当前结果、逐任务完整指标和诊断、实际配置、训练/验证曲线、选中轮次与来源哈希，以及逐种子/逐风场 CSV；完整五种子与局部配对结果分列，运行中任务不计作最终成绩。
6. [06_completed_experiments_20261117.txt](06_completed_experiments_20261117.txt)：11-17 完成报告（2023 验证集）、v4–v7 的 105 次新训练与配对证据、v8 两 seed 采样审计；保留正负反馈、门槛未通过原因和未开展的验证。该文件是结果证据包，不是全部新训练源码包。

7. [07_test2024_results_20261117.txt](07_test2024_results_20261117.txt)：2024 跨年测试报告、120 个冻结检查点的逐风场逐种子证据及两年配对比较；不包含原始数据、权重或全部训练源码。

实际训练过程见[全部 2571 项任务的训练记录](../../project/analysis/daily_protocol_review_20261115/provenance/)。三个 JSON 含实际配置、逐轮曲线、选中轮次、指标摘要与原件哈希；可以按需单独下载上传。

[MANIFEST.json](MANIFEST.json)列出各 TXT 及其来源文件的 SHA-256。TXT 是针对当前实验链条的阅读副本，完整仓库仍应通过 Gemini 的“导入代码”入口导入。原始气象/功率数据、权重和全量预测保留在研究服务器。

重新导出及检查命令见[结果到代码导航](../RESULTS_TO_CODE_CN.md)。检查能验证交付内容，不保证第三方网页解析器成功；导入后应先要求 Gemini 引用实际函数和训练记录，再让它分析机制。
