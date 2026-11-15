# 实验章节数据来源（2026-10-17）

本轮为写作整合，不启动训练，也未重新计算原预测文件。

## 可访问的核验来源

1. `/home/liying/codex/paper/MODULE_EVIDENCE_AND_DECISIONS_20261016_CN.md`
2. `/home/liying/codex/paper/SCIENTIFIC_QUESTIONS_AND_CONTRIBUTIONS_20261016_CN.md`

这两份记录包含原报告索引、结果和科学解释边界。当前机器上原 `/storage/liying/new_try/wind_cluster_forced_evolution` 未挂载；已向用户询问迁移位置。以下表格依据上述已核验汇总，并非本轮重新读取 checkpoint 或逐 seed 预测的独立复算。

## 表格映射

- `tab:main-comparison`：来源 1 的第 3 节。8 个配置的两年均值和新增摘要参数逐项转录。
- `tab:ablation`：来源 1 的第 4 节与来源 2 的 Q1/Q2、关键数值表。canonical、最终骨干和迁移实验分别分组。JM−G 与 PJ−P0 的胜出数当前汇总未提供，保留破折号，不推测。
- `tab:summary-paired`：来源 1 的第 3 节。显示单位乘以 1000，不重新估计置信区间。
- CBH 置换/Jacobian：来源 2 的 Q3；保留约数和 CBH 标签。
- SLAST 相对 CB、CBH 的百分比：由六位小数均值计算并四舍五入至两位百分数，不作为新的配对统计。

## 比较范围

主对比表是最终骨干上的已完成结构比较，尚不是外部算法基准大表。未填入未经核对协议的 Transformer、PatchTST 等其他方法，也未从不同协议的绝对 MAE 推导排名。取得原始完整报告后可另补统一协议的外部对比表。

已有 canonical W/History 等比较的差分可以使用，但不以缺失的原始绝对均值反推新数据，不虚构 RMSE、标准差、参数总数或推理耗时。

当前 SLAST 选择不把历史 CBH 诊断迁移为 SLAST 结果。2023 为反复开发使用的验证年，2024 亦已查看，只称跨年复评。
