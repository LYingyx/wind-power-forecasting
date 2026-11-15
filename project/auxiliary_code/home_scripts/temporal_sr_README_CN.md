# temporal_sr_original_cbh_20260917_v1

这是“小时 NWP → 潜在 15 分钟代理天气 → 原 CBH weather-to-power mapping”的隔离探索目录。没有修改或覆盖既有框架、checkpoint 和旧实验结果。

本批固定 10 个非爬坡筛选场站、seeds 42–51。原 CBH 直接评测 10 个 seed；9 条单站适配路线各完成 10 站×10 seeds，共 900 个 station×seed 训练结果。完整性、预测重算、样本聚合、checkpoint/protocol/hash、生成器来源和下游梯度审计全部通过。

主要结论：交接文档确定的原 CBH 在所选 10 站上为 `0.082244 ± 0.000549` MAE，优于全部潜在高频适配路线。适配路线中 LINEAR 为 `0.086943 ± 0.000684`；最好的非 LINEAR 方法 DET 为 `0.086976 ± 0.000746`，相对 LINEAR 的差异不确定。当前证据不支持用这批生成路线替换原 CBH。

这里从未假设存在真实 15 分钟天气。中间天气是潜变量，主要按下游 15 分钟功率预测评价；4h→1h/2h→1h 天气遮盖只作为正则与诊断。

建议阅读顺序：

1. `analysis/FINAL_REPORT_CN.md`：最终结论、全部模型均值/SD/最低/最高、配对统计和下一步；
2. `analysis/RESEARCH_AND_PROTOCOL_CN.md`：原 CBH 身份、生成后连接、损失、站点与统计协议；
3. `analysis/LITERATURE_SYNTHESIS_CN.md`：CSDI、TimeBridge、I²SB/DDBM、flow matching、EDM、MeanFlow 和天气降尺度边界；
4. `analysis/METHOD_MATRIX_CN.md`：不同 flow/diffusion 的 path、target、prior、solver 区分；
5. `status/FINAL_AUDIT.json`：900/900 结果与哈希审计；
6. `status/CBH_GENERATOR_GRADIENT_AUDIT.json`：只用功率 MAE 时对八种生成器的非零梯度。

统计明细位于 `analysis/overall_seed_summary.csv`、`analysis/per_station_seed_summary.csv`、`analysis/paired_comparisons.csv`、`analysis/proxy_summary.csv`。逐 station×seed 的 checkpoint、预测、样本、训练历史和代理指标位于 `results/`。原 CBH 的直接预测与 baseline cache 位于 `original_eval/` 和 `cache/`。最终源码快照和 SHA256 位于 `vendor/final_code_snapshot/`、`vendor/generator_snapshot/` 和 `references/final_code_hashes.json`。

上一批实际命名 baseline（DLinear、PatchTST、HistGradientBoostingRegressor、TiDE、TFT、NBEATSx）以及 TiDE 后接八种天气生成器的 2000 个结果保留在相邻目录 `temporal_sr_named_models_20260917_v2`；其中 `P0I_CBH_SINGLE_*` 只作为简化控制，不再代表原 CBH。
