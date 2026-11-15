**A6 follow-up 结果入口（2026-09-13）**

完整结果、真实架构、统计方法、机制证据边界及 56 场站逐站对照已整理到：

[完整中文报告](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/audit_workspace_20260913/temporal_alignment_followup_review_20260913/REPORT_CN.md)

A6-N7 相对 clean A1 的十 seed 验证 CF MAE 下降 0.000463（9/10 胜出，配对 seed bootstrap 95% CI [−0.000706, −0.000227]）；2024 跨年评估下降 0.000735（10/10 胜出）。第八个天气节点不是获取此收益的必要条件。R3/R4 的三 seed 对照尚未证明 native 表示优于插值表示，也未证明天气 anchor 有害。

本批完成 A6N8/A6N7/R3/R4N/R4A；新的 local/global、weather encoder、history early/late 和 A6 matched mechanism controls 没有已完成结果。当前建议固定 A6-N7 为后续机制研究候选。

- [全部总体指标](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/audit_workspace_20260913/temporal_alignment_followup_review_20260913/aggregate_metrics.csv)
- [逐场站 MAE/MSE/RMSE/R²/MW 指标](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/audit_workspace_20260913/temporal_alignment_followup_review_20260913/station_metrics_by_model.csv)
- [配对统计](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/audit_workspace_20260913/temporal_alignment_followup_review_20260913/paired_seed_comparisons.csv)
- [时距增益图](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/audit_workspace_20260913/temporal_alignment_followup_review_20260913/horizon_gain.png)
- [本轮 checkpoint 重放检查](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/audit_workspace_20260913/temporal_alignment_followup_review_20260913/checkpoint_replay.csv)
