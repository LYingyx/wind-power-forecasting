# 论文证据补充：已完成

2026-11-19 更新。最新补充批次已于 2026-11-18 14:13 完成 **151/151** 项任务：121 次训练（含 16 次调参）及 30 次已有权重复评。

- [最新科学问题及全部结果的 Gemini 阅读入口](../docs/GEMINI_REVIEW_20261119_CN.md)
- [基线与消融表](runs/daily_paper_evidence_20261118_v10/analysis/BASELINE_ABLATION_SUMMARY_CN.md)
- [完成标志](runs/daily_paper_evidence_20261118_v10/COMPLETE.json)
- [逐模型、种子、年份指标](runs/daily_paper_evidence_20261118_v10/analysis/METRIC_RECORDS.json)
- [任务配置与训练记录](runs/daily_paper_evidence_20261118_v10/analysis/TASK_EVIDENCE.json)
- [评估加载修复审计](runs/daily_paper_evidence_20261118_v10/operations/evaluation_loader_repair_20261118/VERIFIED.json)

原 README_CN.md 和 BASELINE_AUDIT_CN.md 保留启动／中途快照，其中“待补”“尚未完成”不代表当前状态。新增窗口基线、线性基线、精确消融、十种子扩展及两年复评均已完成。模型和数据协议未修改；五个线性基线只修复了训练结束后的模型加载分派。

整体模型相对线性插值的净收益获得支持，但不能宣称全部内部机制或相对强 Transformer 的完整门槛均已通过。左侧状态摘要的收益跨年不稳定；当前主模型尚未移除此路径。
