# 当前实验结果总入口：逐风场、逐随机种子

本轮独立单风场实验的9500个训练阶段已全部完成，最终完成审计于2026-10-21 23:03:42通过。

本次按用户要求重新整理：保留每个风场、每个随机种子的实际MAE/RMSE；不跨种子或风场取均值。缩写对应的真实结构、输入、输出锚点和实际配置均单独解释。候选误差只要严格低于对照就记作改善，不设置百分比门槛，同时保留全部退化与相同情况。

- [Excel总表](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/EXPERIMENT_RESULTS_PER_SEED_20261022.xlsx)
- [结果审查与设计结论](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/REPORT_CN.md)
- [模型缩写、结构及实际超参数](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/ARCHITECTURES_CN.md)
- [离线交互浏览器](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/index.html)
- [完整交付包](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/RESULTS_BUNDLE.zip)
- [导出检查](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/VALIDATION.json)
- [独立来源复核](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/INDEPENDENT_VALIDATION.json)

主表覆盖全部9500阶段，包括正式、参考、消融及调参；5个seed的消融不补造其余5个seed。持续性没有随机种子；E10是原验证期选定基线的结果复用，表中说明其实际family和来源job，不计为新增实验。

分时效、分场景、逐日、逐epoch和固定案例逐seed预测见结果报告内链接。历史共享、迁移及旧独立实验见[单独索引](runs/independent_single_farm_20261021_v1/seedwise_review_20261022/legacy/README_CN.md)，不与本轮混合。

接手项目的其他模型应先读[项目总览](PROJECT_HANDOFF_CN.md)顶部最新更新，再读本次结果报告。旧计划中的效果预期不能替代已经完成的实测结果；完整E2C-CBH并非在所有设计或指标上都最优。
