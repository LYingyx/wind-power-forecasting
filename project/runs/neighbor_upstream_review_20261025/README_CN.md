# 邻站结果和上游机制复核

本目录只含可复用的审查与补充线性诊断，不是新的神经网络训练队列。

- [完整结论与修订方案](../../NEIGHBOR_UPSTREAM_REVIEW_20261025_CN.md)
- [结果复核审计](REVIEW_AUDIT.json)、[诊断完成审计](CORRECTED_DIAGNOSTIC_AUDIT.json)、[数值一致性检查](CHECKS.json)
- [模型比较](model_comparisons.csv)、[场景比较](scenario_comparisons.csv)、[干预](intervention_comparisons.csv)
- [旧/新线性诊断汇总](linear_diagnostic_summaries.csv)、[修订诊断逐对原值](paired_predictability_corrected.csv)
- [诊断脚本](paired_diagnostic.py)、[汇总脚本](review_results.py)、[原文件哈希](REVIEW_SOURCES.json)

原神经网络逐风场逐seed指标保留在上一轮目录；这里不复制权重、天气、功率或预测数组。旧上游诊断原文件保持不变。修订线性复核仅使用正式train/val缓存，固定15min邻站延迟；中点风由两端插值风的均值近似，不能称旧算法的完全复现。原始相关峰的独立年度验证尚未进行。
