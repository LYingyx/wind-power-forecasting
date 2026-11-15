# 分时段天气视图实验关键结果

日期：2026-09-02  
状态：实验完成

## Stage 0：零训练互补性审计

仅使用 2023 validation 和已有 10-seed 预测。对 Normal、NearestHour、NoAnchor、KD、KDNoAnchor、FullHistory 做四季度 blocked cross-validation。

| 方法 | MAE（MW） | 相对最佳单一模型改善（MW） |
|---|---:|---:|
| 最佳单一模型 KDNoAnchor | 242.102 | -- |
| 全局 convex blend | 241.641 | 0.461 |
| 三时段 convex blend | **241.547** | **0.555** |
| 三时段硬切换 | 242.389 | -0.287 |

三时段 convex blend 在短、中、长时域相对各自最佳 constituent 的改善分别为 0.483、0.488、0.348 MW。

关键结论：现有模型存在少量互补，但没有达到预注册的“总体 1 MW或任一时段 2 MW”门槛。因此停止高成本 H0/H1/H2 和异构 ensemble 训练，只继续轻量 L1/L2/L3。Sample-wise oracle 的大幅收益不可实现，不能用来支持专家模型。

机器可读结果位于 `artifacts/horizon_specialist_v1/stage0/`。

## 结构筛选

- L1（horizon anchor taper）通过：3-seed validation 平均改善 1.168 MW，3/3 seeds 获胜。
- L2（near→full weather gate）失败：平均恶化 0.530 MW，短时恶化 4.085 MW。其 ensemble 看似改善 2.042 MW，但单 checkpoint 结果不支持，因此不采纳。
- L3（gate+taper）总体改善 1.510 MW，但短时恶化 2.060 MW，违反安全约束。
- Stage 0 未显示足够互补性，因此不运行双专家 H0/H1/H2 和学习式门控 L4。

最终性能将以独立单模型为准。10 seeds 只用于稳定性统计；总体、逐 lead 和分时段结果都先按每个 checkpoint 独立计算再跨 seed 汇总，不报告十个蒸馏学生平均后的推理性能。只要新蒸馏学生相对当前 KD 单模型落在预注册的 2 MW 非劣界限内，即认为部署性能相当；ensemble 仅保留为 teacher 和方差诊断。

## 10-seed 正式结构验证

L1（horizon-dependent anchor）相对 Normal 的单模型平均差值为 -0.871 MW，10 个 seeds 中 8 个获胜，paired 95% CI 为 [-1.887, 0.145] MW，双侧 p=0.0845。短、中、长时段平均差值分别为 0.277、0.087、-1.892 MW；只有长时段 CI 完全低于 0。validation 中位代表 checkpoint 为 seed 44，MAE 257.702 MW；对应 Normal 代表 checkpoint 为 seed 44，MAE 258.363 MW。

结论：L1 没有确认总体 superiority，不能写成显著优于 Normal；但它满足在查看新 test 前冻结的安全非劣条件（2 MW 非劣、8/10 胜、各时段平均退化不超过 1 MW），因此允许继续 KD。这个继续条件只用于检验最终单学生能否保持相当性能，不把 ensemble 性能当最终结果。

## KD validation

10 个独立 KD 学生的 validation MAE 为 **249.980 ± 1.623 MW**，当前 KD 为 251.116 ± 1.617 MW。按相同 seed 配对，新学生平均改善 1.136 MW，9/10 seeds 获胜，95% CI 为 [-1.920, -0.352] MW，p=0.0096。三个时段的平均差值分别为 -0.779、-0.946、-1.382 MW，没有用短时退化换取长时收益。

冻结的代表模型选择规则是“最接近 10-seed validation 中位数的 checkpoint”。新学生与当前 KD 均选中 seed 46；validation MAE 分别为 249.298 和 250.311 MW。该选择在查看 2024 test 前完成。

## 冻结测试结果

| 单模型口径 | 新 KD 学生 | 当前 KD | 差值（新-旧） |
|---|---:|---:|---:|
| validation 预选代表 checkpoint（seed 46） | **247.233** | 250.093 | **-2.860 MW** |
| 10 个独立 checkpoint 的均值 ± 标准差（仅稳定性参考） | **249.012 ± 2.242** | 250.966 ± 3.184 | **-1.954 MW** |

代表学生 seed 46 的 test RMSE 为 363.733 MW、R² 为 0.9088、station MAE 为 10.373 MW；当前 KD seed 46 分别为 368.962 MW、0.9061 和 10.464 MW。两者参数量相同，均为 1,287,343。

10-seed test 配对差值的 95% CI 为 [-3.984, 0.076] MW，8/10 seeds 获胜。其上界远低于预注册的 2 MW 非劣界限，因此确认“单模型性能至少相当”；但 superiority 的双侧 p=0.057，不能写成测试集上统计显著优越。最稳妥的表述是：**最终可部署的单 checkpoint 保持了当前 KD 的性能，点估计更好，并且没有增加参数量。**

## 最终结论

有效机制不是 near/full 天气专家切换，而是随预测时距逐步放松历史基线约束。该机制单独使用时主要改善长时段；经过 ensemble-consensus KD 后，收益扩展到全部时段，并成功迁移到单个学生模型。

最终论文与部署结果以 validation 预选的 seed 46 单模型 247.233 MW 为准。10 seeds 只证明结果的稳定性；10-model prediction average 不作为最终成绩。Ensemble 只承担 teacher 和方差诊断角色。

机器可读统计位于 `artifacts/horizon_specialist_v1/analysis_val/` 和 `artifacts/horizon_specialist_v1/analysis_test/`。
