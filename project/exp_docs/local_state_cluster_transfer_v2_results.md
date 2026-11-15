# Bilinear / Fixed 4×4 × Add / Interaction 集群迁移实验

## 目的与冻结协议

本实验不是继续搜索结构，而是补足最后一个归因控制：已在 station-only 阶段观察到的 history–weather 非可分离性，经过 cluster-aware fine-tuning 后是否仍然成立。完整矩阵为两种气象支持乘三种响应：共享输出 Add、等参数独立双分支 Add、Interaction。独立 Add 使用互不共享的 history/weather 分支，且响应模块与 Interaction 参数量严格相同。

56 个场站全部保留；2022 train、2023 validation、冻结后 2024 test；seeds 42–51。先用 station CF loss 预训练 40 epochs，再从对应 checkpoint 自由微调 20 epochs，目标为 `station loss + 2 × cluster loss`。主结论要求 Interaction 同时超过两个 Add 对照。

## 主结果

| 模型 | 2023 station CF MAE | 2024 station CF MAE | 2023 transfer cluster MAE (MW) | 2024 transfer cluster MAE (MW) |
|---|---:|---:|---:|---:|
| `b_shared_add` | 0.089 ± 0.000 | 0.086 ± 0.001 | 264.622 ± 3.707 | 262.354 ± 2.093 |
| `b_independent_add` | 0.090 ± 0.000 | 0.086 ± 0.001 | 263.861 ± 2.286 | 263.816 ± 2.129 |
| `b_int` | 0.088 ± 0.000 | 0.085 ± 0.000 | 266.900 ± 3.562 | 259.428 ± 2.808 |
| `f_shared_add` | 0.089 ± 0.000 | 0.086 ± 0.000 | 265.962 ± 3.152 | 263.788 ± 2.118 |
| `f_independent_add` | 0.090 ± 0.000 | 0.086 ± 0.000 | 265.312 ± 1.830 | 265.411 ± 2.543 |
| `f_int` | 0.087 ± 0.000 | 0.085 ± 0.000 | 264.563 ± 2.830 | 259.956 ± 3.069 |

配对项为 candidate−control；负值更好，括号为配对 95% CI 与获胜 seeds。

| Interaction 比较 | 2023 station | 2024 station | 2023 cluster transfer | 2024 cluster transfer |
|---|---:|---:|---:|---:|
| `b_int-minus-b_shared_add` | -0.001773 [-0.001929, -0.001617], 10/10 | -0.001218 [-0.001688, -0.000748], 10/10 | +2.278 [-1.538, +6.094], 3/10 | -2.927 [-4.524, -1.330], 10/10 |
| `b_int-minus-b_independent_add` | -0.001907 [-0.002059, -0.001756], 10/10 | -0.001329 [-0.001776, -0.000881], 10/10 | +3.039 [+0.485, +5.594], 1/10 | -4.388 [-6.570, -2.207], 9/10 |
| `f_int-minus-f_shared_add` | -0.001951 [-0.002198, -0.001703], 10/10 | -0.001379 [-0.001737, -0.001022], 10/10 | -1.399 [-4.633, +1.835], 6/10 | -3.831 [-6.837, -0.825], 9/10 |
| `f_int-minus-f_independent_add` | -0.002179 [-0.002386, -0.001972], 10/10 | -0.001427 [-0.001723, -0.001131], 10/10 | -0.749 [-2.911, +1.413], 7/10 | -5.454 [-8.666, -2.243], 10/10 |

## Bilinear 与 Fixed 4×4

| Fixed−Bilinear | 2023 station | 2024 station | 2023 cluster transfer | 2024 cluster transfer |
|---|---:|---:|---:|---:|
| `f_shared_add-minus-b_shared_add` | -0.000090 [-0.000187, +0.000007], 8/10 | -0.000002 [-0.000261, +0.000257], 8/10 | +1.340 [-1.259, +3.939], 4/10 | +1.433 [-0.608, +3.475], 3/10 |
| `f_independent_add-minus-b_independent_add` | +0.000003 [-0.000131, +0.000137], 6/10 | -0.000065 [-0.000321, +0.000190], 5/10 | +1.452 [-0.249, +3.152], 3/10 | +1.595 [-0.548, +3.737], 3/10 |
| `f_int-minus-b_int` | -0.000268 [-0.000461, -0.000075], 9/10 | -0.000164 [-0.000393, +0.000065], 7/10 | -2.337 [-5.283, +0.609], 8/10 | +0.529 [-1.653, +2.711], 4/10 |

Fixed 4×4 是否特异增强 interaction 由 support-DID 判断：

| Add 对照 | 2023 transfer DID | 2024 transfer DID |
|---|---:|---:|
| `shared_add` | -3.677 [-7.467, +0.112], 8/10 | -0.904 [-4.121, +2.313], 5/10 |
| `independent_add` | -3.789 [-5.688, -1.890], 9/10 | -1.066 [-4.641, +2.509], 5/10 |

## 求和误差协方差分解

严格使用导出的 56 场 MW 残差验证：

\[
\operatorname{MSE}(\sum_i e_i)=\operatorname{Bias}(\sum_i e_i)^2
+\sum_i\operatorname{Var}(e_i)+2\sum_{i<j}\operatorname{Cov}(e_i,e_j).
\]

下表为 transfer-stage Interaction 的 Fixed4−Bilinear 配对差。`cancellation_ratio=-cross/diagonal`，正值表示跨场协方差帮助抵消误差。只有 cross-covariance 的配对区间跨年稳定不跨 0，才把“4×4 改变空间误差相关性”写成结论。

| 分解项 | 2023 | 2024 |
|---|---:|---:|
| `aggregate_bias_squared_mw2` | -607.8 [-1639.5, +423.8], 7/10 | -308.6 [-907.3, +290.1], 6/10 |
| `diagonal_variance_mw2` | -219.2 [-461.1, +22.7], 7/10 | -172.6 [-425.6, +80.3], 8/10 |
| `cross_covariance_mw2` | -1724.5 [-4276.4, +827.5], 7/10 | +1144.4 [-1010.1, +3299.0], 4/10 |
| `cancellation_ratio` | +0.020182 [-0.049708, +0.090073], 4/10 | -0.102324 [-0.170422, -0.034225], 9/10 |

所有 horizon（all、0–1 h、1–3 h、3–6 h）、station/transfer 两阶段和逐 seed 的精确分解均在 `analysis/summary.json`。最大恒等式闭合误差为 `2.037e-10` MW²。

## 严格结论与路线判定

- Station interaction：`{"b": true, "f": true}`
- Cluster-transfer interaction：`{"b": false, "f": false}`
- Fixed 4×4 特异增强 interaction：`False`
- Fixed 4×4 稳定增加跨场协方差：`False`

结论：interaction is identifiable for farms but does not survive the cluster target; use farm forecasting as the mechanism-focused primary task and cluster aggregation as the scale-effect extension.

这里的“interaction”只表示历史发电与未来 Pangu 在预测上不可由匹配的独立主效应替代；没有额外观测时，不把它强行解释为已辨识的风机内部因果状态。不能按结果筛选场站形成主结论。

## 证据文件

- `artifacts/local_state_cluster_transfer_v2/selection/validation_freeze.json`
- `artifacts/local_state_cluster_transfer_v2/source_checkpoint_audit.json`
- `artifacts/local_state_cluster_transfer_v2/analysis/summary.json`
- `artifacts/local_state_cluster_transfer_v2/COMPLETE`
