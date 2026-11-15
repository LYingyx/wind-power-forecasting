# Local state 多阶段迁移实验结果

## 实验回答的问题

本实验不预设多阶段训练优于端到端训练。它以场站为预测单元、跨场共享编码器与响应网络、容量加权物理求和为集群输出，严格比较：

- `E0/E1`：从头端到端联合训练，无/有 local state interaction；
- `P0/P1-Free`：station-only 预训练后联合微调，无/有 local state interaction；
- `P1-Preserve`：联合微调时冻结已学得的 state encoder 与 interaction；
- `Additive`：同参数化的历史主效应控制，用来区分 state 本身预测与 weather×state interaction。

Stage 1 只按等权场站 capacity-factor MAE 选 checkpoint；Stage 2 才按 cluster MAE 选。2024 test 在 `selection/validation_freeze.json` 写入后才读取。

> 适用边界：本轮使用固定双线性局地 Pangu forcing。它检验训练日程与局地状态是否被 aggregate optimization 舍弃，但不等价于旧单场实验的 4×4 周边气象代理。

## 10-seed 主结果

| 模型 | 2023 cluster MAE (MW) | 2024 cluster MAE (MW) | 2023 station macro CF MAE |
|---|---:|---:|---:|
| e0 | 270.662 ± 1.586 | 269.864 ± 3.111 | 0.09438 |
| e_int | 268.977 ± 1.315 | 264.976 ± 3.593 | 0.09349 |
| e_add | 268.877 ± 1.782 | 266.747 ± 4.112 | 0.09329 |
| p0_free | 268.927 ± 2.749 | 266.367 ± 2.744 | 0.09197 |
| p_int_free | 266.177 ± 3.793 | 262.279 ± 2.790 | 0.09058 |
| p_int_preserve | 266.742 ± 4.011 | 263.863 ± 2.706 | 0.09069 |
| p_add_free | 264.973 ± 3.265 | 259.857 ± 2.030 | 0.09008 |

配对差值均为 candidate − control，负值代表候选更好：

| 比较 | mean Δ | paired 95% CI | wins |
|---|---:|---:|---:|
| stage1_state | -0.001428 | [-0.001603, -0.001253] | 10/10 |
| e2e_state | -1.684210 | [-2.842526, -0.525894] | 8/10 |
| pretrained_free_state | -2.749939 | [-5.593173, 0.093295] | 7/10 |
| pretrained_preserve_state | -2.185333 | [-5.150878, 0.780211] | 7/10 |
| preserve_vs_free | 0.564605 | [-0.235507, 1.364718] | 3/10 |
| e2e_interaction_vs_additive | 0.100899 | [-1.260299, 1.462097] | 5/10 |
| pretrained_interaction_vs_additive | 1.203830 | [-0.963393, 3.371053] | 5/10 |
| stage1_interaction_vs_additive | 0.000078 | [-0.000075, 0.000232] | 3/10 |

## 状态干预

正 degradation 表示替换状态后性能恶化，即模型确实依赖正确状态：

| 阶段 | 干预 | mean degradation | positive seeds |
|---|---|---:|---:|
| stage1 | strong_sample | 0.031036 | 10/10 |
| stage1 | station_permutation | 0.023183 | 10/10 |
| stage1 | reference | 0.009508 | 10/10 |
| e2e | strong_sample | 235.524068 | 10/10 |
| e2e | station_permutation | 126.635150 | 10/10 |
| e2e | reference | 48.989071 | 10/10 |
| pretrained_free | strong_sample | 264.907875 | 10/10 |
| pretrained_free | station_permutation | 158.703965 | 10/10 |
| pretrained_free | reference | 53.177975 | 10/10 |
| pretrained_preserve | strong_sample | 260.498580 | 10/10 |
| pretrained_preserve | station_permutation | 156.781250 | 10/10 |
| pretrained_preserve | reference | 51.688518 | 10/10 |

## 严格判定

- 局地状态在 station-only 阶段被识别：`True`
- 状态收益在集群目标下保留：`True`
- 多阶段相对端到端对“状态增量”的差分中的差分：`-1.065729`（负值表示多阶段更有利于保留状态）
- 冻结式与自由微调差异：`0.564605 MW`

结论：继续 local-physical-unit → shared learning → physical sum 的集群路线；是否采用多阶段训练由 P1-vs-E1 和差分中的差分决定。

不能按“挑选支持结论的场站”形成主结果。逐场结果只用于异质性分析或预注册分层；完整 56 场 macro 与 cluster sum 始终是主证据。逐场差值见机器可读文件。

## 证据文件

- `artifacts/local_state_transfer_multistage_v1/analysis/summary.json`
- `artifacts/local_state_transfer_multistage_v1/selection/screening.json`
- `artifacts/local_state_transfer_multistage_v1/selection/validation_freeze.json`
