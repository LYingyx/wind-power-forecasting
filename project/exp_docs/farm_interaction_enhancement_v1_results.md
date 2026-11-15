# Farm-level Interaction Enhancement v1 完整结果

2023 冻结主模型：formal/c2_metadata_film/int。固定 56 场，2022 训练，2024 冻结测试。
本轮采用 CF residual 输出，M0 同协议重训；最终 CF 不裁剪，以保留严格可分离性。旧 logit baseline 单列为历史参考。
每组 IAdd/Int 等参数、同 seed/schedule。ID-FiLM 仅作诊断，metadata 仅经纬度及容量。

## val: 全部目标模型

| Group | Seeds | Int MAE CF | IAdd MAE CF | Int-IAdd 95% seed CI | Farms improved |
|---|---:|---:|---:|---|---:|
| m0 | 10 | 0.087137 | 0.090907 | -0.003770 [-0.003975, -0.003569] | 56/56 |
| m0_bounded | 10 | 0.087196 | 0.090523 | -0.003327 [-0.003499, -0.003143] | 48/56 |
| c2_metadata_film | 10 | 0.085990 | 0.089394 | -0.003404 [-0.003751, -0.003049] | 55/56 |
| c3_id_film | 10 | 0.085793 | 0.089288 | -0.003495 [-0.003737, -0.003241] | 55/56 |
| d0_capacity_aux | 10 | 0.087277 | 0.090897 | -0.003620 [-0.003756, -0.003480] | 55/56 |
| d1_level_shape | 10 | 0.086991 | 0.090785 | -0.003794 [-0.003994, -0.003610] | 56/56 |
| bounded_finalist | 10 | 0.086179 | 0.089038 | -0.002859 [-0.003059, -0.002663] | 52/56 |
| a1_lead_summary | 5 | 0.087221 | 0.090973 | -0.003751 [-0.004065, -0.003413] | 54/56 |
| a2_cross_attention | 5 | 0.087087 | 0.090964 | -0.003878 [-0.004156, -0.003646] | 52/56 |
| b1_origin_delta_linear | 5 | 0.087336 | 0.091072 | -0.003736 [-0.003916, -0.003580] | 53/56 |
| b2_origin_delta_tcn | 5 | 0.087480 | 0.091717 | -0.004237 [-0.004551, -0.003929] | 55/56 |
| c1_metadata_residual | 5 | 0.086254 | 0.090135 | -0.003881 [-0.004268, -0.003512] | 55/56 |
| c2_metadata_film_head_capacity_aux | 5 | 0.086009 | 0.089368 | -0.003359 [-0.003657, -0.003063] | 55/56 |
| c2_metadata_film_head_level_shape | 5 | 0.085729 | 0.089429 | -0.003699 [-0.003925, -0.003531] | 56/56 |

### 全部历史 baseline

| Model | MAE CF | RMSE CF | Physical-sum MAE MW |
|---|---:|---:|---:|
| H-Matched | 0.111785 | 0.170022 | 453.867 |
| H-Strong | 0.111707 | 0.169898 | 453.492 |
| W | 0.090045 | 0.140713 | 283.617 |
| HW-None | 0.089622 | 0.140328 | 288.577 |
| HW-SAdd | 0.089402 | 0.139361 | 288.220 |
| HW-IAdd | 0.089537 | 0.139829 | 289.874 |
| HW-Int | 0.087629 | 0.137238 | 278.692 |
| P | 0.117542 | 0.182200 | 468.058 |

## test: 全部目标模型

| Group | Seeds | Int MAE CF | IAdd MAE CF | Int-IAdd 95% seed CI | Farms improved |
|---|---:|---:|---:|---|---:|
| m0 | 10 | 0.084394 | 0.087206 | -0.002812 [-0.003081, -0.002545] | 53/56 |
| m0_bounded | 10 | 0.084303 | 0.087265 | -0.002963 [-0.003237, -0.002653] | 48/56 |
| c2_metadata_film | 10 | 0.083808 | 0.086618 | -0.002810 [-0.003148, -0.002473] | 55/56 |
| c3_id_film | 10 | 0.084396 | 0.087440 | -0.003044 [-0.003364, -0.002742] | 54/56 |
| d0_capacity_aux | 10 | 0.084229 | 0.087285 | -0.003057 [-0.003340, -0.002774] | 54/56 |
| d1_level_shape | 10 | 0.083977 | 0.086998 | -0.003021 [-0.003247, -0.002814] | 55/56 |
| bounded_finalist | 10 | 0.084195 | 0.086488 | -0.002293 [-0.002754, -0.001888] | 50/56 |
| a1_lead_summary | 5 | 0.084284 | 0.087227 | -0.002943 [-0.003510, -0.002376] | 52/56 |
| a2_cross_attention | 5 | 0.084242 | 0.087375 | -0.003133 [-0.003739, -0.002689] | 50/56 |
| b1_origin_delta_linear | 5 | 0.084483 | 0.087160 | -0.002677 [-0.002872, -0.002445] | 51/56 |
| b2_origin_delta_tcn | 5 | 0.084823 | 0.087733 | -0.002910 [-0.003318, -0.002502] | 55/56 |
| c1_metadata_residual | 5 | 0.084119 | 0.087119 | -0.003000 [-0.003454, -0.002653] | 53/56 |
| c2_metadata_film_head_capacity_aux | 5 | 0.083732 | 0.086666 | -0.002934 [-0.003169, -0.002676] | 55/56 |
| c2_metadata_film_head_level_shape | 5 | 0.083683 | 0.086759 | -0.003076 [-0.003575, -0.002380] | 56/56 |

### 全部历史 baseline

| Model | MAE CF | RMSE CF | Physical-sum MAE MW |
|---|---:|---:|---:|
| H-Matched | 0.106924 | 0.163759 | 437.459 |
| H-Strong | 0.106843 | 0.163624 | 436.664 |
| W | 0.086943 | 0.136320 | 282.254 |
| HW-None | 0.086265 | 0.135651 | 288.837 |
| HW-SAdd | 0.085978 | 0.134626 | 287.041 |
| HW-IAdd | 0.086089 | 0.135034 | 288.998 |
| HW-Int | 0.084760 | 0.133092 | 273.781 |
| P | 0.112763 | 0.175619 | 452.319 |

## 指标口径与证据

seed_metrics.csv 与 complete_baselines.csv 完整报告 station CF 与 physical-sum cluster MW 的 MAE、15 min/1 h/3 h RampMAE、方向准确率和极端 ramp precision/recall/F1。极端阈值为 freeze 中的 2022 q90。
formal_comparisons.csv 同时含 screening 与 formal 标记、各指标 paired seed bootstrap CI、station MAE 的共享 UTC-day block CI、per-farm coverage、parent 对照与 interaction DID。
站场 ramp 绝对值的平均不等于容量加权后集群 ramp 的绝对值。不能合并声称 ramp 改善。Attention 是诊断而非因果证据；见 history_attention.csv。
cf_residual 的越界比例另报 out_of_range_cf_fraction；不事后裁剪或用 2024 改选模型。
strengthening_interventions.csv 报告冻结后的 history-reference / stratified weather permutation 干预；它们只说明输入依赖，不识别因果物理状态。
component_ablations.csv 报告删去各组件的配对差；output_link_robustness.csv 在 CF residual 和 bounded CF mixture 两套严格可分离协议内分别复核 Int-IAdd。
D1 level-shape 的归因对照是等参数 D0 capacity-aux，详见 formal_comparisons.csv 的 Int-ParentInt 和 Interaction-DID。
所有 mean-error/shape MSE、预测内部及历史边界 ramp 分别报告；不能用平均水平校正替代局地动态恢复证据。
