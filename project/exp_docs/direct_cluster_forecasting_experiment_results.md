# Aggregate-only 集群功率预测实验结果

本实验只比较新 aggregate-only 任务内重新训练的模型；所有数值均为独立 checkpoint 的多随机种子统计，不包含预测平均、KD或旧任务结果。

## 核心回答

- 初始矩阵207个训练checkpoint和150个冻结测试任务已经完成；R2补充矩阵另完成20个训练、20个冻结测试和20个状态干预推理；严格状态—天气调制阶段又完成87个训练checkpoint及对应控制、干预和不变性审计。
- 当前按2023 validation冻结的最终单模型为 `A0-FT`：拓扑仍是R2-W，即原生小时天气先经过CNN编码和容量池化，再在任务latent空间插值到24个15 min目标时刻；历史只通过3 h TCN基线进入，不再重复输入correction。A0-FT从同seed R2-W嵌套初始化并以低学习率继续训练天气解码器。
- A0-FT validation为 `273.195 ± 2.435 MW`，比validation选出的最强fusion baseline PredictionBlend低 `20.022 MW`。
- A0-FT在2024内部时间留出上为 `270.325 ± 3.949 MW`，比冻结的PredictionBlend低 `19.019 MW`，比2024数值最好的fusion baseline TFT低 `18.402 MW`。
- R2-W比R1低 `1.242 MW`（validation），但2024高 `0.475 MW`。选型严格依据validation，不使用2024反向调整。
- StateShift证明旧R2-Full确实使用当前状态；新的严格拓扑又证明E1专家路由确实使用baseline latent，但E1仍不如无状态A0-FT。因此不把correction状态调制作为性能贡献。

## 冻结选择

- 历史基线范围：`hb_tcn_3h`（12个15 min点）。
- correction状态：不使用；历史状态只通过 `hb_tcn_3h` 基线进入。
- 最终天气路线：`r2_latent_interpolation`。
- 最终训练方案：同seed R2-W嵌套初始化后的 `a0_ft` 低学习率继续训练。
- 当前最强aggregate fusion baseline：`f_prediction_blend`。

## 3-seed 完整筛选

| family | method | validation MAE/MW |
|---|---|---:|
| history | hb_persistence | 468.058 ± 0.000 |
| history | hb_dlinear | 436.255 ± 0.007 |
| history | hb_tcn_1h | 416.405 ± 0.846 |
| history | hb_tcn_3h | 413.463 ± 1.494 |
| history | hb_tcn_6h | 413.142 ± 0.904 |
| history | hb_patchtst | 414.148 ± 1.103 |
| state | sc_current | 279.243 ± 2.616 |
| state | sc_1h | 280.492 ± 0.194 |
| state | sc_3h | 284.433 ± 4.736 |
| state | sc_6h | 281.408 ± 1.525 |
| state | sc_multi | 281.619 ± 1.687 |
| state | sc_horizon | 281.165 ± 1.073 |
| baselines | w_only | 373.591 ± 6.773 |
| weather | r1_raw_interpolation | 275.338 ± 1.433 |
| weather | r2_latent_interpolation | 277.190 ± 2.690 |
| weather | r3_full_path | 280.279 ± 3.558 |
| weather | r4_target_value | 281.902 ± 1.844 |
| weather | r5_dynamic_pool | 294.171 ± 0.612 |
| weather | r6_regional_tokens | 298.866 ± 5.849 |
| weather | r7_patch_regional | 293.309 ± 0.938 |
| weather | r8_joint_spacetime | 300.054 ± 4.579 |
| fusion | f_early_concat | 300.466 ± 3.533 |
| fusion | f_target_concat | 292.299 ± 3.798 |
| fusion | f_lead_gate | 294.532 ± 0.883 |
| fusion | f_prediction_blend | 291.476 ± 4.662 |
| fusion | f_cross_attention | 344.624 ± 6.683 |
| fusion | f_bidirectional | 301.402 ± 4.806 |
| fusion | f_tft | 291.315 ± 4.229 |
| fusion | f_astafn | 293.037 ± 1.682 |
| fusion | f_timexer | 295.534 ± 3.715 |
| mechanism | m0 | 412.739 ± 0.344 |
| mechanism | ms | 412.541 ± 0.399 |
| mechanism | mw | 276.595 ± 5.763 |
| mechanism | madd | 284.527 ± 3.486 |

## 10-seed 正式结果

### 2023 validation

| family | method | MAE/MW | RMSE/MW | R² | Ramp MAE/MW |
|---|---|---:|---:|---:|---:|
| history | hb_tcn_6h | 412.839 ± 0.498 | 610.993 ± 1.315 | 0.7634 ± 0.0010 | 61.152 ± 0.040 |
| history | hb_tcn_3h | 412.962 ± 0.834 | 611.160 ± 1.751 | 0.7632 ± 0.0014 | 61.131 ± 0.051 |
| state | sc_1h | 281.585 ± 2.710 | 402.384 ± 4.662 | 0.8974 ± 0.0024 | 55.663 ± 0.301 |
| state | sc_current | 280.908 ± 2.802 | 401.137 ± 3.699 | 0.8980 ± 0.0019 | 55.702 ± 0.225 |
| baselines | w_only | 372.493 ± 5.427 | 502.657 ± 6.473 | 0.8398 ± 0.0041 | 71.148 ± 0.416 |
| fusion | f_tft | 294.138 ± 5.521 | 413.953 ± 8.192 | 0.8913 ± 0.0043 | 58.230 ± 0.803 |
| fusion | f_prediction_blend | 293.217 ± 5.215 | 415.253 ± 8.498 | 0.8907 ± 0.0045 | 57.718 ± 0.738 |
| fusion | f_target_concat | 294.696 ± 4.209 | 413.464 ± 4.679 | 0.8916 ± 0.0025 | 58.166 ± 1.254 |
| weather | r3_full_path | 280.738 ± 3.621 | 400.359 ± 5.198 | 0.8984 ± 0.0026 | 55.966 ± 0.457 |
| weather | r1_raw_interpolation | 276.605 ± 2.396 | 394.736 ± 3.961 | 0.9012 ± 0.0020 | 58.413 ± 0.485 |
| weather | r2_full_state_weather | 276.114 ± 3.276 | 395.107 ± 6.170 | 0.9010 ± 0.0031 | 58.292 ± 0.274 |
| weather | r2_w_base | 275.363 ± 2.599 | 395.438 ± 3.816 | 0.9010 ± 0.0019 | 60.479 ± 0.642 |
| **weather** | **a0_ft_final** | **273.195 ± 2.435** | **391.839 ± 3.625** | **0.903 ± 0.002** | 60.102 ± 0.385 |
| mechanism | m0 | 413.277 ± 1.063 | 612.276 ± 3.210 | 0.7624 ± 0.0025 | 61.263 ± 0.143 |
| mechanism | ms | 413.403 ± 1.119 | 612.845 ± 3.290 | 0.7619 ± 0.0026 | 61.258 ± 0.136 |
| mechanism | mw | 276.636 ± 4.053 | 396.111 ± 6.732 | 0.9005 ± 0.0034 | 57.076 ± 0.904 |
| mechanism | madd | 282.271 ± 3.781 | 402.736 ± 5.448 | 0.8972 ± 0.0028 | 55.913 ± 0.426 |

### 2024 内部时间留出

| family | method | MAE/MW | RMSE/MW | R² | Ramp MAE/MW |
|---|---|---:|---:|---:|---:|
| history | hb_tcn_6h | 395.669 ± 0.593 | 590.377 ± 1.808 | 0.7596 ± 0.0015 | 57.988 ± 0.035 |
| history | hb_tcn_3h | 395.983 ± 0.929 | 590.380 ± 1.404 | 0.7596 ± 0.0011 | 57.979 ± 0.061 |
| state | sc_1h | 274.865 ± 3.303 | 401.571 ± 7.481 | 0.8887 ± 0.0042 | 53.130 ± 0.285 |
| state | sc_current | 274.437 ± 3.421 | 401.328 ± 5.596 | 0.8889 ± 0.0031 | 53.165 ± 0.210 |
| baselines | w_only | 369.500 ± 5.198 | 513.521 ± 8.683 | 0.8181 ± 0.0062 | 68.545 ± 0.307 |
| fusion | f_tft | 288.727 ± 6.148 | 414.703 ± 8.294 | 0.8813 ± 0.0048 | 55.698 ± 0.829 |
| fusion | f_prediction_blend | 289.344 ± 5.232 | 420.369 ± 10.683 | 0.8781 ± 0.0063 | 55.269 ± 0.626 |
| fusion | f_target_concat | 291.903 ± 2.161 | 420.787 ± 5.436 | 0.8779 ± 0.0032 | 55.713 ± 1.207 |
| weather | r3_full_path | 276.284 ± 3.703 | 403.475 ± 5.792 | 0.8877 ± 0.0032 | 53.446 ± 0.356 |
| weather | r1_raw_interpolation | 272.242 ± 4.466 | 396.265 ± 6.130 | 0.8917 ± 0.0033 | 56.088 ± 0.488 |
| weather | r2_full_state_weather | 270.330 ± 3.411 | 394.281 ± 6.578 | 0.8928 ± 0.0036 | 55.975 ± 0.335 |
| weather | r2_w_base | 272.717 ± 4.578 | 397.189 ± 7.017 | 0.8912 ± 0.0039 | 58.194 ± 0.736 |
| **weather** | **a0_ft_final** | **270.325 ± 3.949** | **393.077 ± 6.612** | **0.893 ± 0.004** | 57.818 ± 0.465 |
| mechanism | m0 | 396.354 ± 1.084 | 591.758 ± 2.874 | 0.7585 ± 0.0023 | 58.092 ± 0.143 |
| mechanism | ms | 396.614 ± 1.096 | 592.450 ± 2.899 | 0.7579 ± 0.0024 | 58.099 ± 0.141 |
| mechanism | mw | 273.117 ± 5.468 | 398.868 ± 10.153 | 0.8902 ± 0.0056 | 54.539 ± 0.864 |
| mechanism | madd | 276.468 ± 3.968 | 403.799 ± 6.181 | 0.8875 ± 0.0034 | 53.402 ± 0.421 |

最终A0-FT相对冻结的最强fusion baseline PredictionBlend：validation ΔMAE=`-20.022 MW`；2024内部时间留出 ΔMAE=`-19.019 MW`。负值表示最终路线更好。本轮按用户要求关注多seed平均值，不以显著性作为选型门槛。

## 状态—天气机制

- validation MFull−M0：-132.539 MW。
- validation MFull−MS：-132.666 MW。
- validation MFull−MW：+4.102 MW。
- validation MFull−MAdd：-1.533 MW。

MFull在R3路线下反而比MW高 `4.102 MW`。新增的R2专门消融进一步表明：R2-W在validation比R2-Full低 `0.751 MW`，但2024高 `2.387 MW`；StateShift在两个split均使Full恶化约 `2.61 MW`。严格调制阶段中，最佳严格候选E1也比同预算无状态控制A0-FT高 `1.439 MW`。因此状态会被模型使用，却没有形成净增益；论文贡献集中在“3 h历史基线 + 天气容量支持聚合 + latent跨分辨率适配”，最终checkpoint采用A0-FT。

## 单模型计算代价

| method | Trainable params | Profiled GFLOPs/origin | Latency ms/origin | Peak MiB |
|---|---:|---:|---:|---:|
| hb_tcn_3h | 51801 | 0.003 | 3.189 | 9.5 |
| w_only | 271575 | 0.244 | 9.054 | 16.5 |
| f_prediction_blend | 393560 | 0.251 | 12.455 | 17.0 |
| r2_full_state_weather / parameter-matched r2_w | 412335 | 0.253 | 19.139 | 17.7 |

## 分层稳健性（原R2-Full审计）

- 相对最强 fusion，validation 最大优势出现在 `history_variability/low`：ΔMAE=-23.604 MW。
- 最弱分层为 `future_wind_variation/high`：ΔMAE=-12.860 MW。

## 产物索引

原始矩阵位于 `artifacts/aggregate_cluster_v1/analysis/`；R2状态补充矩阵位于 `artifacts/aggregate_cluster_r2_state_v1/analysis/`；最终选择、严格调制控制和机制诊断位于 `artifacts/aggregate_strict_modulation_v1/`。

<!-- R2_STATE_MECHANISM_START -->
## R2 correction状态专门消融（10 seeds）

- 本阶段按2023 validation平均MAE选择 **R2-W**；后续严格调制阶段最终选择更新为同拓扑的A0-FT。
- 相对PredictionBlend：validation `-17.853 MW`，2024 `-16.627 MW`。
- R2-Full − R2-W：validation `+0.751 MW`，2024 `-2.387 MW`。
- R2-Full − R2-Add：validation `-4.536 MW`。
- StateShift退化：validation `+2.611 MW`，2024 `+2.614 MW`。
- 最终处理：删除修正分支的重复状态输入，采用更简洁的 B(H)+C(W) 结构。
- Full/W在validation和2024方向反转，因此不把correction状态调制作为稳健核心贡献。
- 历史基线固定为过去3 h TCN，不再扩展历史窗口或多尺度状态搜索。
- R2-W运行时跳过未激活的state encoder，但保留参数键以兼容正式checkpoint。

完整记录见 `exp_docs/aggregate_cluster_r2_state_mechanism_results.md`。
<!-- R2_STATE_MECHANISM_END -->

<!-- STRICT_STATE_WEATHER_START -->
## 严格状态—天气调制（10 seeds）

- validation冻结结果：**A0-FT（R2-W低学习率继续训练）**。
- 最佳严格候选 E1 相对R2-W：validation `-0.730 MW`，2024 `-0.252 MW`。
- E1 相对A0-FT：validation `+1.439 MW`，2024 `+2.140 MW`；严格调制没有形成净增益。
- A0-FT相对旧R2-W：validation `-2.169 MW`，2024 `-2.392 MW`。
- E1相对旧R2-W的Ramp MAE差值：validation `-0.546 MW`，2024 `-0.477 MW`；这不是相对最终A0-FT的收益。
- 全局置乱状态的MAE变化：validation `+9.005 MW`，2024 `+10.468 MW`。
- 零天气latent不变性最大误差 `0.000e+00`；严格排除了state-only correction捷径。
- 完整记录：`exp_docs/aggregate_strict_state_weather_modulation_results.md`。
<!-- STRICT_STATE_WEATHER_END -->
