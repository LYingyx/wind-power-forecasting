# 气象—功率响应状态辨识实验结果

## 结论

- 全部结构由2023 validation冻结；最终按10个独立checkpoint平均MAE选择 **C3 仅历史天气 token**。
- 最佳响应状态候选是 **C3 仅历史天气 token**。相对无响应状态A0-FT，validation差值为 `-0.027 MW`，2024内部时间留出差值为 `-0.619 MW`。
- 最终模型validation / 2024平均MAE分别为 `273.168 / 269.706 MW`；不使用ensemble，也不挑最好seed。
- 零未来天气不变量在10个seed上全部通过，最大状态相关差异为 `0.000e+00`（归一化功率）。
- 本实验回答的是response state的信息来源是否足够，而不是继续搜索调制算子；历史baseline仍固定为已经验证的3 h TCN。

## 3-seed筛选

### Adapter-only

| 模型 | 3-seed validation MAE / MW | 相对 A0-FT / MW |
|---|---:|---:|
| C2 仅历史功率 token | 274.879 ± 2.309 | +0.440 |
| C3 仅历史天气 token | 274.097 ± 2.364 | -0.342 |
| R1 天气—功率 level pair | 274.246 ± 2.279 | -0.193 |
| R2 天气—功率 transition pair | 274.047 ± 2.356 | -0.392 |
| R3 transition pair + identification loss | 274.300 ± 2.193 | -0.139 |

### Joint fine-tuning

| 模型 | 3-seed validation MAE / MW | 相对 A0-FT / MW |
|---|---:|---:|
| C2 仅历史功率 token | 274.552 ± 2.437 | +0.113 |
| C3 仅历史天气 token | 273.640 ± 1.517 | -0.799 |
| R1 天气—功率 level pair | 274.107 ± 1.658 | -0.332 |
| R2 天气—功率 transition pair | 273.689 ± 1.555 | -0.750 |
| R3 transition pair + identification loss | 273.785 ± 1.537 | -0.653 |

按预注册规则补齐10 seeds：`r2, c3`。

## 10-seed正式结果：2023 validation

| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1 h | 1–3 h | 3–6 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 A0-FT（无响应状态） | 273.195 ± 2.435 | 391.839 ± 3.625 | 0.903 ± 0.002 | 60.102 ± 0.385 | 103.246 ± 0.894 | 253.228 ± 1.395 | 343.156 ± 3.987 |
| C1 功率历史 latent | 274.693 ± 2.801 | 393.743 ± 4.277 | 0.902 ± 0.002 | 60.053 ± 0.423 | 103.213 ± 0.860 | 253.385 ± 2.015 | 346.058 ± 4.365 |
| R2 天气—功率 transition pair | 273.173 ± 2.425 | 391.893 ± 4.245 | 0.903 ± 0.002 | 59.910 ± 0.428 | 103.201 ± 0.941 | 252.864 ± 1.754 | 343.369 ± 3.612 |
| C3 仅历史天气 token | 273.168 ± 2.444 | 391.804 ± 4.222 | 0.903 ± 0.002 | 59.912 ± 0.441 | 103.163 ± 0.924 | 252.892 ± 1.759 | 343.354 ± 3.638 |

## 冻结后的2024内部时间留出

| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1 h | 1–3 h | 3–6 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 A0-FT（无响应状态） | 270.325 ± 3.949 | 393.077 ± 6.612 | 0.893 ± 0.004 | 57.818 ± 0.465 | 97.372 ± 0.799 | 244.996 ± 2.437 | 344.862 ± 6.178 |
| C1 功率历史 latent | 272.176 ± 4.443 | 395.874 ± 7.731 | 0.892 ± 0.004 | 57.816 ± 0.515 | 97.400 ± 0.736 | 245.898 ± 3.111 | 347.953 ± 6.943 |
| R2 天气—功率 transition pair | 269.714 ± 3.701 | 391.849 ± 6.649 | 0.894 ± 0.004 | 57.576 ± 0.480 | 97.284 ± 0.772 | 244.301 ± 2.336 | 344.132 ± 5.760 |
| C3 仅历史天气 token | 269.706 ± 3.693 | 391.774 ± 6.733 | 0.894 ± 0.004 | 57.579 ± 0.488 | 97.248 ± 0.753 | 244.332 ± 2.337 | 344.108 ± 5.740 |

## 状态干预

以下差值均为干预后MAE减原模型MAE，正值表示破坏配对后变差。

| 干预 | validation ΔMAE / MW | 2024 ΔMAE / MW |
|---|---:|---:|
| power_shuffle | +0.000 ± 0.000 | +0.000 ± 0.000 |
| weather_shuffle | +4.321 ± 2.743 | +5.540 ± 2.180 |
| pair_shift | +0.000 ± 0.000 | +0.000 ± 0.000 |
| response_mean | +3.683 ± 3.219 | +4.344 ± 3.502 |
| support_only | +0.487 ± 0.710 | +0.108 ± 0.400 |

## Regime诊断

当前功率五分位×未来10 m风速变化×预测时段共45格，其中 `30/45` 个可解释单元跨年份同号。负值表示响应状态优于A0-FT。

可视化：

- `artifacts/aggregate_response_state_v1/figures/response_regime_val.png`
- `artifacts/aggregate_response_state_v1/figures/response_regime_test.png`

## 产物

- 原始训练与预测：`artifacts/aggregate_response_state_v1/{screen_adapter,screen_joint,formal_adapter,formal_joint}/`
- 干预与架构诊断：`artifacts/aggregate_response_state_v1/{interventions,diagnostics}/`
- 机器可读汇总：`artifacts/aggregate_response_state_v1/analysis/summary.json`
