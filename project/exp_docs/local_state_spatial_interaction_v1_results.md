# 单场空间支持 × 历史利用方式决定性实验

## 实验目的

本实验补上此前缺失的严格交叉矩阵。所有模型使用同一批 56 场、同一 3 h history baseline、同一共享 backbone、相同 station-only loss 和相同训练预算，只改变：

1. 气象空间支持：raw bilinear、fixed 4×4 Gaussian、learned 4×4 + wind geometry；
2. response branch：no-extra-history、additive history、concat interaction。

主指标为 56 场等权 station macro CF MAE。2023 完成全矩阵和冻结后，才读取 2024。主机制比较是每种空间支持内部的 `Int−Add`，而不是 `Int−None`。周边天气必要性由以下差分中的差分判断：

\[
igl(\mathrm{Int}-\mathrm{Add}igr)_{4	imes4}
-
igl(\mathrm{Int}-\mathrm{Add}igr)_{bilinear}.
\]

负值且配对区间不跨 0，才说明 4×4 特异地增强 interaction。

## 10-seed 主结果

| 模型 | 2023 station CF MAE | 2024 station CF MAE | 2023 summed cluster MAE | 2024 summed cluster MAE |
|---|---:|---:|---:|---:|
| `bilinear_none` | 0.089622 ± 0.000177 | 0.086265 ± 0.000319 | 288.577 | 288.837 |
| `bilinear_add` | 0.089402 ± 0.000279 | 0.085978 ± 0.000627 | 288.220 | 287.041 |
| `bilinear_int` | 0.087629 ± 0.000202 | 0.084760 ± 0.000402 | 278.692 | 273.781 |
| `fixed4_none` | 0.089540 ± 0.000201 | 0.086125 ± 0.000372 | 289.937 | 288.865 |
| `fixed4_add` | 0.089312 ± 0.000291 | 0.085976 ± 0.000422 | 286.558 | 285.866 |
| `fixed4_int` | 0.087361 ± 0.000197 | 0.084597 ± 0.000258 | 280.090 | 276.502 |
| `learned4_none` | 0.089444 ± 0.000244 | 0.086191 ± 0.000338 | 286.935 | 289.126 |
| `learned4_add` | 0.089269 ± 0.000265 | 0.085822 ± 0.000317 | 287.759 | 286.170 |
| `learned4_int` | 0.087618 ± 0.000379 | 0.084714 ± 0.000537 | 281.321 | 278.546 |

## 历史机制配对比较

表中每个单元依次为 `mean Δ | paired 95% CI | wins`；负值代表前者更好。

| 比较 | 2023 validation | 2024 frozen test |
|---|---:|---:|
| `bilinear_add_minus_bilinear_none` | -0.000220; [-0.000368, -0.000073]; 9/10 | -0.000287; [-0.000841, +0.000268]; 7/10 |
| `bilinear_int_minus_bilinear_none` | -0.001993; [-0.002138, -0.001849]; 10/10 | -0.001505; [-0.001909, -0.001100]; 10/10 |
| **`bilinear_int_minus_bilinear_add`** | **-0.001773; [-0.001929, -0.001617]; 10/10** | **-0.001218; [-0.001688, -0.000748]; 10/10** |
| `fixed4_add_minus_fixed4_none` | -0.000228; [-0.000371, -0.000085]; 8/10 | -0.000149; [-0.000419, +0.000121]; 7/10 |
| `fixed4_int_minus_fixed4_none` | -0.002179; [-0.002356, -0.002001]; 10/10 | -0.001529; [-0.001940, -0.001117]; 10/10 |
| **`fixed4_int_minus_fixed4_add`** | **-0.001951; [-0.002198, -0.001703]; 10/10** | **-0.001379; [-0.001737, -0.001022]; 10/10** |
| `learned4_add_minus_learned4_none` | -0.000176; [-0.000350, -0.000002]; 7/10 | -0.000368; [-0.000558, -0.000178]; 10/10 |
| `learned4_int_minus_learned4_none` | -0.001827; [-0.002070, -0.001584]; 10/10 | -0.001477; [-0.001918, -0.001036]; 10/10 |
| **`learned4_int_minus_learned4_add`** | **-0.001651; [-0.001851, -0.001451]; 10/10** | **-0.001109; [-0.001488, -0.000729]; 10/10** |

## 周边天气是否特异地增强 interaction

| support DID | 2023 validation | 2024 frozen test |
|---|---:|---:|
| `fixed4_interaction_did` | -0.000178; [-0.000404, +0.000048]; 6/10 | -0.000161; [-0.000500, +0.000177]; 6/10 |
| `learned4_interaction_did` | +0.000122; [-0.000216, +0.000460]; 4/10 | +0.000109; [-0.000425, +0.000644]; 6/10 |

## 状态干预

下表为 2023 station CF MAE degradation，括号内为正退化 seeds。Add 与 Int 都执行相同干预，避免把“使用历史”误写成“识别 interaction”。

| 模型 | strong donor | far-station permutation | reference state |
|---|---:|---:|---:|
| `bilinear_add` | +0.063663 (10/10) | +0.052177 (10/10) | +0.010714 (10/10) |
| `bilinear_int` | +0.039377 (10/10) | +0.031408 (10/10) | +0.013760 (10/10) |
| `fixed4_add` | +0.068489 (10/10) | +0.056630 (10/10) | +0.011139 (10/10) |
| `fixed4_int` | +0.037914 (10/10) | +0.030380 (10/10) | +0.013559 (10/10) |
| `learned4_add` | +0.063759 (10/10) | +0.052021 (10/10) | +0.010965 (10/10) |
| `learned4_int` | +0.038778 (10/10) | +0.030433 (10/10) | +0.013413 (10/10) |

## 决策

没有周边 4×4 表示同时通过 interaction 与 support-DID 判据；停止把 state modulation 作为主线，保留可分离局地动力学的集群路线。

机器判定：

```json
{
  "interaction_supported_by_spatial_mode": {
    "bilinear": true,
    "fixed4": true,
    "learned4": true
  },
  "surrounding_weather_specific_interaction_supported": {
    "fixed4": false,
    "learned4": false
  },
  "switch_primary_task_to_single_farm_state_modulation": false,
  "rule": "Switch only if a 4x4 Int-Add contrast is significant on frozen validation, confirms direction on 2024, and its interaction advantage is stronger than bilinear under the paired support DID."
}
```

场站不能按结果事后筛选。逐场结果、完整配对 seed 差值、干预及参数量保存在机器汇总中。

## 证据

- `artifacts/local_state_spatial_interaction_v1/selection/validation_freeze.json`
- `artifacts/local_state_spatial_interaction_v1/analysis/summary.json`
- `artifacts/local_state_spatial_interaction_v1/COMPLETE`
