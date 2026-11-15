# 2023–2024 冻结验证报告：局地历史状态 × 气象空间支持

更新时间：2026-09-07。2023 用于 architecture selection；2024 只在 2023 全矩阵完成并写入 `validation_freeze.json`（`test_seen=false`）后一次性评估。本文统一报告两个年份，不用 2024 反向修改模型、假设或判据。

## 1. 科学问题与严格对照

本实验回答两个必须分开的命题：

1. 在本场历史功率已经用于轨迹 baseline 的条件下，额外历史状态是否还需要与未来天气进行非可分离联合映射？
2. 如果需要，这种联合映射是否必须依赖风场周围 4×4 Pangu 场，而非风场坐标处的双线性插值天气？

固定条件为 56 个风场、2022 train、2023 validation、2024 frozen test、10 seeds（42–51）、3 h 历史、24 个 15 min 输出、相同共享 backbone、相同训练预算和 station-only loss。第一轮不使用 station embedding、adapter、GNN 或 cluster loss。主指标为 56 场等权 station macro CF MAE。

三种天气空间支持：

- `bilinear`：在原始 14 个变量上双线性插值到场站，再进行天气编码；
- `fixed4`：先编码场站周围 4×4 网格，再以固定 Gaussian 权重聚合；
- `learned4`：先编码周围 4×4 网格，再以包含风向几何的可学习 query 聚合。

三种历史利用方式：

- `None`：保留历史轨迹 baseline 和天气 correction，但不给 correction branch 额外历史状态；
- `Add`：在完整 response decoder 的输出级严格限制为

  \[
  g(W,0)+g(0,H)-g(0,0),
  \]

  允许天气主效应和历史主效应，但不允许非可分离交互；
- `Int`：忠实的 concat MLP：

  \[
  g(W,H).
  \]

在同一空间支持内，`None/Add/Int` 参数量完全相同。因此主机制比较是 `Int−Add`，而不是 `Int−None`。

周边天气是否特异增强交互采用预注册的 difference-in-differences：

\[
\mathrm{DID}_{4\times4}
=
(\mathrm{Int}-\mathrm{Add})_{4\times4}
-(\mathrm{Int}-\mathrm{Add})_{bilinear}.
\]

只有 DID 小于 0 且配对 95% CI 完全低于 0，才能声称周边 4×4 对交互是必要或特异有利的。

## 2. 10-seed 主结果

### 2.1 Station macro CF MAE（主指标）

| 天气支持 | 历史方式 | 2023 validation | 2024 frozen test | 参数量 |
|---|---|---:|---:|---:|
| Bilinear | None | 0.089622 ± 0.000177 | 0.086265 ± 0.000319 | 542,893 |
| Bilinear | Add | 0.089402 ± 0.000279 | 0.085978 ± 0.000627 | 542,893 |
| Bilinear | **Int** | **0.087629 ± 0.000202** | **0.084760 ± 0.000402** | 542,893 |
| Fixed 4×4 | None | 0.089540 ± 0.000201 | 0.086125 ± 0.000372 | 542,893 |
| Fixed 4×4 | Add | 0.089312 ± 0.000291 | 0.085976 ± 0.000422 | 542,893 |
| Fixed 4×4 | **Int** | **0.087361 ± 0.000197** | **0.084597 ± 0.000258** | 542,893 |
| Learned 4×4 | None | 0.089444 ± 0.000244 | 0.086191 ± 0.000338 | 551,790 |
| Learned 4×4 | Add | 0.089269 ± 0.000265 | 0.085822 ± 0.000317 | 551,790 |
| Learned 4×4 | **Int** | **0.087618 ± 0.000379** | **0.084714 ± 0.000537** | 551,790 |

2023 冻结选择和 2024 数值最优均为 `fixed4_int`：

\[
\boxed{0.087361\pm0.000197\quad(2023)}
\]

\[
\boxed{0.084597\pm0.000258\quad(2024)}.
\]

### 2.2 物理求和后的 cluster MAE（secondary metric）

| 模型 | 2023 (MW) | 2024 (MW) |
|---|---:|---:|
| `bilinear_none` | 288.577 ± 9.770 | 288.837 ± 12.070 |
| `bilinear_add` | 288.220 ± 8.248 | 287.041 ± 7.515 |
| **`bilinear_int`** | **278.692 ± 4.968** | **273.781 ± 4.958** |
| `fixed4_none` | 289.937 ± 9.921 | 288.865 ± 10.236 |
| `fixed4_add` | 286.558 ± 6.477 | 285.866 ± 6.181 |
| `fixed4_int` | 280.090 ± 7.656 | 276.502 ± 5.448 |
| `learned4_none` | 286.935 ± 7.390 | 289.126 ± 9.050 |
| `learned4_add` | 287.759 ± 8.240 | 286.170 ± 9.142 |
| `learned4_int` | 281.321 ± 7.147 | 278.546 ± 7.191 |

两个年份的 summed-cluster MAE 均由 `bilinear_int` 最优。不过本实验按 station-only supervision 和 station CF MAE 冻结，不能事后用 cluster MAE 改选模型，也不能与经过 cluster fine-tuning 的模型作等价训练预算比较。

## 3. 核心结果：Interaction 是否严格超过 Additive？

负值代表 Interaction 更好。

| 空间支持 | 2023 `Int−Add`（95% CI；wins） | 2024 `Int−Add`（95% CI；wins） | 2023/2024 改善场站数 |
|---|---:|---:|---:|
| Bilinear | **−0.001773 [−0.001929, −0.001617]; 10/10** | **−0.001218 [−0.001688, −0.000748]; 10/10** | 44/56；45/56 |
| Fixed 4×4 | **−0.001951 [−0.002198, −0.001703]; 10/10** | **−0.001379 [−0.001737, −0.001022]; 10/10** | 51/56；46/56 |
| Learned 4×4 | **−0.001651 [−0.001851, −0.001451]; 10/10** | **−0.001109 [−0.001488, −0.000729]; 10/10** | 48/56；45/56 |

这是本实验最强的跨年结论：

\[
\boxed{g(W,H)\;\text{稳定优于}\;g(W,0)+g(0,H)-g(0,0).}
\]

三个空间支持均在两个年份取得 10/10 seed wins，配对区间完全低于 0，并覆盖多数场站。2024 改善幅度较 2023 缩小，但方向和显著性完整复现。

相较之下，`Add−None` 的跨年稳定性较弱：Bilinear 和 Fixed 4×4 在 2024 的 CI 均跨 0，只有 Learned 4×4 仍显著。因此不能再把主要数据规律概括为“历史主效应 + 天气主效应近似可分离”；新的严格 matched control 支持的是非可分离联合映射具有额外预测价值。

## 4. 周边 4×4 是否特异地增强 Interaction？

### 4.1 空间表示的直接比较

| 比较 | 2023 Δ（95% CI；wins） | 2024 Δ（95% CI；wins） |
|---|---:|---:|
| `fixed4_int − bilinear_int` | **−0.000268 [−0.000461, −0.000075]; 9/10** | −0.000164 [−0.000393, +0.000065]; 7/10 |
| `learned4_int − bilinear_int` | −0.000012 [−0.000343, +0.000320]; 5/10 | −0.000047 [−0.000501, +0.000408]; 5/10 |

Fixed 4×4 在 2023 有很小的直接收益，但 2024 区间跨 0；Learned 4×4 两年均没有可靠收益。即使 Fixed 4×4 的绝对 station 均值两年最低，也不能仅凭直接比较证明它增强的是 history-weather interaction。

### 4.2 预注册 support-DID

| DID | 2023 validation | 2024 frozen test |
|---|---:|---:|
| `(Int−Add)_fixed4 − (Int−Add)_bilinear` | −0.000178 [−0.000404, +0.000048]; 6/10 | −0.000161 [−0.000500, +0.000177]; 6/10 |
| `(Int−Add)_learned4 − (Int−Add)_bilinear` | +0.000122 [−0.000216, +0.000460]; 4/10 | +0.000109 [−0.000425, +0.000644]; 6/10 |

两个年份、两种 4×4 表示的 DID 均跨 0。故以下强命题被否定：

> 历史状态之所以有效，是因为它改变了模型对周围 4×4 Pangu 空间结构的解释。

严格支持的是更窄命题：

> 历史功率和未来 Pangu forcing 存在稳定的预测性非可分离联合信息；这种信息在 Bilinear 点天气上已经能够识别，没有证据证明周边 4×4 是必要条件。

## 5. 状态干预：模型是否真的使用正确的历史状态？

表中为干预后 station CF MAE 的平均退化，格式为 `2023 / 2024`；所有单元均为 10/10 seeds 正向退化。

| 模型 | 强样本错配 | 远场站置换 | 统一参考状态 |
|---|---:|---:|---:|
| `bilinear_add` | +0.063663 / +0.063248 | +0.052177 / +0.057755 | +0.010714 / +0.010813 |
| `bilinear_int` | +0.039377 / +0.036714 | +0.031408 / +0.031315 | +0.013760 / +0.013177 |
| `fixed4_add` | +0.068489 / +0.068002 | +0.056630 / +0.062686 | +0.011139 / +0.011219 |
| `fixed4_int` | +0.037914 / +0.035279 | +0.030380 / +0.030196 | +0.013559 / +0.012861 |
| `learned4_add` | +0.063759 / +0.063134 | +0.052021 / +0.057647 | +0.010965 / +0.011166 |
| `learned4_int` | +0.038778 / +0.036106 | +0.030433 / +0.030339 | +0.013413 / +0.012832 |

两年结果几乎重合。这排除了“状态 branch 存在但被模型忽略”的解释：正确样本状态、正确 station-state 对应和状态变化都显著影响预测。

解释边界仍需保留：

1. 干预证明模型依赖历史状态，但不能单独把 latent 识别成风机状态、边界层状态、可用容量或限电状态；
2. `Add` 对极端错配比 `Int` 更敏感，而 `Int` 正常预测更准，因此干预退化大小不是交互强度或模型质量的单调指标；
3. “预测非可分离性”不等价于已经识别了因果物理调制机制。

## 6. Horizon 与 ramp 的跨年表现

`Int−Add` 的物理求和 cluster 指标变化如下，负值代表改善：

| 空间支持 | 年份 | 0–1 h MAE | 1–3 h MAE | 3–6 h MAE | Ramp MAE | Overall cluster MAE |
|---|---|---:|---:|---:|---:|---:|
| Bilinear | 2023 | −0.367 | −6.866 | −14.356 | −0.916 | −9.528 |
| Bilinear | 2024 | −2.055 | −10.898 | −18.570 | −1.011 | −13.260 |
| Fixed 4×4 | 2023 | −0.062 | −3.906 | −10.312 | −0.804 | −6.468 |
| Fixed 4×4 | 2024 | −1.171 | −7.256 | −13.502 | −0.880 | −9.364 |
| Learned 4×4 | 2023 | +3.524 | −2.884 | −12.127 | −0.472 | −6.438 |
| Learned 4×4 | 2024 | +2.730 | −4.646 | −13.062 | −0.465 | −7.625 |

交互收益稳定集中在 1–6 h，尤其 3–6 h，而非只发生在最短时距；Ramp MAE 也持续改善。这与“额外历史只复制当前值 persistence”的解释不一致。Learned 4×4 在 0–1 h 反而持续退化，是不保留复杂空间 query 的另一项证据。

## 7. 统一科学结论

### 得到跨年支持

1. 在本场历史 baseline 已存在的前提下，额外历史状态仍被真实使用；
2. `g(W,H)` 在三种空间支持、两个年份中都严格超过输出级 Additive 对照，证明存在预测上可辨识的 history-weather 非可分离信息；
3. 该增益覆盖多数场站，且主要作用于 1–6 h 和 ramp，而不是简单 persistence；
4. 不需要 station embedding，也不需要独立训练 56 个模型，就能在共享 station-level 模型中识别这一规律；
5. Learned 4×4 没有提供稳定收益，复杂 learnable spatial alignment 应停止。

### 未得到支持

1. 两年 support-DID 都失败，周边 4×4 不是交互成立的已证实必要条件；
2. 当前 latent 只可称为 local generation-history representation，不能命名为任何未观测的具体物理状态；
3. 当前证据支持预测性非可分离性，不足以声称历史状态因果地改变了真实风机的天气—功率传递定律；
4. 不能事后选择“更支持故事”的场站；场站异质性只能作为预注册的后续解释分析。

## 8. 路线判决

机器预注册判定为：

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
  "switch_primary_task_to_single_farm_state_modulation": false
}
```

这里的 `false` 只表示“4×4 周边天气特异的 state-modulation 主张没有通过”，不表示 local history-weather interaction 无效。二者必须区分。

据此建议：

1. **不切换到 56 个完全独立的单风场模型。** 保留“场站是物理计算单元、跨场共享参数、最后物理求和”的框架；
2. **不再研究复杂空间 alignment。** 若强调 2023 预注册 station 主指标，保留 `fixed4_int`；若后续强调简约性和 cluster extension，Bilinear Interaction 是合理、且机制结论等价的表示，但不能用 2024 事后改选；
3. **修正此前‘历史与天气近似可分离’的结论。** 新实验用严格输出级 Additive matched control 后，Interaction 的跨年优势明确成立；
4. **仍不要把模型结果强行写成已识别物理机理。** 当前论文可严谨声称的是“局地历史条件与未来天气具有稳定的预测非可分离性”，并将物理来源识别列为后续工作；
5. 若最终目标仍是 cluster MAE，下一步应以冻结的 local Interaction 模型做 local pretraining，再进行 matched aggregate-aware fine-tuning，并始终保留同训练预算的 Additive control。当前 station-only 实验本身不能替代这一集群训练比较。

因此，当前路线可以继续，但主故事应是：

\[
\boxed{
\text{local history-conditioned weather prediction}
\rightarrow
\text{cross-farm shared learning}
\rightarrow
\text{physical aggregation}
}
\]

而不是“周边 4×4 天气揭示了一个已识别的物理状态调制机制”。

## 9. 证据文件

- `artifacts/local_state_spatial_interaction_v1/selection/validation_freeze.json`
- `artifacts/local_state_spatial_interaction_v1/analysis/summary.json`
- `artifacts/local_state_spatial_interaction_v1/COMPLETE`
- `exp_docs/local_state_spatial_interaction_v1_results.md`

完整 seed 差值、逐场 MAE、2023/2024 干预和参数量均保存在机器汇总 `summary.json` 中。

<!-- CLUSTER_TRANSFER_V2_START -->
## Cluster-aware transfer 补充结论

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
<!-- CLUSTER_TRANSFER_V2_END -->
