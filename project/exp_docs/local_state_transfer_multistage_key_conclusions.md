# 局地历史迁移与多阶段集群预测：关键结论

更新日期：2026-09-06

## 1. 最终结论

本轮实验已经全部完成，包括 3-seed screening、10-seed 正式验证、验证集冻结选择、冻结后的 2024 测试，以及预先安排的 interaction 模型状态干预。验证集冻结文件选择了：

```text
formal_transfer/p_add_free
```

冻结时 `test_seen=false`，因此模型选择没有使用 2024 结果。

最重要的结论不是“局地状态调制机制已经成立”，而是：

1. **每个风场最近 3 h 的正确功率历史具有真实、稳定且广泛的增量预测价值。**
2. **现有结果不支持把这项增益专门归因于 weather × state interaction。** 与 concat interaction 参数化匹配的 additive history control 表现相当或更好。
3. 当前最佳模型是多阶段训练的 `P-Add-Free`，其结构应概括为：

   \[
   \text{local history baseline}
   +\text{local weather main effect}
   +\text{local history main effect},
   \]

   然后将 56 个场站的预测按容量恢复为 MW 并物理求和。它不是已经得到机制验证的“state-conditioned weather response”模型。
4. **集群预测路线应继续。** 当前证据支持“场站作为局地计算单元、跨场共享学习、场站监督预训练、集群联合微调、最后物理求和”，不支持退回 56 个完全独立模型后再简单求和。
5. **停止继续搜索 interaction operator。** 如果后续仍要主张历史状态改变 weather→power transfer law，必须在旧 4×4 周边气象代理上做输入完全匹配的 no-state/additive/interaction 对照；本轮双线性点 forcing 不能回答旧 4×4 机制是否成立。

当前推荐的工程模型为 `P-Add-Free`；当前可辩护的论文主线是“局地历史增量 + 跨场共享 + 多阶段优化 + 物理聚合”，而不是“状态调制天气响应”。

## 2. 实验设计与模型边界

### 2.1 固定的数据与结构

- 56 个风场全部参与训练与评价；
- Pangu forcing 在场站坐标处采用固定双线性插值；
- 每个风场使用最近 3 h、12 个 15 min capacity-factor 历史点；
- weather encoder、history encoder、response network 和 decoder 跨场站共享；
- 本轮没有启用 station embedding、station adapter 或 GNN；配置中的 embedding table 在 `station_context_mode=none` 下不进入前向计算；
- 每个场站得到 24 步预测，再按场站容量恢复 MW 并严格求和为 cluster forecast；
- 联合阶段的相对目标权重等价于

  \[
  L=L_{\rm cluster}+0.5L_{\rm station};
  \]

- Stage 1 按等权 station macro CF MAE 选 checkpoint；Stage 2 按 cluster MAE 选 checkpoint；
- 正式结果使用 seeds 42–51，共 10 seeds。

所有模型都已有从本场历史功率到未来轨迹的 baseline/current-anchor 路径。因此，本轮所谓 `state` 实验检验的准确问题是：

> 在历史 baseline 已经存在后，把同一局地历史再送入 response branch，是否仍有增量；该增量必须通过 weather×history 联合映射才能获得，还是一个独立的 history main effect 已经足够？

它并不是“历史功率是否对预测有用”的从零比较。

### 2.2 对照组定义

| 模型 | 训练方式 | response branch 中的额外历史路径 |
|---|---|---|
| `E0` | 从头端到端 | 无 |
| `E-Add` | 从头端到端 | weather main effect + history main effect，无联合交互 |
| `E-Int` | 从头端到端 | `MLP([weather, history state])` |
| `P0-Free` | station-only 预训练 → 自由联合微调 | 无 |
| `P-Add-Free` | station-only 预训练 → 自由联合微调 | 加性 history main effect |
| `P-Int-Free` | station-only 预训练 → 自由联合微调 | concat interaction |
| `P-Int-Preserve` | station-only 预训练 → 联合微调 | concat interaction；Stage 2 冻结 state encoder 与 interaction |

`Additive` 不是简单删除历史，而是 interaction 的关键机制对照。代码分别计算 weather-only 与 history-only 的主效应，不进行二者的联合前向映射。因而：

\[
\text{Int} \not< \text{Add}
\]

时，不能声称性能增益必须依赖 weather×history interaction。

## 3. 10-seed 主结果

| 模型 | 2023 validation MAE (MW) | 2024 frozen test MAE (MW) | 2023 station macro CF MAE |
|---|---:|---:|---:|
| `E0` | 270.662 ± 1.586 | 269.864 ± 3.111 | 0.09438 |
| `E-Add` | 268.877 ± 1.782 | 266.747 ± 4.112 | 0.09329 |
| `E-Int` | 268.977 ± 1.315 | 264.976 ± 3.593 | 0.09349 |
| `P0-Free` | 268.927 ± 2.749 | 266.367 ± 2.744 | 0.09197 |
| **`P-Add-Free`** | **264.973 ± 3.265** | **259.857 ± 2.030** | **0.09008** |
| `P-Int-Free` | 266.177 ± 3.793 | 262.279 ± 2.790 | 0.09058 |
| `P-Int-Preserve` | 266.742 ± 4.011 | 263.863 ± 2.706 | 0.09069 |

验证集选择与 2024 冻结测试都由 `P-Add-Free` 最优。2024 只用于冻结后的外年检验，不参与模型反选。

## 4. 历史路径有用，但 interaction 没有得到识别

以下差值统一为 candidate − control；负值表示 candidate 更好。置信区间为 10 个配对 seed 的 95% CI。

### 4.1 Station-only 阶段

| 比较 | mean Δ | 95% CI | wins |
|---|---:|---:|---:|
| `S-Add − S0` | −0.001506 CF | [−0.001689, −0.001323] | 10/10 |
| `S-Int − S0` | −0.001428 CF | [−0.001603, −0.001253] | 10/10 |
| `S-Int − S-Add` | +0.000078 CF | [−0.000075, +0.000232] | 3/10 |

局地历史的额外 response path 在单场监督目标下明确有效，但 interaction 并没有超过 additive control。

### 4.2 从头端到端训练

| 比较 | 2023 mean Δ | 95% CI | wins | 2024 mean Δ |
|---|---:|---:|---:|---:|
| `E-Add − E0` | −1.785 MW | [−3.312, −0.259] | 8/10 | −3.116 MW |
| `E-Int − E0` | −1.684 MW | [−2.843, −0.526] | 8/10 | −4.888 MW |
| `E-Int − E-Add` | +0.101 MW | [−1.260, +1.462] | 5/10 | −1.771 MW，CI 跨 0 |

端到端训练下，历史路径相对 no-extra-state control 有增益；但 additive 与 interaction 在验证集不可区分。

### 4.3 多阶段训练

| 比较 | 2023 mean Δ | 95% CI | wins | 2024 mean Δ | 2024 95% CI / wins |
|---|---:|---:|---:|---:|---:|
| `P-Add − P0` | **−3.954 MW** | **[−6.853, −1.054]** | 8/10 | **−6.510 MW** | **[−8.509, −4.511], 10/10** |
| `P-Int − P0` | −2.750 MW | [−5.593, +0.093] | 7/10 | −4.088 MW | [−7.039, −1.137], 8/10 |
| `P-Int − P-Add` | +1.204 MW | [−0.963, +3.371] | 5/10 | **+2.422 MW** | **[+0.284, +4.559], 2/10** |

这组结果是全实验最关键的机制判定：

- `P-Add` 对 `P0` 的增益明确、跨年稳定；
- `P-Int` 没有在验证集显著超过 `P0`；
- `P-Int` 没有超过 `P-Add`，且在冻结的 2024 上反而显著更差。

因此正确表述是：

\[
\boxed{\text{局地历史有增量价值，但没有证据证明增量来自天气响应律被状态调制。}}
\]

## 5. 为什么状态干预不能证明 interaction

对 concat interaction 模型，替换历史状态会产生很大退化：

| 模型阶段 | strong temporal donor | far-station permutation | reference state | positive seeds |
|---|---:|---:|---:|---:|
| `S-Int` station-only | +0.031036 CF | +0.023183 CF | +0.009508 CF | 均为 10/10 |
| `E-Int` validation | +235.524 MW | +126.635 MW | +48.989 MW | 均为 10/10 |
| `P-Int-Free` validation | +264.908 MW | +158.704 MW | +53.178 MW | 均为 10/10 |

这些干预严格证明：concat 模型大量依赖“正确样本、正确场站”的历史表示。它们不能单独证明：

\[
\frac{\partial^2 \hat P}{\partial W\,\partial S}\ne0
\]

是性能改善的来源。原因是 `MLP([W,S])` 同时允许：

- history main effect；
- weather main effect；
- weather×history interaction。

打乱 \(S\) 会同时破坏前两类与第三类中所有依赖 \(S\) 的成分。只有与 matched additive control 比较，才能判断联合交互是否必要；而本轮 `Int − Add` 没有给出正面证据。

此外，正式 10-seed intervention 集中在 interaction 模型；不能把这些退化数字直接记到最终选择的 `P-Add-Free` 名下。`P-Add-Free` 的历史价值由其相对 `P0-Free` 的配对性能增益及广泛的逐场改善支持。

## 6. 多阶段训练的真实作用

### 6.1 多阶段相对端到端

| 比较 | 2023 mean Δ | 95% CI | wins | 2024 mean Δ |
|---|---:|---:|---:|---:|
| `P0 − E0` | −1.735 MW | [−3.845, +0.376] | 7/10 | −3.497 MW |
| `P-Add − E-Add` | **−3.904 MW** | **[−6.292, −1.515]** | 9/10 | **−6.890 MW** |
| `P-Int − E-Int` | −2.801 MW | [−5.561, −0.040] | 7/10 | −2.697 MW |

多阶段训练明显改善绝对性能，尤其对 additive 模型有效。合理解释是：station-only 预训练先建立可辨识的局地预测单元，随后 cluster+station 联合微调再校准区域总量。这是一种有效的优化课程与监督组织方式。

但不能进一步声称它专门“保存了状态交互机制”。差分中的差分未显著：

- interaction DID，validation：−1.066 MW，95% CI [−4.533, +2.402]；
- additive DID，validation：−2.169 MW，95% CI [−5.876, +1.539]。

两者都跨 0，而且 additive 的点估计更大。因此当前证据支持“多阶段训练提升预测”，不支持“多阶段训练特异地保护 interaction”。

### 6.2 Free 与 Preserve

| 比较 | 2023 mean Δ | 95% CI | 2024 mean Δ | 2024 95% CI |
|---|---:|---:|---:|---:|
| `P-Int-Preserve − P-Int-Free` | +0.565 MW | [−0.236, +1.365] | **+1.584 MW** | **[+0.853, +2.314]** |

冻结 state encoder 与 interaction 没有更好，2024 还显著更差。这排除了“必须硬冻结才能避免 aggregate fine-tuning 遗忘局地机制”的说法。自由联合微调应作为默认方案。

## 7. 收益来自哪些时效

`P-Add-Free − P0-Free` 的分时效差值为：

| 时效/指标 | 2023 validation Δ | 2024 frozen test Δ |
|---|---:|---:|
| 0–1 h MAE | +0.276 MW | +0.009 MW |
| 1–3 h MAE | −1.150 MW | −3.816 MW |
| 3–6 h MAE | **−7.233 MW** | **−10.478 MW** |
| RampMAE | −1.824 MW | −1.856 MW |

历史增量路径不是在超短时效重复 persistence 的主要收益；其优势集中在中长时效，尤其 3–6 h，并同时改善 ramp。比较稳妥的解释是：共享 baseline/current anchor 负责短时连续性，额外历史表征为轨迹演化提供持续动态信息。仅凭这些结果仍不能把该信息解释成某一种具体设备状态、限电状态或 transfer-law regime。

## 8. 场站覆盖与禁止事后挑选

按 10-seed 平均逐场 CF MAE 统计：

| 比较 | 改善场站数 |
|---|---:|
| `S-Add − S0` | 53/56 |
| `S-Int − S0` | 53/56 |
| `E-Add − E0` | 49/56 |
| `E-Int − E0` | 50/56 |
| `P-Add − P0` | **55/56** |
| `P-Int − P0` | 51/56 |

`P-Add` 的收益不是由几个特殊风场驱动，因此没有依据为了制造更强结论而筛选“支持假设”的场站。主结果必须保留完整 56 场：

- full-farm station macro 是局地能力证据；
- full-cluster physical sum 是最终任务证据；
- 逐场差异只能用于异质性描述，或用于预先定义的容量、地形、数据质量等分层敏感性分析；
- 任何场站剔除必须依据与模型结果无关的预注册数据质量标准，不能按增益大小事后筛选。

## 9. 与既有集群基线的关系

既有记录中的参考结果为：

| 模型 | 2023 MAE | 2024 MAE |
|---|---:|---:|
| `A0-FT` aggregate baseline | 273.195 MW | 270.325 MW |
| `L1` local no-state | 269.437 MW | 266.837 MW |
| `G-Static` previous best | 267.985 MW | 264.817 MW |
| **`P-Add-Free`** | **264.973 MW** | **259.857 MW** |

`P-Add-Free` 的均值改善为：

- 相对 `G-Static`：−3.012 / −4.960 MW；
- 相对 `A0-FT`：−8.222 / −10.468 MW；
- 相对 `L1`：−4.464 / −6.980 MW。

这些旧基线不是本轮逐 seed 的完全配对控制，因此上述差值用于定位当前数值水平，不能替代本轮 `P-Add vs P0`、`P-Int vs P-Add` 的正式机制检验。就已记录的实验均值而言，`P-Add-Free` 是当前最强模型。

## 10. 对自动报告中“local state identified”的修正

机器汇总文件给出了：

```json
"local_state_identified": true,
"cluster_state_survives": true
```

这两个字段的程序判据主要来自 interaction 相对 no-state 的提升和状态干预敏感性，命名过强。它们在论文与后续总结中只能解释为：

- 正确局地历史条件被模型使用；
- 该历史条件的预测增量在集群联合目标下仍可观察到。

它们不能解释为：

- 已经识别出一个具有明确物理含义的隐状态；
- 已经证明该状态改变局地 Pangu forcing 到功率的 transfer function；
- 已经证明 multi-stage 特异地保存了 weather×state interaction。

更准确的机器字段应是：

```text
local_history_incremental = true
local_history_used_by_concat_model = true
weather_state_interaction_identified = false
multistage_interaction_preservation_identified = false
```

## 11. 旧单风场 4×4 结果与本轮结果为什么不矛盾

本轮 forcing 是场站坐标处的双线性局地点值；旧单风场实验使用场站周围 4×4 气象块形成代理气象表示。两者提供的空间信息不同：

\[
W_i^{\rm bilinear}\ne E_W(W_{\Omega_i}^{4\times4}).
\]

因此，本轮结果不能否定“在 4×4 周边气象代理下，历史对空间天气表示存在有用条件化”；同样，旧结果也不能直接证明该机制能迁移到双线性局地 forcing 和 56 场共享模型。

此外，concat 模型本身允许 history-only predictive path。除非旧实验也包含参数、训练预算和输入完全匹配的 additive control，否则旧 concat 的优势同样不能全部解释为纯 interaction。

如果必须继续追问 transfer-law modulation，唯一有判别力的下一步是：固定旧 4×4 输入和原单场最佳 backbone，在 no-extra-history、additive history、concat interaction 三个模型之间做配对多 seed 比较，并同时报告 interaction 相对 additive 的增量。继续换 gate、乘法、低秩或 embedding 不会解决当前的可归因性问题。

## 12. 路线建议与论文表述

### 12.1 当前应保留的路线

建议保留以下层级：

```text
每场局地 Pangu forcing + 每场历史功率
                  ↓
       跨场共享的局地预测网络
                  ↓
     station-supervised pretraining
                  ↓
 cluster + station aggregation-aware fine-tuning
                  ↓
          容量恢复与物理求和
```

工程上采用 `P-Add-Free`：

\[
\hat P_{\Sigma,q}
=
\sum_{i=1}^{56} C_i\,
F_{\rm add}(H_i,W_i,q).
\]

这个方案同时保留局地作用单元、跨场统计共享和集群最终目标，不需要在“单风场预测”与“集群预测”之间二选一。

### 12.2 当前不应声称的内容

不应声称：

- history state 已被证明在物理上调制 weather response；
- aggregate optimization 会遗忘 interaction，而冻结成功保护了它；
- 某些挑选出来的场站证明了普适机制；
- station embedding 是当前改进的必要条件。

### 12.3 可辩护的科学问题与答案

科学问题应写成：

> 在局地 Pangu forcing 与以当前功率为锚的轨迹预测之外，风场最近运行历史是否仍包含可跨场共享、并能改善区域集群预测的动态信息？这种信息必须通过改变天气—功率响应关系发挥作用，还是作为独立的历史动态修正已经足够？

本轮答案是：

> 最近局地功率历史确实包含广泛、跨年稳定的增量信息，且局地监督预训练后再进行集群联合微调能最好地利用该信息；但是 matched additive control 不弱于 concat interaction，因此目前只能确认独立历史动态修正，不能确认天气—功率响应律被状态调制。

### 12.4 是否加入 station embedding

当前正式最佳模型在不启用 station identity 的条件下已经实现 55/56 场改善，说明 station embedding 不是建立当前结论的必要条件。若以后研究场站异质性，必须固定 `P-Add-Free` 的训练流程，至少做以下 2×2 matched control：

| | 无 station ID | 有 station ID |
|---|---|---|
| 无额外 history path | control | ID-only |
| additive history path | `P-Add-Free` | Add+ID |

只有 `Add+ID` 相对 `ID-only` 仍保留历史增量，才能说 station identity 没有替代 history；不能直接加入 embedding 后把任意 MAE 改善解释为 state 更有效。

## 13. 最终决策

1. **模型选择：** 冻结并保留 `formal_transfer/p_add_free`。
2. **任务选择：** 继续 56 场集群预测，以场站为计算单元，不退回完全独立的单场模型集合。
3. **训练选择：** 保留 station-only pretraining → free aggregation-aware fine-tuning；不冻结 interaction/state 模块。
4. **机制选择：** 把“局地历史加性动态修正”作为当前结论；停止把 interaction/state modulation 作为已验证主张。
5. **场站选择：** 不按实验增益事后挑场；完整 56 场作为主证据。
6. **后续优先级：** 首先稳定复现和整理 `P-Add-Free`；只有论文必须回答旧单风场 modulation 时，才在完全相同的 4×4 forcing 下补 no-state/additive/interaction 严格对照。

## 14. 证据位置

- 冻结选择：`artifacts/local_state_transfer_multistage_v1/selection/validation_freeze.json`
- 机器汇总：`artifacts/local_state_transfer_multistage_v1/analysis/summary.json`
- 自动结果报告：`exp_docs/local_state_transfer_multistage_v1_results.md`
- 完成标志：`artifacts/local_state_transfer_multistage_v1/COMPLETE`
