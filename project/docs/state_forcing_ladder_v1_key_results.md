# 冻结基线状态阶梯：关键实验结论

日期：2026-08-27

## 结论

附件的研究方向是对的，但“已有明确机制证据”需要收窄为：已有较强的**预测机制证据**，尚未识别物理因果。严格冻结历史基线后，correction 对未来天气的有效利用确实依赖历史状态；其中最重要的不是当前场站空间快照，也不是完整 6h 长记忆，而是：

> **当前聚合 operating level + 短时演化趋势 + 最近四个 15 分钟功率状态点。**

因此下一代主线应从泛化的 `C(H,W)` 收敛为 **compact recent-state conditioned future forcing correction**。完整冻结只适合作机制探针；最终预测模型仍应使用弱 anchor，因为冻结 full 的 ensemble MAE 比 anchor 0.20 差 6.151 MW。

## 对原分析的修正

- 四格 MAE 的差中之差只能证明预测误差上的非加性互补，不能推出函数分解或物理因果上的 `f(H,W) != f(H)+g(W)`。
- `P=F(s,W,τ)` 和偏导表达只能作设计启发；本实验没有做天气干预或同天气反事实。
- 状态阶梯必须固定同一个 `B0(H)`，否则 baseline 会补偿 correction 缺失的信息。本实验的 60 个 checkpoint 均逐张量确认 `B0` 与原模型完全相同。
- 阶梯并非一条完全嵌套的单轴序列：station-current 检验空间快照，ramp 检验聚合趋势，recent-state 同时包含近期场站与聚合轨迹，full-vs-recent 才直接检验更长记忆。
- 旧联合训练实验中 0.25h 的状态条件化收益为负；冻结协议下 full-vs-weather 在 0.25h 为 `+0.769 MW [0.321, 1.208]`。稳健结论是收益随时效增强，不是极短时效必然有害。

## 实验协议与审计

- 6 个条件 × 10 个随机种子 `42–51`，共 60 次独立训练；全部 `COMPLETE`。
- 仅使用 2023 validation；每个预测文件包含 32,550 个完全对齐 origins、359 个 UTC 日；test 未使用。
- 每组使用相同的 1,287,343 参数结构、full-frequency correction、相同训练预算与 early stopping；只改变 correction 可见的状态。
- 共同的预训练 `B0(H)` 始终看完整历史并完全冻结；60/60 checkpoint 状态精确一致。baseline 预测重放平均绝对差 0.011 MW，最大 0.040 MW。
- 单模型差异按相同 seed 配对；ensemble 差异使用 10,000 次 sample-weighted paired UTC-day block bootstrap。

状态输入依次为：无 correction-state、当前聚合功率、当前 56 站向量、当前聚合功率加 15 分钟 ramp、最近四个 15 分钟场站/聚合状态点、完整 24 个历史状态点。

## 主结果

| correction 状态 | 单模型 MAE（MW） | 10-model ensemble MAE（MW） |
|---|---:|---:|
| weather-only | 300.716 ± 3.300 | 287.404 |
| cluster-current | 282.879 ± 3.500 | 268.894 |
| station-current | 281.825 ± 3.461 | 268.268 |
| cluster-current + 15min ramp | 270.785 ± 4.128 | 256.105 |
| recent four points | 265.759 ± 3.770 | 250.637 |
| full 24 points | **264.858 ± 3.337** | **249.067** |

下表把改善写成正值：

| 信息增量 | 单模型配对改善 [95% CI] | 胜场 | ensemble 改善 [按日 95% CI] | 判断 |
|---|---:|---:|---:|---|
| cluster-current vs 无状态 | 17.837 [16.611, 19.062] | 10/10 | 18.509 [14.737, 22.157] | operating level 必要 |
| station-current vs cluster-current | 1.054 [0.363, 1.745] | 8/10 | 0.626 [-0.312, 1.575] | 空间 snapshot 证据混合且很弱 |
| 15min ramp vs cluster-current | 12.094 [11.170, 13.019] | 10/10 | 12.789 [10.483, 15.115] | 短时趋势必要 |
| recent four points vs 15min ramp | 5.026 [4.030, 6.022] | 10/10 | 5.467 [4.013, 6.943] | 简单一阶 ramp 不充分 |
| full 24 points vs recent four points | 0.900 [-0.144, 1.945] | 8/10 | 1.571 [0.199, 2.892] | 小幅、对训练随机性不稳健 |

最强结论是：当前总功率只是第一层状态；加入演化趋势和近期轨迹后仍有稳定增益。当前 56 站 snapshot 的额外价值未同时通过两类不确定性检验，不能据此把“空间链接/图网络”定为下一步主线。完整历史的边际收益远小于近期状态，暂不足以支撑重型长记忆架构。

## 时效与 regime

- full-history 相对 weather-only 的 ensemble 改善从 0.25h 的 0.769 MW 增至 1h 的 9.961、3h 的 45.256、6h 的 56.109 MW；与 lead 的相关系数为 0.939，24/24 个 lead 均为正。
- ramp 相对 current 在 0.25h/0.5h 分别恶化 2.081/1.304 MW，从 1h 开始转为可靠正收益，6h 改善 20.914 MW。这支持让趋势作用随 horizon 增强，而不是所有时效统一融合。
- full 相对 weather-only 的总体条件化收益为 38.337 MW；0–1h、1–3h、3–6h 分别为 4.953、30.857、54.451 MW。
- 条件化收益在高当前功率、高近期波动、高场站离散度样本中分别比低分位高 53.728、51.172、28.779 MW，区间均不跨 0。但离散度结果是边际关联，而受控 station-vs-cluster ladder 很弱，因此不能据此宣称空间因果机制。
- 强下降和强上升 ramp 相对中间稳定组的条件化收益分别高 28.125 和 36.439 MW；状态调制主要发生在动态工况，而非只对应某一个 ramp 方向。
- 各天气 regime 对“状态条件化增益”的极端分位差异均无可靠证据。天气趋势却与 correction 方向一致：最强下降组平均修正 -252.102 MW、正修正概率 0.245；最强上升组为 +367.835 MW、概率 0.760。两组 correction usefulness 分别为 194.367 和 260.474 MW。
- correction usefulness 从 0.25h 的 0.149 MW（区间跨 0）增至 6h 的 333.985 MW `[310.933, 357.616]`，与 horizon 的相关系数为 0.997。

## 最终框架与下一步

机制测试与最终模型应分开：

1. **机制形式**：`Y = B0(H) + C(S_recent, W, τ)`，冻结 `B0`，用于识别 correction 真正需要的状态。
2. **部署形式**：`Y = B(H) + C(S_recent, W, τ)`，让 `B` 可训练但以预训练 `B0` 做弱 anchor；不再使用 DCT 低频硬约束。
3. **状态模块**：优先编码 current aggregate、15min/1h ramps 和最近四个状态点；暂不增加完整 6h 重型 memory。
4. **融合模块**：显式做 state–weather interaction，并使用 horizon gate，使趋势/轨迹分支在短时弱、长时强。
5. **空间决策实验**：冻结 B0 的 `aggregate-vs-station × snapshot-vs-recent-trajectory` 2×2 已完成。station recent 相对 aggregate recent 有稳定但较小的增益，因此值得做轻量空间链接的受控消融；现阶段仍不能把图传播本身当作已验证机制。
6. **最终确认**：方案和超参数在 validation 上冻结后，再一次性使用未触碰 test；当前所有结论仍是单数据集 validation 证据。

创新性判断：作为纯模型算子属于中等创新；作为“用冻结基线识别 operating state、trajectory 与 future forcing 的预测关系，并据此设计 horizon-aware correction”的科学故事较清楚，值得继续。当前最有支撑的创新点是**近期动态状态条件化**，不是长记忆，也不是空间图本身。

## 结果位置

- 完整训练：`artifacts/state_forcing_ladder_v1/groups/`
- ladder 汇总与逐时效统计：`artifacts/state_forcing_ladder_v1/analysis/`
- 空间 snapshot × recent trajectory 2×2：`artifacts/state_forcing_ladder_v1/spatial_trajectory_analysis/`
- frozen regime/correction 统计：`artifacts/state_forcing_diagnostics_frozen_v1/`
- 运行队列：`artifacts/state_forcing_ladder_v1/logs/jobs.tsv`

## 补充验证：空间 snapshot × recent trajectory 2×2

为区分“近期轨迹价值”和“空间轨迹价值”，补齐了缺失的 aggregate-recent 单元，并与已有三个单元组成严格 2×2。四格共享相同的冻结 B0、未来天气、correction backbone、训练预算和 10 个随机种子，只改变 correction 可见的功率状态。

| correction 状态 | 单模型 MAE（MW） | 10-model ensemble MAE（MW） |
|---|---:|---:|
| aggregate snapshot | 282.879 ± 3.500 | 268.894 |
| station snapshot | 281.825 ± 3.461 | 268.268 |
| aggregate recent 1h | 267.214 ± 3.985 | 252.577 |
| station recent 1h | **265.759 ± 3.770** | **250.637** |

下表把 MAE 改善写成正值：

| 受控信息增量 | 单模型配对改善 [95% CI] | 胜场 | ensemble 改善 [按日 95% CI] |
|---|---:|---:|---:|
| aggregate trajectory vs aggregate snapshot | 15.665 [14.483, 16.848] | 10/10 | 16.317 [13.557, 19.026] |
| station trajectory vs station snapshot | 16.066 [14.844, 17.289] | 10/10 | 17.631 [14.894, 20.253] |
| station snapshot vs aggregate snapshot | 1.054 [0.363, 1.745] | 8/10 | 0.626 [-0.313, 1.589] |
| station recent vs aggregate recent | 1.455 [0.434, 2.476] | 9/10 | 1.939 [0.799, 3.091] |
| aggregate recent vs aggregate current + 15min ramp | 3.571 [2.705, 4.437] | 10/10 | 3.528 [2.576, 4.466] |

2×2 的空间×轨迹交互项在 ensemble 误差上为 1.314 MW [0.439, 2.222]，但逐 seed 为 0.401 MW [-0.822, 1.623]，仅 7/10 为正。因此，“相同近期时间窗下，场站级轨迹比聚合轨迹更好”已经同时通过两类不确定性检验；但“轨迹会稳定放大空间信息价值”仍只得到混合证据，不能当作已确立机制。

逐时效结果进一步表明：aggregate trajectory 在 0.25h 恶化 1.082 MW [-1.421, -0.755]，0.5h 基本持平，从 0.75h 开始可靠改善，3h 和 6h 分别改善 18.908 和 23.671 MW。recent spatial benefit 在 24/24 个 lead 为正，其中 22/24 的逐 lead 区间为正，从 0.25h 的 0.322 MW 增至 6h 的 2.997 MW。逐 lead 区间未作多重比较校正，应主要用于判断形态。

统一结论是：

1. **主机制是近期动态状态，不是空间 snapshot。** 最近 1h 的聚合轨迹带来约 16.3 MW ensemble 改善，远大于空间轨迹的约 1.9 MW。
2. **单一 15min ramp 不充分。** 再加入 30/45min 前的聚合状态仍稳定改善约 3.5 MW，因此状态模块应保留最近四点，而不能只保留 current + 一阶差分。
3. **空间信息需要以轨迹而非单帧表达。** 当前场站 snapshot 的 ensemble 区间跨 0，而相同时间窗下 station recent 稳定优于 aggregate recent；这把空间状态从“不值得做”提升为“值得做受控的轻量扩展”。
4. **尚不足以直接宣称空间传播或图机制。** 现有结果只证明场站分辨率近期功率轨迹提供预测增量，未证明地理边、图消息传播或物理因果。下一步若做空间链接，必须在 station-recent 输入不变时比较无图 pooling、固定邻接和 learned graph；只有图本身继续提供稳定增益，才能把空间链接提升为创新主张。
5. 最终框架仍应以 compact recent-state conditioned future forcing correction 为主线：最近四个聚合/场站状态点编码动态状态，显式与未来天气交互，并用 horizon gate 抑制极短时轨迹分支；空间模块是第二层增量，不应取代这条主线。

补充审计：新增组 10/10 runs 完成；四格 40 个预测档案均为 2023 validation，32,550 个 origins 和 targets 完全对齐，共 359 个 UTC 日；参数量均为 1,287,343。新增 10/10 checkpoint 的 B0 状态逐张量精确冻结，baseline 重放平均绝对差 0.011 MW、最大 0.040 MW；未生成或使用 test 产物。正式结果保存在 `artifacts/state_forcing_ladder_v1/spatial_trajectory_analysis/`。
