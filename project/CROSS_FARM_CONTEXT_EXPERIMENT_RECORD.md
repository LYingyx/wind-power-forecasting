# 跨场站 recent-state 上下文：正式实验记录

日期：2026-08-31

## 一句话结论

固定地理邻居、全局其他场站摘要和 NWP 一致性邻居都没有在 10 seeds 下证明“其他场站 recent power 对目标场站有稳定的结构增益”。当前主线应停在

\[
\hat Y=B(H)+C(S_{\Sigma,\mathrm{recent}},S_{i,\mathrm{recent}},W_i,\tau),
\]

不加入跨场站消息传递，不上 GNN，不为这些结构重建 teacher/KD。G4 存在值得记录的 ensemble/长时效信号，但它更像多样性收益，而不是已确立的跨场站机制。

## 原计划的漏洞与修正

原问题“其他场站的近期状态是否还能补充目标场站自身状态”值得验证，但不能直接用 GNN 回答。正式实验作了三项修正：

1. 用严格冗余的 `self + self duplicate` 作为 G1，使 G1--G4 都有相同双通道编码器；四组参数量完全一致，均为 `1,287,407`。
2. 将 `global-other` 定位为非局地、高冗余控制，而非 regional mechanism。其与 aggregate recent 的相关系数为 `0.996`，`self + aggregate` 已解释其 `99.1%` 方差。
3. 不把多一个编码分支的容量当作信息收益。逐 seed 配置审计确认 G1--G4 只在 `cross_farm_context_mode` 与 experiment ID 上不同。

最终矩阵为：G0 aggregate recent（复用最终 Normal）；G1 self + self duplicate；G2 self + geographic K4；G3 self + global-other mean；G4 self + weather-coherent K4。G4 仅用发布时已知的 NWP 前端输出，按未来 8 h weather-only 场站轨迹 MSE 动态选 K4，再池化 recent power；top-k 不反传梯度，没有目标泄漏或 learned graph。

G0 的参数量为 `1,287,343`，比双通道 G1 少 64 个参数，所以 G1 vs G0 在本轮只作最终配置下的方向性复核，不作为严格容量匹配的因果消融；跨场站主问题 G2/G3/G4 vs G1 才是完全参数匹配的正式检验。

正式口径：seeds 42--51，Normal training，无 KD，相同 B、weak anchor、NWP 前端和训练预算；只使用 2023 validation。单模型用同 seed 配对区间，ensemble 用 10,000 次 UTC-day block bootstrap（359 日），G2/G3/G4 vs G1 做 Holm 校正。候选进入主模型必须同时有稳定证据、无严重 horizon 退化，且 single 与 ensemble 中心都约改善至少 2 MW。

## 4×4 NWP 空间支持决策

10 个 checkpoint、每 seed 8,146 个 NWP cycles 的诊断结果为：有效 token 中位数 `12.85/16`，edge relative mass `0.815`，far-quartile relative mass `0.558`，平均有效半径 `31.05 km`，attention 质心 support fraction 的中位数/p95 为 `0.193/0.362`。换算后的实际 edge mass 约 `0.611`（均匀基准 `0.75`），最远四分位实际质量约 `0.140`（均匀基准 `0.25`）。

因此 attention 虽较分散，却没有向窗口边界或更远格点聚集的压力。按预注册规则冻结 4×4，不做正式 6×6 搜索，也不做 state-conditioned spatial attention。该结论只支持“4×4 是当前合理默认值”，不证明它全局最优。

## 10-seed 结果

| 组别 | 单模型 MAE，mean ± std（MW） | 10-model ensemble（MW） | single--ensemble gap（MW） |
|---|---:|---:|---:|
| G0 aggregate recent | 258.563 ± 1.816 | 244.732 | 13.831 |
| G1 self duplicate | 257.480 ± 1.793 | 243.916 | 13.565 |
| G2 geographic K4 | 257.205 ± 1.996 | 243.369 | 13.836 |
| G3 global-other | 257.163 ± 2.124 | 243.434 | 13.729 |
| G4 weather-coherent K4 | **256.858 ± 2.307** | **242.412** | **14.446** |

下表使用“候选减参考”，负值表示候选更好：

| 对比 | 单模型配对差 [95% CI]（MW） | 胜场 | Holm p | ensemble 差 [按日 95% CI]（MW） |
|---|---:|---:|---:|---:|
| G1 vs G0 | -1.082 [-2.328, 0.164] | 7/10 | -- | -0.816 [-2.268, 0.626] |
| G2 vs G1 | -0.275 [-1.163, 0.612] | 5/10 | 1.000 | -0.547 [-1.655, 0.595] |
| G3 vs G1 | -0.317 [-1.077, 0.442] | 7/10 | 1.000 | -0.482 [-1.155, 0.165] |
| G4 vs G1 | -0.622 [-2.014, 0.770] | 7/10 | 1.000 | **-1.504 [-2.494, -0.488]** |

G2 与 G3 几乎等价：单模型差 `+0.042 MW`，ensemble 差 `-0.065 MW`，区间均跨 0。因此没有证据表明地理局地性优于任意其他场站摘要。G2 的 K4 距离中位数为 `21.06 km`、均值 `25.96 km`；该输入确有额外信息（`self + aggregate` 只解释 `76.5%` 方差），所以 G2 的负结果不是因为它与已有输入完全相同。

## G4 为什么不能被直接选中

G4 的 ensemble 总体区间为正面证据，且相对 G1 的分时效差为：

- 0.25--1 h：`-0.162 [-0.346, 0.028] MW`；
- 1.25--3 h：`-0.623 [-1.420, 0.224] MW`；
- 3.25--6 h：`-2.538 [-4.000, -1.012] MW`。

但它的 single-model seed 差异很大，范围约为改善 `3.475 MW` 到恶化 `3.361 MW`，配对区间跨 0，Holm 后也不成立，且 single 中心只改善 `0.622 MW`。其 ensemble 优势 `1.504 MW` 可分解为约 `0.622 MW` 的平均单模型中心差和 `0.882 MW` 的额外 ensemble gap；约 59% 来自跨 seed 多样性增加。因此当前证据支持“G4 可能是 ensemble 工程候选”，不支持“weather-coherent 邻居是稳定预测机制”，更未达到预注册的 2 MW 主模型门槛。

## 对既有 self-state 证据的影响

先前冻结 B0 的专门 2×2 实验中，station recent 相对 aggregate recent 的单模型改善为 `1.455 [0.434, 2.476] MW`，ensemble 改善为 `1.939 [0.799, 3.091] MW`。本轮最终 joint/weak-anchor 配置下，G1 vs G0 的两个区间都跨 0，未独立复现同等强度。

两者并不矛盾：冻结 B0 更接近信息增量识别；joint training 时 B 可重新适配并吸收一部分状态信息。合理表述应是：目标场站 recent trajectory 有此前受控机制证据，本轮中心方向一致但部署增益减弱且配置依赖。不能把本轮写成再次证明 self-state，也没有理由把不稳定的跨场站结构叠加到主模型。

## 最终决策与故事线

1. 其他场站可被解释为潜在区域天气状态的分布式观测，但本数据没有证明这种信息在 `aggregate + self` 之外形成稳定可学习增量；不能表述为功率传播或场站间因果影响。
2. 主故事保持为：历史基线给出内生演化，目标场站/集群近期状态描述当前 realized state，局地 NWP 描述未来外部 forcing，horizon-aware correction 学习两者的条件关系。
3. 冻结 4×4；停止 GNN、learned graph、更多 K 搜索、6×6 正式搜索和 state-conditioned spatial attention。
4. 不为 G2/G3/G4 重建 teacher，不做 KD。若未来目标转为纯工程 ensemble，可用现有 G4 checkpoints 单独研究异构混合；该方向不得回写为当前机制贡献。

## 结果位置

- 正式训练：`artifacts/cross_farm_context_v1/groups/`
- 完整统计：`artifacts/cross_farm_context_v1/analysis/analysis.json`
- 汇总表：`artifacts/cross_farm_context_v1/analysis/summaries.csv`
- 配对比较：`artifacts/cross_farm_context_v1/analysis/comparisons.csv`
- horizon：`artifacts/cross_farm_context_v1/analysis/horizon_bands.csv`
- 4×4 诊断：`artifacts/cross_farm_context_v1/spatial_diagnosis/diagnosis.json`

审计：40/40 正式 runs 完成；40 份 validation metrics 与 predictions 齐全；实验目录内 test 产物为 0；跨场站上下文参数匹配与信息构造的直接回归检查通过。
