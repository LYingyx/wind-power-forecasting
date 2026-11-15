# 单风场尺度历史功率—未来气象实验完整报告

## 1. 结论摘要

本轮实验回答了三个彼此不同的问题。

第一，在参与监督训练的 56 个风场上，未来局地 Pangu forcing 与本场历史功率具有稳定的**预测非可分离性**。与参数量匹配、history/weather 两分支完全独立的 `HW-IAdd` 相比，`HW-Int` 的 macro station CF MAE 为：

| 年份 | HW-IAdd | HW-Int | Int−IAdd 配对差 | 95% CI | 获胜 seeds |
|---|---:|---:|---:|---:|---:|
| 2023 validation | 0.089537 | **0.087629** | **−0.001907** | [−0.002059, −0.001756] | 10/10 |
| 2024 frozen test | 0.086089 | **0.084760** | **−0.001329** | [−0.001777, −0.000881] | 10/10 |

第二，这个优势主要出现在 1–6 h、尤其 3–6 h，并随未来风速轨迹变化程度增大；它不是只靠复制当前功率获得的短时 persistence 收益。但 interaction 没有改善逐场站 ramp CF MAE，因此不能声称它已经更准确地恢复了真实 ramp 动力学。

第三，interaction **没有通过 held-out-farm 零样本迁移检验**。对标签完全不参与训练和选模的场站，`Int−IAdd` 为：

| 年份 | IAdd | Int | Int−IAdd | 95% CI | Int 改善 folds |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.093733 | 0.095767 | +0.002034 | [−0.001279, +0.005347] | 3/8 |
| 2024 | 0.089580 | 0.091214 | +0.001634 | [−0.001256, +0.004523] | 4/8 |

因此最严格的总判断是：

> 历史功率与未来气象的非可分离联合预测关系，在接受过本站标签监督的固定风场集合中稳定存在，但当前完全共享、无场站校准的 concat interaction 不能稳定迁移到标签未见风场。现有结果不支持把它表述成跨风场普适的因果物理状态。

---

## 2. 科学问题与解释边界

本轮主问题是：

\[
F(H,W)\stackrel{?}{=}F_H(H)+F_W(W),
\]

其中：

- \(H\) 是本场近期实际功率轨迹，称为 **realized local generation condition**；
- \(W\) 是插值到本场位置的未来 Pangu 气象，称为 **future exogenous meteorological forcing**；
- `HW-IAdd` 显式限制为独立 history/weather 主效应之和；
- `HW-Int` 允许两种信息联合进入同一个 concat MLP。

实验可以识别：

\[
F(H,W)\neq F_H(H)+F_W(W)
\]

是否具有额外预测价值，但不能仅凭功率和 Pangu 数据识别以下因果机制：

- 风机可利用率；
- 限电或控制策略；
- 边界层稳定度；
- 尾流状态；
- 一个真实改变功率曲线的 turbine operating state。

因此报告使用“history-conditioned forcing response”或“预测非可分离性”，不把网络中的 history representation 强称为已辨识物理状态。

---

## 3. 实验设计

### 3.1 数据与冻结顺序

- 56 个风场；
- 2022：训练；
- 2023：验证、early stopping 和 H-Strong 窗口选择；
- 2024：冻结测试；
- 正式全量实验 seeds 42–51；
- held-out 实验 seeds 42、46、50；
- 主天气表示为局地 Bilinear Pangu；
- 不使用 station embedding、adapter、GNN 或按结果筛选场站。

2024 不参与模型结构、history 窗口、checkpoint 或 regime 分箱边界的选择。Regime 的非固定边界全部由 2022 训练集冻结。

### 3.2 Baseline 定义

| 模型 | 历史信息 | 未来 Pangu | 融合约束 |
|---|---|---|---|
| `P` | 当前 CF | 无 | persistence |
| `H-Matched` | 当前锚点 + 3 h 历史 | 无 | history-only |
| `H-Strong` | 1/3/6 h 中由 2023 选择 | 无 | history-only |
| `W` | 仅当前 CF 锚点 | 有 | 对更早历史严格不变 |
| `HW-None` | 3 h trajectory baseline | 有 | 无额外 history response branch |
| `HW-SAdd` | 3 h baseline + history state | 有 | 共享函数的 additive 主效应 |
| `HW-IAdd` | 3 h baseline + history state | 有 | 独立双分支 additive；与 Int 等参数量 |
| `HW-Int` | 3 h baseline + history state | 有 | faithful concat interaction |

所有神经模型保留相同骨架，总参数量为 542,893；`HW-IAdd` 与 `HW-Int` 的 response 模块参数量严格匹配。`history_only` 对天气输入严格不变，`W` 对 forecast origin 之前的功率历史严格不变。

### 3.3 统计口径

- 主指标：对 56 场等权的 macro station CF MAE；
- 辅助指标：macro CF RMSE、station ramp CF MAE、分时效 MAE、物理求和 cluster MAE；
- 全量比较：相同 seeds 的配对差和配对 t 区间；
- held-out：每折先平均 3 seeds，再对 8 folds 做配对区间；
- regime：10,000 次共享 UTC-day block bootstrap，报告 equal-farm estimand。

Regime 分析包含多个条件分箱，未做 family-wise 多重比较校正，应视为机制定位和假设生成证据，而不是新的独立确认性检验。

---

## 4. 完整 Baseline 结果

### 4.1 2023 validation

| 模型 | Station MAE CF | Station RMSE CF | Station Ramp MAE CF | 0–1 h | 1–3 h | 3–6 h | 物理求和 MAE MW |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 0.117542 | 0.182200 | **0.031039** | 0.052309 | 0.103207 | 0.148743 | 468.058 |
| H-Matched | 0.111785 | 0.170022 | 0.031100 | 0.051529 | 0.099858 | 0.139631 | 453.867 |
| H-Strong | 0.111707 | 0.169898 | 0.031113 | 0.051506 | 0.099765 | 0.139571 | 453.492 |
| W | 0.090045 | 0.140713 | 0.031417 | 0.049741 | 0.086379 | 0.105770 | 283.617 |
| HW-None | 0.089622 | 0.140328 | 0.031314 | 0.049233 | 0.085494 | 0.105703 | 288.577 |
| HW-SAdd | 0.089402 | 0.139361 | 0.031247 | 0.049317 | 0.085610 | 0.105144 | 288.220 |
| HW-IAdd | 0.089537 | 0.139829 | 0.031191 | 0.049302 | 0.085543 | 0.105470 | 289.874 |
| **HW-Int** | **0.087629** | **0.137238** | 0.031270 | **0.049017** | **0.084441** | **0.102478** | **278.692** |

### 4.2 2024 frozen test

| 模型 | Station MAE CF | Station RMSE CF | Station Ramp MAE CF | 0–1 h | 1–3 h | 3–6 h | 物理求和 MAE MW |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 0.112763 | 0.175619 | **0.028624** | 0.048741 | 0.098127 | 0.143763 | 452.319 |
| H-Matched | 0.106924 | 0.163759 | 0.028697 | 0.048119 | 0.094911 | 0.134380 | 437.459 |
| H-Strong | 0.106843 | 0.163624 | 0.028709 | 0.048088 | 0.094835 | 0.134304 | 436.664 |
| W | 0.086943 | 0.136320 | 0.029102 | 0.046755 | 0.082880 | 0.102934 | 282.254 |
| HW-None | 0.086265 | 0.135651 | 0.028989 | 0.046151 | 0.081765 | 0.102527 | 288.837 |
| HW-SAdd | 0.085978 | 0.134626 | 0.028914 | 0.046264 | 0.081830 | 0.101873 | 287.041 |
| HW-IAdd | 0.086089 | 0.135034 | 0.028862 | 0.046230 | 0.081736 | 0.102170 | 288.998 |
| **HW-Int** | **0.084760** | **0.133092** | 0.028944 | **0.046028** | **0.081109** | **0.099996** | **273.781** |

`HW-Int` 是全量固定场站预测的最佳模型：相对 persistence 的 MAE skill 为 25.45%（2023）和 24.83%（2024）。它同时取得最优 station MAE、station RMSE 和未经 cluster fine-tuning 的物理求和 MAE。

---

## 5. 两类数据分别提供什么信息

### 5.1 历史功率单独有效，但贡献有限

`H-Strong` 相对 persistence：

- 2023：−0.005836 CF，约 4.97% 相对改善；
- 2024：−0.005921 CF，约 5.25% 相对改善；
- 两年均 10/10 seeds 改善，配对区间完全低于 0。

H-Strong 根据 2023 三个 screening seeds 选择了 24 个 15-min steps，即 6 h 历史。但它相对 3 h `H-Matched` 只改善：

- 2023：−0.000079 CF；
- 2024：−0.000082 CF。

这个差异虽在 seeds 上稳定，但绝对量很小。因此不能把“6 h 最优”解释为明确物理记忆尺度；更稳妥的结论是，3–6 h 历史单独提供了有限但稳定的时序信息，扩展到 6 h 的边际收益很小。

### 5.2 未来气象是主要信息源

`W` 相对 persistence：

- 2023：−0.027497 CF；
- 2024：−0.025820 CF；
- 两年均 10/10 seeds 改善。

这远大于 history-only 的约 −0.0059 CF，说明 0–6 h 任务的主体可预测性来自未来外部气象 forcing，而不是历史功率延续。

### 5.3 更早历史对 weather model 仍有小幅独立增量

`HW-None−W`：

- 2023：−0.000423 CF，95% CI [−0.000545, −0.000300]；
- 2024：−0.000678 CF，95% CI [−0.001248, −0.000109]。

因此在相同 weather forcing 下，forecast origin 之前的本场功率轨迹仍提供小而稳定的信息。但 `HW-IAdd−HW-None` 的区间两年均跨 0，说明再增加一个独立 history main-effect branch 并没有可靠增量：

- 2023：−0.000086 [−0.000193, +0.000022]；
- 2024：−0.000176 [−0.000654, +0.000301]。

换言之，历史轨迹的大部分可分离主效应已经被 trajectory baseline 吸收；新的显著收益来自 history 与 weather 的联合表示，而不是简单再加一次 history-only correction。

---

## 6. Interaction 的严格归因

### 6.1 相对两个 Additive controls 均成立

| 对比（Int−control） | 2023 | 2024 |
|---|---:|---:|
| vs HW-SAdd | −0.001773 [−0.001929, −0.001617], 10/10 | −0.001218 [−0.001688, −0.000748], 10/10 |
| vs HW-IAdd | **−0.001907 [−0.002059, −0.001756], 10/10** | **−0.001329 [−0.001777, −0.000881], 10/10** |
| vs HW-None | −0.001993 [−0.002138, −0.001849], 10/10 | −0.001505 [−0.001909, −0.001100], 10/10 |

由于 `HW-IAdd` 的 history/weather 分支彼此独立，并与 `HW-Int` 等参数量，结果不能仅用“Int 参数更多”或“Add 共享函数过强”解释。在参与监督的场站上，数据确实要求某种非可分离联合预测项。

### 6.2 RMSE 与场站覆盖

相对 `HW-IAdd`，station RMSE CF：

- 2023：−0.002591 [−0.003281, −0.001901]，10/10 seeds；
- 2024：−0.001941 [−0.002908, −0.000975]，10/10 seeds。

按每场先跨 seeds 平均后比较，Interaction 改善：

- 2023：44/56 场；
- 2024：43/56 场。

因此全量结果不是由极少数大场站驱动，但也不是所有场站都获益。不能通过事后删除不获益场站来增强主结论。

### 6.3 Interaction 收益随时效增强

`HW-Int−HW-IAdd` 的 station MAE CF：

| 时效 | 2023 | 2024 |
|---|---:|---:|
| 0–1 h | −0.000285 [−0.000411, −0.000159] | −0.000202 [−0.000379, −0.000025] |
| 1–3 h | −0.001102 [−0.001339, −0.000866] | −0.000627 [−0.001051, −0.000202] |
| 3–6 h | **−0.002992 [−0.003330, −0.002654]** | **−0.002174 [−0.002882, −0.001466]** |

收益不是集中于 persistence 最强的 0–1 h，而是随 weather-driven horizon 墑长，在 3–6 h 最大。这支持“history 帮助解释未来 forcing”的预测现象，而不是“模型只是重复最近功率”。

### 6.4 但 station ramp 没有改善

`HW-Int−HW-IAdd` 的 station ramp CF MAE：

- 2023：+0.000079 [ +0.000002, +0.000155 ]，仅 2/10 seeds 更好；
- 2024：+0.000082 [−0.000001, +0.000165]，仅 2/10 seeds 更好。

因此 Interaction 改善的是未来功率水平/轨迹的绝对误差，不是相邻 15-min 增量误差。不能把 horizon 结果进一步扩张成“更好识别了 ramp momentum”。

---

## 7. Interaction 在什么条件下有用

Regime gain 定义为：

\[
G=|e_{\rm IAdd}|-|e_{\rm Int}|,
\]

正值表示 Interaction 更好。这里使用 10-seed ensemble，因此 gain 数值不能直接与单 seed 配对均值作线性比较。

### 7.1 当前功率水平

| 当前 CF | 2023 gain | 2024 gain |
|---|---:|---:|
| 0–0.2 | +0.002355 | +0.002140 |
| 0.2–0.4 | +0.001730 | +0.001768 |
| 0.4–0.6 | +0.002671 | +0.001741 |
| 0.6–0.8 | +0.005908 | +0.002299 |
| 0.8–1.0 | **+0.010260** | **+0.006921** |

所有区间的 day-block CI 均大于 0，高 CF 尤其是 0.8–1.0 区间收益最大。数据说明 full-data 模型在高发电状态下更需要联合解释，但不能据此判定原因是饱和功率曲线、限功率还是 availability，因为缺少对应观测。

### 7.2 历史 ramp

按 signed slope 或绝对 slope 分箱后，所有区间均有正 gain，但跨年不存在清晰单调关系：

- 2023 的最大绝对 slope 分位收益较强（+0.003512）；
- 2024 各分位约 +0.001975 至 +0.002429，最大 ramp 不再明显占优；
- signed slope 的 falling/stable/rising 排序也不跨年稳定。

因此不能写成“Interaction 主要由历史 ramp 驱动”。历史 ramp 不是 interaction 存在的必要 regime。

### 7.3 未来气象变化

未来约 6 h 的 Bilinear 10 m 风速变化呈现更稳定的结构：

- 近似稳定风速区间的 gain 最小，约 +0.0014 至 +0.0015；
- 最大正风速变化分位：2023 +0.005849，2024 +0.003729；
- 未来风速时间标准差从最低到最高分位，gain 基本单调增加：
  - 2023：+0.001294 → +0.004547；
  - 2024：+0.001472 → +0.003197。

二维 ramp × future-wind-change map 也显示：未来风速上升分位通常比稳定分位更依赖 Interaction，尤其在历史 falling/stable 时跨年一致。历史 rising × future rising 在 2024 没有保持同等强度。

所以目前最稳妥的 regime 结论是：

> Interaction 的价值更清楚地随未来 forcing 的非平稳性增强，而不是由某一种历史 ramp 方向唯一决定。

这仍然是条件预测关系，不足以识别其背后的具体物理变量。

---

## 8. Held-out-farm 泛化实验

### 8.1 它如何避免“56 场都训练过”的混淆

全量主实验与 held-out 实验是两套独立训练。Held-out 部分将 56 场固定分为 8 folds，每折 7 场；每次重新训练：

\[
49\text{ 场监督训练和选模}+7\text{ 场标签完全留出}.
\]

每折训练 `HW-IAdd` 和 `HW-Int`，各 3 seeds，共 48 个模型。留出场站：

- 不进入 2022 station loss；
- 不进入 2023 checkpoint metric；
- 不进入 cluster loss（cluster loss 固定为 0）；
- 不使用 station embedding；
- checkpoint 冻结后才计算其 2023/2024 指标。

张量为保持模型形状仍计算 56 场输出，但被留出 7 场的目标项不产生梯度。模型允许使用推理时本来可知的新场站历史功率、Pangu、位置和容量，因此这是 **label-held-out zero-shot transfer**，不是“连新场站元数据也不可见”的测试。

### 8.2 结果

2023：

- IAdd：0.093733 CF；
- Int：0.095767 CF；
- Int−IAdd：+0.002034，95% CI [−0.001279, +0.005347]；
- 仅 3/8 folds 改善。

2024：

- IAdd：0.089580 CF；
- Int：0.091214 CF；
- Int−IAdd：+0.001634，95% CI [−0.001256, +0.004523]；
- 仅 4/8 folds 改善。

分时效同样没有隐藏的正迁移信号。尤其 0–1 h：

- 2023：+0.000793 [ +0.000027, +0.001559 ]，仅 1/8 folds 改善；
- 2024：+0.000670 [ +0.000164, +0.001175 ]，仅 1/8 folds 改善。

1–3 h 和 3–6 h 的区间均跨 0。预注册的“CI 上界小于 0 且至少 6/8 folds 改善”两年均失败。

### 8.3 含义

这否定的是：

> 一个完全共享、无需目标场站标签校准的 history–weather interaction law 可以稳定迁移到任意新风场。

它不否定：

> 对于有本站历史监督的固定运行风场，联合 history–weather 模型优于可分离模型。

可能原因包括场站响应异质性、功率历史所代理的未观测因素具有场站特异性，或 concat 模型需要本站标签才能确定如何解释 history。现有实验不能区分这些解释，因此不据此引入 station embedding 或作因果结论。

---

## 9. 物理求和与 cluster 尺度边界

Station-only checkpoint 直接物理求和时，Interaction 相对 IAdd 仍然显著更好：

- 2023：289.874 → **278.692 MW**，差 −11.182 MW [−17.449, −4.914]，10/10 seeds；
- 2024：288.998 → **273.781 MW**，差 −15.217 MW [−23.623, −6.811]，10/10 seeds。

所以 local interaction 的收益不是在求和操作中自然消失。

但使用 `station loss + 2 × cluster loss` 做 aggregate-aware fine-tuning 后，模型排序不跨年稳定：

| 年份 | IAdd transfer | Int transfer | Int−IAdd |
|---|---:|---:|---:|
| 2023 | **263.861 MW** | 266.900 MW | +3.039 [+0.485, +5.594] |
| 2024 | 263.816 MW | **259.428 MW** | −4.388 [−6.570, −2.207] |

2023 中 IAdd 从 station-only 求和到 cluster fine-tuning 改善约 26.013 MW，而 Int 改善约 11.792 MW；2024 分别约 25.182 MW 与 14.353 MW。Additive 模型对 aggregate objective 有更大的再优化空间，导致 2023 排名逆转，而 2024 中 Int 的初始优势仍被保留。

因此：

\[
\text{farm-level 最优表示}\neq\text{aggregate-target 最优误差结构}.
\]

该结果适合作为尺度边界扩展，而不适合用 cluster fine-tuning 重新定义局地 interaction 是否存在。

---

## 10. 最终路线判断

### 10.1 如果目标是固定的 56 场运营预测

可以保留 farm-level `HW-Int` 作为主模型。它在：

- 两个年份；
- 10/10 seeds；
- 两个 additive controls；
- MAE 与 RMSE；
- 多数场站；
- 物理求和；
- 尤其 3–6 h；

均获得一致优势。最终区域输出可以继续由逐场预测物理求和形成。

### 10.2 如果论文要声称跨场站普适机制

当前证据不够。Held-out 结果明确失败，因此不能写：

> 历史发电状态对天气响应的调制规律可以无校准迁移到新风场。

论文主张必须收缩为：

> 在具有本站监督的多风场联合学习中，历史功率与未来气象存在稳定的预测非可分离性；其跨场零样本迁移受到明显场站异质性限制。

### 10.3 是否需要直接放弃这条路线

不需要直接放弃，但需要明确应用边界：

- 固定已知场站：Interaction 路线成立且值得使用；
- 新场站零样本：当前 IAdd 更安全，Interaction 不应作为默认；
- 若继续研究新场站，应研究少量本站校准、分层共享或基于物理属性的异质性，而不是先声称一个完全共享的普适 modulation law；
- 不按结果筛掉不支持结论的场站。

当前最有价值、且完全由数据支持的科学表述是：

> 对已观测风场，相同的未来局地气象 forcing 不能脱离已经实现的本场发电历史条件被最优解释；这种联合预测价值在长时效、高发电水平和未来风速变化较强时更明显，但其形式具有场站依赖性，尚不能零样本迁移为统一的跨场物理规律。

---

## 11. 结果文件

- `artifacts/farm_level_history_weather_v1/analysis/summary.json`
- `artifacts/farm_level_history_weather_v1/analysis/baseline_seed_metrics.csv`
- `artifacts/farm_level_history_weather_v1/analysis/baseline_per_farm.csv`
- `artifacts/farm_level_history_weather_v1/analysis/baseline_per_lead.csv`
- `artifacts/farm_level_history_weather_v1/analysis/heldout_fold_metrics.csv`
- `artifacts/farm_level_history_weather_v1/analysis/interaction_regimes.csv`
- `artifacts/farm_level_history_weather_v1/analysis/cluster_finetune_trajectory.csv`
- `artifacts/farm_level_history_weather_v1/analysis/figures/baseline_station_mae.png`
- `artifacts/farm_level_history_weather_v1/analysis/figures/regime_ramp_wind_val.png`
- `artifacts/farm_level_history_weather_v1/analysis/figures/regime_ramp_wind_test.png`
- `artifacts/farm_level_history_weather_v1/analysis/figures/cluster_finetune_trajectory.png`
- `artifacts/farm_level_history_weather_v1/selection/fold_manifest.json`
- `artifacts/farm_level_history_weather_v1/selection/history_validation_freeze.json`
- `artifacts/farm_level_history_weather_v1/selection/regime_protocol.json`
- `artifacts/farm_level_history_weather_v1/COMPLETE`
