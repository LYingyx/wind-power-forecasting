# 历史状态可辨识性 v2：完整结果、关键结论与路线决策

状态：**全部实验完成**。本轮以 2022 为训练集、2023 validation 为唯一结构选择集；2024 只在 2023 冻结后统一评估。不过，2024 已在更早研究中被查看过，因此本文只称其为“冻结后的跨年检查”，不称为纯盲外部测试。

## 1. 完成性审计

- 顶层 `artifacts/local_state_identifiability_v2/COMPLETE` 存在，launcher 已正常退出。
- 训练 `COMPLETE` 标记：141 个。
- 正式 `TEST_COMPLETE` 标记：80 个。
- validation / test 指标文件：250 / 190 个。
- 全部日志中 `Traceback`、OOM、`Killed`、`RuntimeError`、`ValueError` 命中数为 0。
- 正式模型均完成 seeds 42–51 的 10-seed 配对评估；干预与多尺度分析也已完成。

因此，下面的结论不是运行中间态。

## 2. 本轮到底比较了什么

所有正式模型均采用同一个 shared local-to-global 主干，而不是 56 个独立模型：

1. 56 个场站的未来 Pangu 格点天气按场站坐标做固定双线性插值；
2. 所有场站共享 weather encoder、历史 encoder、时间查询与 decoder；
3. 各场站输出 capacity factor 轨迹，再按装机容量固定求和得到集群总功率；
4. baseline 使用各场站最近 3 h、共 12 个 15 min 历史点，并以当前场站功率为 logit anchor；
5. 主损失为 cluster loss，辅助使用 station CF loss，比例等价于
   \[
   \mathcal L=\mathcal L_{\rm cluster}+0.5\mathcal L_{\rm station\,CF};
   \]
6. 正式控制和候选具有相同的 checkpoint 参数量 530,029，训练预算、seed 和数据完全配对。

正式模型定义如下。

| 简称 | response state / context | 状态如何进入天气响应 |
|---|---|---|
| **B0** | 无状态、无场站 context | 无 interaction |
| **B-Static** | 仅标准化的 latitude、longitude、log-capacity | strict zero-invariant interaction |
| **G-Static** | 最近 4 个集群 capacity-factor 点编码出的 aggregate state，加 static context | strict zero-invariant interaction |
| **GL-Static** | aggregate state 加“局地功率减集群功率”的 local-deviation state，再加 static context | strict zero-invariant interaction |
| **P-Direct** | 最近 4 个局地功率点，加 static context | unconstrained direct state path；仅作表达能力诊断 |

最终选择的 **G-Static** 可简写为：

\[
s_g=E_g\!\left(p_{\Sigma,t-3:t}\right),\qquad
e_i=E_{\rm static}(\operatorname{lat}_i,\operatorname{lon}_i,\log C_i),
\]

\[
h'_{i,q}=h^w_{i,q}+0.25\left[M(h^w_{i,q},s_g+e_i)-M(0,s_g+e_i)\right].
\]

因此

\[
h^w_{i,q}=0\Longrightarrow h'_{i,q}=0,
\]

状态和静态 context 不能在零天气 latent 下独自产生 weather correction。所有场站共享这个 interaction，最终

\[
\hat P_q=\sum_{i=1}^{56}C_i\hat p_{i,q}.
\]

正式诊断中 strict 模型的最大零天气不变误差为 0 CF。

需要特别说明：aggregate state 可由原始场站历史求和得到，因此它不是新增外部信息。本轮证明的是“把共同模态历史显式编码为天气响应条件”具有有效的结构归纳偏置，而不是发现了新的输入变量。

## 3. 10-seed 正式结果

### 3.1 2023 validation：用于冻结模型

| Model | MAE / MW | RMSE / MW | R² | Ramp / MW | 0–1 h | 1–3 h | 3–6 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 270.431 ± 2.927 | 395.760 ± 5.195 | 0.901 ± 0.003 | 58.972 ± 0.403 | 109.490 ± 0.825 | 246.684 ± 2.458 | 339.910 ± 4.187 |
| B-Static | 270.162 ± 2.879 | 394.793 ± 4.929 | 0.901 ± 0.002 | 58.922 ± 0.517 | **109.560 ± 0.961** | 246.975 ± 2.632 | 339.153 ± 4.164 |
| **G-Static** | **267.985 ± 3.067** | **391.665 ± 5.568** | **0.903 ± 0.003** | **58.623 ± 0.472** | 110.235 ± 1.227 | **246.276 ± 2.490** | **335.041 ± 4.779** |
| GL-Static | 268.548 ± 3.081 | 391.692 ± 5.873 | 0.903 ± 0.003 | 58.665 ± 0.564 | 110.576 ± 1.366 | 247.619 ± 2.418 | 335.159 ± 5.132 |
| P-Direct | 269.416 ± 3.175 | 392.977 ± 5.915 | 0.902 ± 0.003 | 58.914 ± 0.548 | 110.610 ± 1.180 | 247.450 ± 2.473 | 336.996 ± 5.096 |

按预先规定的 2023 mean MAE，冻结模型为 **G-Static**。

### 3.2 冻结后的 2024 跨年检查

| Model | MAE / MW | RMSE / MW | R² | Ramp / MW | 0–1 h | 1–3 h | 3–6 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 267.261 ± 1.862 | 393.478 ± 3.664 | 0.893 ± 0.002 | 56.514 ± 0.334 | **104.424 ± 0.620** | 240.452 ± 1.804 | 339.413 ± 2.533 |
| B-Static | 266.804 ± 1.771 | 392.336 ± 2.838 | 0.894 ± 0.002 | 56.448 ± 0.393 | 104.412 ± 0.607 | 240.668 ± 1.635 | 338.358 ± 2.749 |
| **G-Static** | 264.817 ± 1.919 | 389.630 ± 3.469 | 0.895 ± 0.002 | **56.185 ± 0.315** | 105.034 ± 1.144 | 240.252 ± 1.931 | 334.455 ± 2.612 |
| GL-Static | 265.198 ± 3.306 | 389.624 ± 4.940 | 0.895 ± 0.003 | 56.228 ± 0.458 | 105.402 ± 1.211 | 241.544 ± 3.003 | 334.232 ± 4.924 |
| P-Direct | **264.525 ± 3.220** | **387.690 ± 5.942** | **0.896 ± 0.003** | 56.415 ± 0.460 | 105.377 ± 0.896 | **240.026 ± 2.152** | **333.908 ± 5.153** |

P-Direct 的 2024 数值最低，但不能据此反向改选：它在 2023 不是最佳模型、允许 state-only shortcut，而且没有通过正式 local-correspondence gate。

## 4. 最关键的配对比较

下表中的差值均为左侧减右侧，负值代表左侧更好；CI 为跨 seed 的配对 95% CI。

| Split | 对比 | ΔMAE / MW | 95% CI / MW | wins |
|---|---|---:|---:|---:|
| 2023 | G-Static − B-Static | **−2.177** | **[−3.150, −1.203]** | **10/10** |
| 2024 | G-Static − B-Static | **−1.986** | **[−3.181, −0.791]** | 8/10 |
| 2023 | G-Static − B0 | **−2.446** | **[−3.104, −1.788]** | **10/10** |
| 2024 | G-Static − B0 | **−2.444** | **[−2.930, −1.958]** | **10/10** |
| 2023 | G-Static − GL-Static | **−0.563** | **[−0.868, −0.259]** | **10/10** |
| 2024 | G-Static − GL-Static | −0.381 | [−2.171, +1.410] | 4/10 |
| 2023 | G-Static − old aggregate A0-FT | **−5.210** | **[−7.844, −2.575]** | 8/10 |
| 2024 | G-Static − old aggregate A0-FT | **−5.508** | **[−8.896, −2.120]** | 8/10 |
| 2023 | G-Static − previous strict G-Int | +0.167 | [−0.460, +0.794] | 5/10 |
| 2024 | G-Static − previous strict G-Int | −0.434 | [−1.192, +0.323] | 6/10 |

G-Static 相对 B-Static 的 2023 Holm 校正后 \(p=0.00205\)。这是本轮用于判断 aggregate state 增量的最干净比较。

相对旧 A0-FT 的 5.210 / 5.508 MW 总改善包含 local-to-global 天气与监督结构变化，不能全部归因于 state。严格的 state 增量应使用 G-Static 对 B-Static，即 2.177 / 1.986 MW。

G-Static 与上一轮 G-Int 在两个年份都统计相当。因此，本轮贡献不是刷新 numerical SOTA，而是用更强干预回答“模型是否依赖正确的样本—状态对应”。

从时效看，G-Static 相对 B-Static 的主要收益来自 3–6 h：2023 / 2024 分别改善 4.112 / 3.903 MW；0–1 h 反而恶化 0.675 / 0.622 MW。Ramp MAE 分别小幅改善 0.299 / 0.263 MW。部署若特别重视超短时效，应单独处理这一 trade-off，不能声称所有 horizon 都改善。

## 5. 强状态干预：终于证明“正确对应关系”重要

### 5.1 训练时 matched shuffled-state control

G-Static 与相同架构、相同 seed、但 response state 使用强 donor 错配训练的 G-Static-Shuf 相比：

\[
\Delta\operatorname{MAE}_{2023}=-2.228\ \mathrm{MW},
\quad 95\%\ \mathrm{CI}=[-3.135,-1.321],
\quad 10/10\ \mathrm{wins}.
\]

baseline 历史没有被错配，只有 response-state 输入被替换，因此这个对照不会把“破坏 baseline”误当作状态证据。

### 5.2 推理时强 donor 错配

donor 在 UTC month、origin phase 和 NWP age 内做固定无放回错排，且没有 fixed point。validation / 2024 donor 与原样本当前集群功率的相关系数仅为 0.100 / 0.115，显著强于旧的相邻 batch-roll 干预。

| Model | 干预 | 2023 degradation / MW | block-bootstrap 95% CI | 2024 degradation / MW | block-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| G-Static | strong sample donor | **+30.652** | **[28.786, 32.466]** | **+34.311** | **[32.133, 36.665]** |
| GL-Static | strong sample donor | +31.808 | [29.672, 34.080] | +37.617 | [35.110, 40.221] |
| GL-Static | far-station permutation | **+5.774** | **[4.804, 6.752]** | **+2.661** | **[1.603, 3.784]** |

旧 batch-roll 基本保留了强自相关状态，因而“roll 后几乎不变”不能推出模型没有使用状态。新结果同时具备 train-time shuffled control 和低相关 donor inference intervention，足以推翻旧的弱干预结论：**G-Static 确实依赖正确的逐样本 aggregate state。**

### 5.3 “被使用”不等于“有增量价值”

GL-Static 的 far-station permutation 会使 farm 和 K=8 normalized MAE 在 10/10 seeds 上都恶化，说明网络确实读取了正确的局地对应关系。然而：

\[
\operatorname{MAE}_{G\text{-Static}}
<
\operatorname{MAE}_{GL\text{-Static}}
\]

在 2023 为 10/10 seeds，均值差 0.563 MW 且 CI 完全小于 0；2024 也没有看到 GL 的稳定优势。因此正确表述是：

> GL-Static 使用了 local-deviation state，但该状态没有在 aggregate common-mode state 之上提供新增预测收益。

这不能写成“局地或层次状态路线成功”。

P-Direct 也会对强错配显著响应，且 2023 相对 B-Static 改善 0.745 MW，但 far-station 干预在 farm / K=8 的逐 seed 正向数只有 6/10 和 7/10，未达到预注册的 8/10 local gate；同时它允许状态直接产生 correction。因此它只能说明历史存在直接预测信息，不能支持“state-conditioned weather response”机制。

## 6. 3-seed screening：只能作为后续假设，不能写成正式发现

| Screening 单元 | 2023 ΔMAE / MW | wins | 判定 |
|---|---:|---:|---|
| Static lat/lon/log-capacity only | −0.275 | 1/3 | 失败；静态物理 context 本身不稳定 |
| Learned station ID only | **−2.673** | **3/3** | 有希望，但尚无 10-seed 正式确认 |
| Local P4 state + ID vs ID-only | +0.507 | 1/3 | local state 没有在 ID 上增加收益 |
| Local state + affine vs affine-only | −0.010 | 2/3 | 无可见增量 |
| Pure local P4 state | +0.325 | 1/3 | 失败 |
| Local P12 state | −0.340 | 2/3 | farm 0/3、shuffle 信号缺失，失败 |
| Hand-crafted physical state | −1.169 | 2/3 | 通过 screening，但未晋级；不能声称成功 |
| Plain historical IO-pair | +0.172 | 2/3 | 失败 |
| IO-pair + support→query ID loss | −0.982 | 2/3 | shuffle +0.553 MW、farm 3/3；仅为有希望的 screening 信号 |

这里有三个边界：

- **纯局地功率状态没有获得支持。** 增加历史长度、learned ID 或小型 affine calibration 都没有救活它。
- **场站身份可能是独立的 weather-power calibration。** ID-only 的 3-seed 结果很好，但 local state + ID 反而更差，所以不能把 ID 的收益记到 local state 名下。
- **物理状态和 IO-pair-ID 尚未达到正式证据等级。** 它们可以作为次级复现实验，但当前没有理由替代更简单、已经通过 10-seed 的 aggregate P4 state。

## 7. 多尺度证据与“为什么继续做集群预测”

下面的 state gain 定义为 matched no-state B-Static MAE 减 G-Static MAE，正值表示 aggregate state 有效。

| 聚合尺度 | 2023 state gain / MW | 2024 state gain / MW |
|---|---:|---:|
| 56 个单场 macro | +0.135 | +0.106 |
| K=8 子集群 | +0.190 | +0.075 |
| K=4 子集群 | +0.381 | +0.091 |
| K=2 子集群 | +1.094 | +0.084 |
| 完整 56 场集群 | **+2.177** | **+1.986** |

aggregate state 的稳定收益在完整集群尺度最清楚，而不是在单场尺度最清楚。这与“aggregate state 表示区域共同模态、系统性偏差或共同运行状态”的解释一致。

相反，GL-Static 虽对 local correspondence 非常敏感，但在 K=8 / K=4 / K=2 上的 state gain 弱或为负，且完整集群上仍不如 G-Static。局地状态被读取后可能增加局部分解约束或方差，却没有转化成更好的总功率预测。

因此主任务应继续使用全部 56 场的集群预测，理由是：

1. 56/56 场均通过仅使用 2022 train 定义的数据质量门槛；没有预先定义的客观剔除依据；
2. 最强证据本身出现在完整集群尺度；
3. 根据 2023/2024 效果挑“支持结论”的场站会构成 outcome-dependent selection，夸大机制效应；
4. farm、K=8/4/2 应保留为预声明的尺度诊断，而不是事后重构主任务。

如果今后另写单场机制研究，应预先按地理、容量、机型或 train-only 数据质量分层，或完整报告 56 场；不能根据 validation/test state gain 选择几个风场。

## 8. 对历史状态路线的最终判断

### 已经得到支持

\[
\boxed{\text{最近 aggregate/common-mode power state 对集群天气响应具有稳定增量价值。}}
\]

证据是：G-Static 相对 matched no-state control 跨年份改善、2023 配对 CI 不跨 0、10/10 获胜、matched train-shuffle 失败、强 inference donor 造成约 31–34 MW 退化，并严格满足 zero-weather invariance。

### 没有得到支持

\[
\boxed{\text{纯 local power-history state 或 hierarchical local deviation 能进一步提高集群预测。}}
\]

局地状态“被模型使用”成立，但“在 aggregate state 之上有增量”不成立。二者不能混为一谈。

### 物理解释应保持克制

G-Static 的状态可能同时编码：

- 区域共同 ramp / 功率水平；
- Pangu 与实际区域功率的短期共同偏差；
- 区域性限电、可用率或未观测天气造成的 common-mode persistence。

现有输入不足以把这些因素拆开，所以当前最严谨名称是 **recent aggregate/common-mode response state**，而不是已被物理辨识的“真实机组运行状态”或“因果 transfer law”。

## 9. 是否加入 station embedding，以及下一步最干净的实验

现在不应把 station embedding 直接塞入 local-state 模型并把任何收益解释为状态收益。现有 3-seed 结果表明 ID-only 可能有效，而 `local state + ID` 没有超过 ID-only；二者很可能是不同机制。

如果继续一轮，唯一优先的正式实验应是 10-seed 的 2×2 因子矩阵：

| | No learned station ID | Learned station ID |
|---|---|---|
| No aggregate state | no-ID baseline | ID-only |
| Aggregate P4 state | G-only | G+ID |

要求：去掉会混淆比较的 static context 或在四格中完全一致；严格配对 seed、预算、参数规模和训练时 strong shuffled-state controls；对 G-only / G+ID 都做低相关 donor intervention。这个矩阵分别回答：

1. station identity 是否只是稳定的局地 weather-power curve calibration；
2. aggregate state 在给定 station identity 后是否仍有增量；
3. 两者是可加、冗余还是交互。

只有 G+ID 同时稳定超过 ID-only 和 G-only，才能把 station embedding 纳入最终模型。否则保留更简单的 G-Static。

IO-pair + identification loss 可作为第二优先级，因为它有初步信号且更接近系统辨识叙事；但它目前只有 3 seeds，数值和证据强度都未超过简单 aggregate P4 state，不能成为当前主线。

## 10. 最终路线决策

本轮自动分析中的 `continue_strict_local_or_hierarchical_state` 只表示 GL-Static 通过了其相对 B-Static 的预注册 gate；它没有包含 G-Static 与 GL-Static 的增量比较，因此不能作为最终科学结论。

修正后的路线决策为：

\[
\boxed{
\begin{aligned}
&\text{保留全部 56 场的 cluster forecasting 主任务；}\\
&\text{保留 recent aggregate/common-mode state；}\\
&\text{停止把纯 local power-history state 作为当前主线；}\\
&\text{不按结果筛选场站；}\\
&\text{station ID 先按独立校准因素做正式 2×2 验证。}
\end{aligned}}
\]

当前正式推荐模型是 **G-Static**：2023 validation `267.985 ± 3.067 MW`，冻结后的 2024 `264.817 ± 1.919 MW`。它是本轮最佳、证据链最完整的模型；但与上一轮 strict G-Int 统计相当，因此应把论文贡献放在“正确样本—aggregate state 对应的严格可辨识证据”和“状态收益随聚合尺度增强”，而不是宣称一个新的数值 SOTA。

## 11. 限制与证据文件

- 2024 对本轮选择是冻结后的，但不是整个研究过程中的 pristine blind test。
- learned ID、hand-crafted physical state、IO-pair-ID 目前都只是 3-seed screening，不能进入正式结论。
- 强错配证明模型依赖状态对应，并不单独证明该 representation 等同于某个唯一物理状态。
- 本轮来自单一区域和既定年份，跨区域、跨装机结构泛化仍需外部数据验证。

机器可读汇总：`artifacts/local_state_identifiability_v2/analysis/summary.json`。

强 donor 数据审计：`data/local_state_identifiability_v2/manifest.json`。

冻结记录：`artifacts/local_state_identifiability_v2/selection/validation_freeze.json`。

实验预注册：`exp_docs/local_state_identifiability_v2_plan.md`。

## 12. 旧 4×4 逐场模型与当前双线性模型为何得到不同的“状态结论”

本节不使用 `main.pdf`，只审计仓库中的旧模型实现、实验记录和当前实现。最重要的修正是：旧模型和当前模型都不是“直接预测一个集群总功率”的纯 aggregate 网络。二者都先产生 56 个场站的功率轨迹，再按装机容量严格求和：

\[
\hat P_q=\sum_{i=1}^{56}C_i\hat p_{i,q}.
\]

因此，当前纯 local state 没有新增收益，不能解释为“现在才改成了局地气象到场站功率的映射”。旧模型本身也是局地天气表示到场站功率、最后求和。真正改变的是气象支持、历史通路、状态约束和所估计的科学量。

| 比较项 | 旧 `forced_evolution/state_conditioned` | 旧 frozen-B0 state ladder | 当前 local-state identifiability v2 |
|---|---|---|---|
| 场站天气 | 周围 4×4 原生格点经内容、相对位置和风向几何学习权重形成代理天气 | 同一类 4×4 learned proxy | 14 个原始气象变量先按场站坐标固定双线性插值，再共享编码 |
| 场站输出 | 逐场输出，再容量加权求和 | 逐场 correction，再容量加权求和 | 逐场 baseline + weather correction，再容量加权求和 |
| 历史 baseline | local/global history 与天气在一个联合网络中直接融合；当前场站功率还是输出 anchor | 完整历史由冻结 B0 使用 | 每场最近 3 h/12 点由强 baseline 使用；当前场站功率是 logit anchor |
| correction state | local history + global history + target time | `recent_1h` 同时向 correction 暴露 local 与 aggregate 最近四点 | 严格拆分 aggregate-only、local-only 和 aggregate+local-deviation |
| 状态路径 | FiLM 含 additive `beta`，且 local/global history 再次直接进入 decoder | 默认 black-box concat，history 可直接形成 correction | strict 模型只允许状态改变 weather latent，禁止 state-only correction |
| 零天气性质 | 不满足；`beta(s)` 和直接 history 路径仍可输出 | black-box ladder 不满足 | 满足 `I(0,s)=0` |
| 主要 estimand | history + weather 的联合预测是否优于单模态 | correction 看近期历史是否优于不看历史 | 强 local-history baseline 已存在后，response state 是否还有严格的增量 |

旧 `state_conditioned` 的实际计算为：

\[
\operatorname{forced}_{i,q}
=
[1+\tanh\gamma(H_i,H_\Sigma,\tau_q)]\odot h^w_{i,q}
+\beta(H_i,H_\Sigma,\tau_q),
\]

随后 decoder 仍直接读取 local history、global history、`forced`、二者乘积和 target token。即使令天气 latent 为零，`beta` 和直接历史通路仍可产生预测。因此旧结果 `265.738 MW` 证明的是历史与天气联合建模有效，不能单独归因于“历史状态调制 weather→power mapping”。同一批实验中的 prediction weighting 为 `265.367 MW`，bidirectional cross-attention 为 `265.669 MW`，也说明 state-conditioned operator 并未被证明优于所有非状态调制的联合融合。

旧 10-seed state ladder 的证据更强，但它也不能被解读为纯 local modulation。`recent_1h` 在实现中同时返回最近四点场站历史和最近四点集群历史。aggregate recent 1h 为 `267.214 MW`，station recent 1h 为 `265.759 MW`；在相同时间窗下，station-resolved trajectory 的真实边际仅为：

\[
1.455\ \mathrm{MW},
\qquad 95\%\mathrm{CI}=[0.434,2.476],
\qquad 9/10\ \text{seeds}.
\]

旧实验从 weather-only 到 station-recent 的大幅改善不能全部记为 local state 收益，它混合了 aggregate operating level、aggregate trajectory、local persistence、允许的 state-only correction 以及 state-weather interaction。当前实验把这些路径拆开以后，得到的结论是：

1. 历史功率当然有用，它已经被每场 baseline 和当前功率 anchor 大量使用；
2. 正确的 aggregate/common-mode response state 在这个强 baseline 之上仍有约 `2 MW` 的稳定增量；
3. local-deviation state 会被网络读取，far-station permutation 会明显破坏结果，但它没有在 aggregate state 之上提高正常预测；
4. 因而当前否定的不是“场站历史功率有预测价值”，而是更窄的命题：**逐场历史在强 local-history baseline 和正确 aggregate state 之外，还能稳定提高严格的 local weather-response modulation。**

### 12.1 是否可能由 4×4 代理天气改成双线性点天气造成

这是一个合理的候选原因，但现有证据不能确认。理论上存在两个方向相反的解释：

- 4×4 支持可能保留天气梯度、上游 forcing 和移动结构，使 local state 有更多可以条件化的信息；
- learned 4×4 proxy 也可能更模糊或有系统偏差，历史功率只是帮助校准 proxy error，从而产生“状态有用”的表象，而非辨识出真实 transfer state。

已有空间 same-support 筛选为：raw 2×2 bilinear `257.826 MW`、latent 2×2 bilinear `257.716 MW`、fixed 4×4 Gaussian `257.763 MW`、learned 4×4 without wind geometry `258.728 MW`、learned 4×4 with wind geometry `256.589 MW`。其中 fixed 4×4 相对 latent bilinear 仅差 `+0.047 MW`，learned 4×4 without geometry 反而比同支持 fixed 4×4 差 `0.965 MW`。带 wind geometry 的 4×4 只有 3-seed 工程均值优势，没有 10-seed 正式确认。

因此目前不能写成“bilinear 使 local state 失效”，也不能写成“4×4 是 local state 生效的必要条件”。真正有判别力的后续实验必须在同一个 baseline、interaction、station context、loss 和预算下完成：

\[
\{\text{bilinear},\ \text{fixed 4×4},\ \text{learned 4×4+geometry}\}
\times
\{\text{aggregate-only},\ \text{aggregate+local}\}.
\]

每个 local-state 单元还必须做 far-station permutation。只有 4×4 下 `aggregate+local` 稳定优于 `aggregate-only`，并在 local permutation 后显著恶化，才能支持“local state 需要周边空间天气上下文”。若只有开放 direct history path 才恢复旧收益，则旧收益应解释为历史直接预测，而不是天气响应调制。

## 13. 当前集群状态的精确定义与使用方式

当前正式模型 G-Static 中的动态集群状态不是 56 场功率向量，也不是未来功率 baseline latent。先将每个场站功率除以其装机容量，并对全区按容量聚合：

\[
p_{\Sigma,k}
=
\frac{P_{\Sigma,k}}{K_\Sigma}
=
\frac{\sum_i C_i p_{i,k}}{\sum_i C_i}.
\]

在每个预测起点 \(t\)，response-state 分支只看最近 4 个 15 min 集群 capacity-factor 点：

\[
\mathbf p^g_t
=
[p_{\Sigma,t-45\mathrm{min}},
p_{\Sigma,t-30\mathrm{min}},
p_{\Sigma,t-15\mathrm{min}},
p_{\Sigma,t}].
\]

三层 causal TCN/SharedHistoryEncoder 对这四点编码，并取最后一个 token 得到 64 维状态：

\[
s_g=E_g(\mathbf p^g_t)\in\mathbb R^{64}.
\]

它同时包含当前区域功率水平以及最近 15/30/45 min 的共同变化形态，但没有被显式分解为 level、ramp、curtailment 或 NWP bias。因此最严谨的名称仍是 **recent aggregate/common-mode response state**。

对每个场站，标准化 latitude、longitude 和 log-capacity 经过静态编码器得到：

\[
e_i=E_{\rm static}(\operatorname{lat}_i,\operatorname{lon}_i,\log C_i).
\]

未来 Pangu 在场站坐标双线性插值、共享 weather encoder、difference forcing path 和目标时刻查询后得到局地天气表示 \(h^w_{i,q}\)。同一个 aggregate state 被广播到 56 场，并只通过以下 strict interaction 作用于天气表示：

\[
c_i=s_g+e_i,
\]

\[
h'_{i,q}
=h^w_{i,q}
+0.25\left[
M([h^w_{i,q},c_i])-M([0,c_i])
\right].
\]

因此：

\[
h^w_{i,q}=0
\Longrightarrow
M([h^w_{i,q},c_i])-M([0,c_i])=0.
\]

也就是说，集群状态不能绕开天气分支独立产生 correction；它只能改变每个局地天气 latent 被映射为场站 correction 的方式。另一方面，每场最近 12 点历史仍由独立的 baseline branch 使用：

\[
b_{i,1:24}=B(H_{i,t-3h:t}),
\]

最终在当前场站 capacity factor 的 logit anchor 上叠加 baseline 与 weather correction：

\[
\hat p_{i,q}
=
\sigma\!\left[
\operatorname{logit}(p_{i,t})
+\Delta b_{i,q}
+\Delta c_{i,q}(h'^w_{i,q})
\right],
\qquad
\hat P_q=\sum_iC_i\hat p_{i,q}.
\]

这里必须区分两个角色：\(H_i\) 是直接预测未来轨迹的 **forecasting state/baseline input**；\(s_g\) 是只改变未来天气 correction 的 **response condition**。本轮强 donor 实验只错配后者，不破坏 baseline，因而能够把二者的贡献分开。

## 14. 物理科学问题、可解释机制与证据边界

### 14.1 科学问题不是“怎样把 state 塞进网络”

当前真正值得研究的物理问题是：

> **在 0–6 h 时间尺度内，区域内多个风场的天气—功率转换关系是否存在一个跨场站共享、短时持续但未被 Pangu 和场站静态参数显式观测的区域条件，使相同或相近的 Pangu forcing 在不同起报时刻对应不同的实际总功率响应？**

最近四点集群功率不是这个物理条件本身，只是一个可能的观测代理。若记未观测的区域共同条件为 \(\xi_t\)，一个物理上合理但尚未被本实验辨识的生成过程可以写成：

\[
W^{\rm true}_{i,q}
=T_i(W^P_{i,q};\xi_t^{\rm met}),
\]

\[
P_{i,q}
=A_i(\xi_t^{\rm op})
F_i(W^{\rm true}_{i,q},X_{i,t})+\epsilon_{i,q}.
\]

这里 \(W^P\) 是 Pangu forcing，\(W^{\rm true}\) 是风机实际经历但没有被完整观测的局地气象，\(A_i\) 是有效可用容量或运行约束，\(X_{i,t}\) 是场站自身动态。只有当 \(\xi_t\) 在区域内具有共同成分并在未来数小时内有一定持续性时，最近集群总功率轨迹才可能帮助校准未来各场站的天气响应。

这个表述说明了“为什么 aggregate state 可能有用”的现实逻辑，但当前实验没有直接观测 \(\xi_t^{\rm met}\) 或 \(\xi_t^{\rm op}\)，所以不能把任何一种解释写成已经证实的物理机制。

### 14.2 三种物理上可能的共同机制

#### A. Pangu 未解析的区域共同气象偏差

Pangu 格点和低层变量不能完整描述每个风场轮毂高度处的边界层风、稳定度、局地地形加速和空气密度。若某一时段 Pangu 对整个区域的有效风资源存在同号偏差，56 场总功率相当于对真实区域风资源的一次容量加权积分观测。最近总功率持续偏高或偏低，可能提示：

\[
W^{\rm true}_{i}\neq W^P_i
\]

且误差具有区域共同性和时间持续性。此时，aggregate history 的作用更接近 **在线校准 Pangu 的有效 forcing**，而不是设备运行状态。

该解释目前有物理合理性，但尚未被验证，因为模型状态只看实际功率，没有同时看对应历史 Pangu 或场站实测风速，因而无法识别“功率高是因为实际风更强，还是因为设备响应更高”。

#### B. 区域共同的有效可用容量或运行约束

电网调度、区域性限电、覆冰、极端天气或共同运维条件可能使多个场站的有效可用容量同时变化。若这种约束短时持续，则相同未来天气对应的可实现功率会不同：

\[
P_{i,q}\approx A_{i,t}^{\rm effective}F_i(W_{i,q}).
\]

最近集群功率可能间接反映共同的 \(A_t^{\rm effective}\)。此时把同一个 aggregate condition 广播给各场站有一定物理依据，因为它代表 common-mode availability，而不是某个场站自己的功率状态。

但当前没有限电指令、开机容量、故障、覆冰或可用率标签，因此不能声称 G-Static 已经识别了有效可用容量。

#### C. 区域天气系统的相位或传播误差

当锋面或大尺度风场在区域内传播时，Pangu 对到达时间可能存在提前或滞后。最近集群 ramp 汇总了多个场站对该天气系统已经发生的响应，因此可能提供天气系统当前相位的信息，并帮助判断未来 Pangu 路径应该提前、滞后或改变响应强度。

这个解释与“状态主要帮助 3–6 h，而不是 0–1 h”的结果并不冲突，但该时效形态也可能来自 baseline 在近时效已经很强，不能单独作为传播机制证据。

### 14.3 一个同样必须保留的非物理解释

aggregate state 并没有引入新观测，因为它完全可以由 56 场 local history 求和得到。B-Static 的 baseline 已经看到了这些 local history，但要求共享网络自己从 56 条序列中学习稳定的 common mode；G-Static 则把容量加权总和显式提供给 correction。约 `2 MW` 的收益因此也可能仅来自：

- 聚合降低场站测量噪声和局地偶然扰动；
- 显式 common-mode summary 更容易优化；
- 它为共享 decoder 提供了合适的归纳偏置；
- 有限模型容量下，显式聚合比让网络隐式恢复求和更高效。

这属于 **representation/optimization benefit**，不是新的物理状态。如果不能用额外观测或针对性控制排除这一解释，就不能把数值改善上升为“发现了区域运行机理”。

此外，zero-weather invariance 只保证状态不能在天气 latent 严格为零时独自产生 correction；它并不保证网络学习到的 \(s_g\)-dependent 变化一定等价于某个真实物理增益或功率曲线变化。强 donor 错配造成约 `31–34 MW` 恶化，也只能证明模型依赖正确状态，不能辨认这种依赖的物理来源。

### 14.4 当前结果到底支持什么

现有观测事实是：

- G-Static 相对 matched B-Static 在 2023/2024 分别改善 `2.177/1.986 MW`；
- 2023 配对 CI 为 `[-3.150,-1.203] MW`，10/10 seeds 同方向；
- 训练时 strong shuffled-state control 失败；
- 推理时低相关 aggregate donor 使 2023/2024 分别恶化 `30.652/34.311 MW`；
- strict zero-weather invariance 精确满足；
- pure local state 没有在 aggregate state 之上带来稳定增量。

这些结果支持的最强表述是：

> **最近区域总功率轨迹包含一个低噪声、短时有效的共同预测信号；把该共同信号显式用于调整未来天气 correction，比只依赖逐场历史 baseline 更准确。**

当前结果尚不能区分该共同信号究竟对应 Pangu common bias、区域天气相位、共同可用容量，还是更容易学习的统计摘要。因此，现阶段不能声称“区域共同功率状态导致了局地功率曲线变化”，也不能称其为已经辨识出的因果 response law。

pure local state 未提供增量与 common-mode 假设在现象上相容：单场历史包含较多局地噪声、个体限电和偶然波动，而容量聚合可能突出区域共同成分。但这也只是一种解释；由于集群目标会发生误差抵消，而且场站身份与设备信息并不完整，不能据此断言局地物理状态不存在。

### 14.5 要把统计信号提升为物理机制，需要什么证据

后续应围绕候选物理原因设计可证伪实验，而不是继续更换 interaction operator：

1. **气象偏差检验：**用历史同源 Pangu 与对应实际功率或场站实测风速构造 forecast–realization residual state。若 residual state 取代绝对 aggregate P4 后仍稳定有效，并与实测 Pangu 风速误差同号，才支持共同气象偏差解释。
2. **有效容量检验：**引入限电、开机容量、故障、覆冰或可用率记录；剔除受限时段或显式控制有效容量后重新估计 state gain。只有增益随这些运行条件系统变化，才能支持 availability 解释。
3. **传播相位检验：**在相同未来 Pangu 路径和相近当前总功率下，检验 aggregate ramp 是否能预测未来 residual 的符号、峰值到达时间和空间传播顺序，而不只是降低总体 MAE。
4. **统计摘要对照：**把同一 aggregate P4 只送入 baseline-only 路径，与 strict weather-interaction 路径做 matched comparison。若 baseline-only 获得相同收益，则不支持“天气响应关系变化”；只有 strict interaction 在状态匹配、天气 regime 和跨年上获得额外收益，才有理由继续讨论 response regime。
5. **空间支持对照：**完成 `bilinear/fixed-4×4/learned-4×4 × aggregate-only/aggregate+local` 矩阵及 local permutation。它只回答局地状态是否需要周边天气上下文，不能替代上述气象和运行观测。

在完成这些检验前，当前主任务仍保留全部 56 场、逐场预测后求和，并以集群 MAE 为 primary metric；不根据 validation/test 结果挑选“支持故事”的场站。当前论文若立即成稿，科学结论应停留在 **区域共同历史信号的可预测性及其聚合尺度效应**，而不强行指定尚未被数据识别的物理成因。
