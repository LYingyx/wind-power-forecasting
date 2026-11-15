# 现有实验总结与跨分辨率研究主线

更新日期：2026-09-02  
代码与结果快照：`wind_cluster_forced_evolution` 当前工作区  
用途：统一说明已有模型究竟是什么结构、关键实验分别验证了什么，并据此把论文主线收敛为“原生分辨率未来气象到细粒度功率响应的跨分辨率建模”。

---

## 1. 先给结论

现有实验已经较稳定地回答了三个问题。

1. **不能再使用“天气只负责低频、历史功率只负责高频”的频谱分工。** 对联合修正轨迹施加 DCT 低频限制没有带来稳定优势，DCT-4 还显著变差。这个结果既不支持低频分工，也不能反向证明天气提供了高频信息。
2. **未来 NWP 是历史基线之外的主要预测增量，近期功率状态的主要价值是帮助模型解释同一未来天气在不同运行状态下对应的功率响应。** 严格四格实验中，额外历史状态在没有天气时几乎无效，在有天气时带来 14.285 MW 的单模型改善；差中之差为 13.932 MW。
3. **当前最可靠的部署候选不是新的显式 fusion 算子，而是一个结构简单、职责清楚的加性模型，再用 ensemble-consensus knowledge distillation（KD）降低训练方差：**

   \[
   \widehat Y=B(H)+C(S_{\mathrm{recent}},W,\tau).
   \]

   其中，基线分支读取完整 6 h 场站与集群历史；修正分支读取未来 NWP、目标时刻和最近 1 h 的**集群总功率轨迹**。当前默认修正内部使用一个统一的 black-box MLP，不强行把“纯天气作用”和“状态—天气交互”拆成两个可解释输出。

2024 held-out test 上，10 个独立 KD 学生的平均单模型 MAE 为 **250.966 ± 3.184 MW**，相对同结构但不蒸馏的模型改善 **6.838 MW，10/10 seeds 获胜**；相对无显式 baseline/correction 分工的 Direct joint 模型改善 **11.068 MW，10/10 seeds 获胜**。不过，单个 KD 学生仍未达到十模型 ensemble 的精度，不能宣称已经无损压缩 ensemble。

如果将故事线转为跨分辨率，当前最准确的定位不是“气象超分辨率”，而是：

> **在不虚构 15 min 高分辨率气象真值的前提下，把原生小时、规则网格的未来 NWP 转化为场站级、15 min 目标时刻的条件功率响应。**

这条故事目前**时间跨分辨率证据较强，空间跨分辨率证据不足**。时间节点差分编码、完整未来路径和目标时刻查询都有 test 支持；但学习式 4×4 空间聚合尚未被证明优于固定双线性方法，而且还缺少原始气象插值、同支持范围和不同空间范围的关键对照。因此，跨分辨率可以成为新的科学问题主线，但尚不能把“学习式时空重采样优于插值”写成已验证贡献。

---

## 2. 数据、任务与证据口径

### 2.1 输入输出分辨率

任务包含三种不一致的分辨率。

| 对象 | 时间分辨率 | 空间分辨率 | 当前数据形状 |
|---|---|---|---|
| 历史功率 | 15 min | 56 个不规则分布场站，并可容量加权为集群功率 | 每场站 24 点，即过去 6 h |
| 未来 NWP | 原生小时节点 | 0.25°规则经纬度网格 | 8 节点 × 14 变量 × 24 × 25 网格 |
| 预测目标 | 15 min | 56 个场站及其容量加权集群 | 24 点，即未来 0.25–6 h |

总装机容量为 6827.01 MW。模型内部主要使用容量归一化功率因子，文档中的 cluster MAE/RMSE 已恢复成 MW。

### 2.2 数据划分

- 2022：train，31,190 个 forecast origins；
- 2023：validation，32,550 个 origins、359 个 UTC 日；
- 2024：held-out test，28,782 个共同 origins、341 个 UTC 日；
- 正式模型均使用 seeds 42–51，共 10 次独立训练；
- checkpoint 由 validation 选择，训练和 early stopping 不读取 2024 test。需要同时承认：仓库已经在该 test split 上完成多组结构消融和反复分析，所以它是相对训练过程的 held-out test，但已不是相对整个研究迭代完全未触碰的 pristine holdout。

需要区分两类证据。

- **2024 test 证据：**主模型比较、最终结构消融、KD target 消融、天气置乱、效率和预测图。可用于论文主结果。
- **2023 validation 机制证据：**DCT、严格四格、状态阶梯、显式 interaction、horizon operator、跨场站和区域 composition。它们用于确定结构与解释作用边界，不能伪装成独立 test 结论。

### 2.3 单模型与 ensemble 的含义

- “单模型 mean ± sd”表示 10 个独立 checkpoint 各自计算指标后，再跨 seed 汇总。
- “10-model ensemble”表示这 10 个 checkpoint 的预测逐点算术平均后再计算指标。
- 当前 ensemble 都是**同构、不同随机种子的模型平均**。它主要平均优化噪声、降低方差，并没有显式让不同成员负责不同特征或时效。
- KD teacher 是 10 个同构 Normal 模型在训练集上的预测均值；KD 学生仍是单个同结构模型。KD 不是简单报告十个 seed 的平均指标，也不是把十个模型直接平均后称作一个模型。

---

## 3. 简称—实际结构字典

仓库中的同一个简称在不同实验中可能代表不同控制语境。本节规定本文档的完整含义。

### 3.1 基础参照

| 文档名称 | 实际输入与结构 | 用途 |
|---|---|---|
| **History-only direct / Power-only direct** | `forced_evolution` 的 `history_only` 模式。读取完整 6 h 的 56 场站历史和集群历史，经共享因果 TCN、相对时间查询和轨迹解码器直接输出未来 24 步；完全不读取天气。 | 功率历史基线。2023 validation 单模型 MAE 410.471 ± 0.754 MW。 |
| **Weather-only direct** | `forced_evolution` 的 `weather_only` 模式。只读取空间适配后的未来 NWP 和目标时间，直接输出场站功率；没有历史基线，也不读取历史状态。 | 检验纯天气直接预测能力。不要与严格四格中的 weather-only 混淆。 |
| **Direct joint d64** | `forced_evolution` 的直接联合预测结构，隐层宽度 64，0.975M 参数。完整历史和未来 NWP 在一个 `state_conditioned` 直接预测网络中融合，并以当前功率为输出基准；没有显式的 \(B+C\) 职责分解。 | 当前最强的直接联合 baseline。不是简单线性拼接，也不是 correction 模型。 |
| **Direct joint d72** | 与 Direct joint d64 相同，只把隐层宽度增加到 72，使参数量增至 1.231M，接近 correction 模型的 1.287M。 | 排除“correction 仅因参数更多而更好”。 |

### 3.2 加性 correction 家族

| 文档名称 | 实际输入与结构 | 是否 KD |
|---|---|---:|
| **FullHistory correction** | \(B(H)+C(H,W,\tau)\)。基线和修正都能读取完整 6 h 历史；修正还读取未来 NWP。部署版本中基线可训练，并以预训练 \(B_0(H)\) 做权重 0.2 的弱预测锚定。 | 否 |
| **Recent-state Normal** | \(B(H)+C(S_{\mathrm{cluster},1h},W,\tau)\)。基线读取完整场站/集群 6 h 历史；修正只读取最近四个 15 min 的**集群总功率**，不读取各场站近期功率轨迹。black-box correction、可训练 baseline、anchor 0.2。 | 否 |
| **Recent-state consensus-KD student** | 与 Recent-state Normal 的输入、网络和 1,287,343 个参数完全相同；训练时额外拟合 10 个 Normal teacher 的训练集 station-level 与 cluster-level 平均预测，KD 权重 0.5。 | 是 |
| **EMA recent-state** | 不改变输入或结构；对 Normal 单次训练的模型权重做 decay=0.999 的指数滑动平均，并在 Normal 选定的 epoch 取值。 | 否 |
| **Checkpoint-average recent-state** | 不改变输入或结构；平均 Normal 最佳 epoch 及其之前最近五个 checkpoint 的权重。 | 否 |

这里的 `Normal` 只表示“未加 KD 的默认 recent-state correction”，不是数据归一化，也不是某种 baseline 名称。`KD` 只表示训练目标发生变化，**不表示网络结构被压小**。

### 3.3 机制探针中的 weather-only 与 history-only

严格四格和状态阶梯中的命名与 direct 模型不同。

- `C(context)`：共同的完整历史基线 \(B(H)\) 保留，但 correction 的历史槽和天气槽都置零，只保留场站 embedding、目标时刻等共同上下文。
- `C(H)`：共同基线保留，correction 只获得额外历史状态，不获得未来天气。
- `C(W)`：共同基线保留，correction 只获得未来天气，不获得额外历史状态。
- `C(H,W)`：共同基线保留，correction 同时获得额外历史状态和未来天气。

所以严格四格中的“weather-only”实际是 \(B(H)+C(W)\)，并非完全不看历史；历史仍通过共同基线存在。比较不同文档的绝对 MAE 前必须先确认结构语境。

### 3.4 冻结基线探针

`Frozen`、`state ladder`、`explicit interaction` 和 `horizon control` 使用同一个完全冻结的预训练 \(B_0(H)\)，只训练 correction。它们用于隔离 correction 所需信息，不能把其绝对 MAE与最终联合训练模型直接比较。最终部署模型的 baseline 不冻结，因为 2024 test 上冻结 baseline 使 ensemble MAE 变差 7.340 MW。

---

## 4. 当前最终候选模型：逐模块结构说明

当前正式候选是 **Recent-state consensus-KD student**。整体计算可以写成：

\[
\begin{aligned}
Z^{\mathrm{grid}}_{m,x,y} &= E_{\mathrm{point}}(W_{m,x,y}),\\
Z^{\mathrm{farm}}_{i,m} &= A_{\mathrm{space}}(Z^{\mathrm{grid}}_{m},W_m;i),\\
\widetilde Z^{\mathrm{farm}}_{i,1:M} &= E_{\mathrm{path}}(Z^{\mathrm{farm}}_{i,1:M},\Delta Z^{\mathrm{farm}}_{i,1:M}),\\
R^{W}_{i,q} &= A_{\mathrm{time}}(Q_{i,q},\widetilde Z^{\mathrm{farm}}_{i,1:M},t_q-t_m),\\
R^{S}_{i,q} &= E_{\mathrm{state}}(S_{\mathrm{cluster},1h},i,t_q),\\
C_{i,q} &= D_{\mathrm{corr}}(R^{S}_{i,q},R^{W}_{i,q},Q_{i,q}),\\
\widehat Y_{i,q} &= \max\{0,B_i(H)+C_{i,q}\}.
\end{aligned}
\]

其中 \(i\) 是场站，\(m\) 是 8 个原生 NWP 小时节点，\(q\) 是 24 个 15 min 目标时刻。下面按框架图顺序说明。

### 4.1 历史基线分支 \(B(H)\)

输入：

- 56 个场站过去 24 个 15 min 功率点；
- 同期容量加权集群功率 24 点；
- 起报相位和 24 个目标 lead。

实现：

1. 共享因果 TCN 分别编码场站历史与集群历史，3 层 dilation residual blocks；
2. 每个 15 min 目标 token 通过带相对时间 bias 的 cross-attention 读取历史轨迹；
3. 场站历史响应、全局集群状态和目标时间经 MLP、轨迹 TCN 与 1×1 head 输出 24 步场站基线；
4. 基线由 history-only checkpoint 初始化，但最终与 correction 联合训练，并非冻结；
5. 训练时用 Smooth-L1 约束当前 \(B(H)\) 不要过度偏离固定预训练函数 \(B_0(H)\)，anchor 权重为 0.2。

anchor 的作用是缓解 \(B+C\) 分解不可辨识：没有约束时，任意只依赖历史的 \(\delta(H)\) 都可在 \(B\) 与 \(C\) 之间转移而不改变总预测。它不是“基线必须始终等于真实标签”的强监督，也不是 test 上已证明全时效都有利的模块。

### 4.2 逐网格点气象编码

输入为 8×14×24×25 的标准化 NWP。

当前正式配置 `include_region_context=false`，因此实际实现是：

- 每个原生网格点独立经过同一个 MLP；
- 在 14 个标准化气象变量之外，从三对风矢量通道计算三个风速幅值作为派生特征；
- MLP 维度为 `17 -> 48 -> 64`，输出每个网格点的 64 维 token；
- **没有在 24×25 网格上运行卷积，也没有全区域 CNN context。**

因此，论文或框架图若把当前 weather encoder 画成卷积网络是不准确的。

### 4.3 空间跨分辨率适配：规则网格到不规则场站

每个场站先根据经纬度确定其所在的 NWP 网格单元。配置 `station_neighborhood_cells=3` 在代码中形成包含该单元及外围一圈的 **4×4、共 16 个固定候选网格点**。

候选集合是固定的，但候选权重是随 NWP 内容变化的。每个权重由以下信息共同决定：

- 当前网格点的气象 token；
- 场站静态 embedding：纬度、经度、容量及对数容量；
- 网格点相对场站的东西/南北距离与总距离；
- 默认配置中的 10 m 风矢量沿场站—网格连线方向和横向的投影。

16 个权重经过 softmax 后加权汇聚为“该场站在该原生小时节点的气象 token”。需要明确：

- 当前只有**候选范围固定、范围内权重自适应**；不是自适应搜索任意远处网格；
- 默认 recent-state 不参与这一步的权重计算；历史状态不会控制空间取点位置；
- 2024 test 尚未证明这一 learned 4×4 聚合优于固定双线性，因此它目前是可用实现，不是已经成立的核心创新。

### 4.4 小时天气路径编码

空间适配后，每个场站得到 8 个原生小时 token。`forcing_path_mode=difference` 对每个节点加入可学习位置编码，并拼接：

- 当前节点 token；
- 当前节点与前一小时节点的差分，首节点差分置零。

拼接结果经 MLP 和 LayerNorm 重新映射到 64 维。这一步显式保留天气系统在相邻小时节点间的变化，而不是把 8 个节点当作互不相关的快照。

每个小时 token 还有一个辅助 head，在训练时预测该 NWP 有效时刻对应的场站功率因子。它使用实际场站功率在这些**小时有效时刻**的标签，损失权重 0.2。这个 head 为天气 token 增加功率响应语义，但它没有生成 15 min 气象，也没有显式生成一条 15 min 代理功率轨迹；最终预测仍由后续目标时刻查询和 correction decoder 产生。

### 4.5 时间跨分辨率适配：8 个小时 token 到 24 个 15 min 目标响应

每个 15 min 目标 \(q\) 构造一个 target token，包含：

- 连续 lead 的线性、正弦和余弦编码；
- 四种起报相位 embedding；
- 场站静态 embedding。

该 token 通过 4-head cross-attention 读取全部 8 个小时天气 token。attention score 除内容相似度外，还加入真实相对时间特征：

\[
\Delta t_{q,m}=t_q-t_m,\qquad |\Delta t_{q,m}|.
\]

输出是“场站 \(i\)、目标时刻 \(q\) 特定的 64 维天气响应表示”。它不是以下任何一种对象：

- 不是插值得到的 15 min、14 变量气象场；
- 不是具有气象单位和气象监督的超分辨率 NWP；
- 不是直接得到的 15 min 代理功率；
- 也不只是相对时间编码——内容 attention、小时路径差分和相对时间 bias 共同参与。

更准确的名称是 **target-time-conditioned latent forcing response**。

### 4.6 近期状态分支

当前默认 `correction_history_mode=cluster_recent_1h`，只把最近四个 15 min 的集群总功率暴露给 correction。

为了兼容固定长度的 24 点 TCN 输入，代码用这四点中最早的一点常数填充前 20 个位置；场站状态编码器收到的是同一条集群轨迹复制到 56 个场站，再与不同的场站 embedding 相加。全局状态编码器也读取这条集群近期轨迹。

这意味着：

- correction 知道集群当前 operating level、15/30/45 min 的近期变化和短轨迹形态；
- correction **不知道每个场站自身最近四点的真实轨迹**；
- 56 场站完整历史仍被基线 \(B(H)\) 使用，因此模型整体并未丢弃场站历史；
- 当前模型也没有图消息传播或显式场站邻接。

### 4.7 状态—天气融合与 correction 输出

默认 `correction_fusion_mode=blackbox`。模型分别得到：

- 近期状态对每个目标时刻的 history response；
- 集群 global state；
- 目标时刻特定的 weather response；
- target token。

四者直接拼接后进入两层 MLP，再经 3 层 trajectory TCN 和 1×1 correction head，输出每场站 24 步 correction。这里没有单独可解释的 `weather-only correction` 与 `state-weather interaction correction` 两个输出头。

这个选择是有意的：严格四格可以在**实验分析层面**区分天气主增量和历史条件收益，但一般神经网络中的两个输出分支存在重参数化与异号抵消，不能仅凭分支数值解释为物理贡献。当前总公式保留单一 \(C(S,W,\tau)\) 更稳妥。

默认状态也不改变 weather attention 读取哪个小时节点；它在天气已被读取之后调制“天气表示如何映射为功率修正”。Conditioned query 和 attention-bias 实验没有支持“状态主要通过改变天气节点选择而起作用”。

### 4.8 层级输出

场站输出为：

\[
\widehat Y_{i,q}=\max\{0,B_i(H)+C_{i,q}\}.
\]

集群输出由 56 个场站结果按装机容量加权：

\[
\widehat Y^{\mathrm{cluster}}_q=
\frac{\sum_i K_i\widehat Y_{i,q}}{\sum_i K_i}.
\]

模型只截断负值，并未硬截断到场站容量上限。2024 test 的 seed42 诊断中，负值比例为 0，超过容量的预测比例约为 0.00115。

### 4.9 训练损失与 KD

非 KD 的监督部分为 station 与 cluster 两层 Smooth-L1：

\[
\mathcal L_{\mathrm{sup}}=
1.0\,\mathcal L_{\mathrm{station}}
+2.0\,\mathcal L_{\mathrm{cluster}}.
\]

再加入：

- 小时天气 token 到小时场站功率的辅助损失，权重 0.2；
- 当前 baseline 到固定 \(B_0\) 预测的 anchor，权重 0.2；
- KD 学生额外拟合 teacher 的 station 与 cluster 两层软目标，\(\lambda_{\mathrm{KD}}=0.5\)。

teacher 的生成过程是：先独立训练 10 个完全相同结构的 Normal 模型，用它们对**训练集**产生预测，再逐点平均为软标签。学生仍同时学习真实标签，形式是

\[
\mathcal L=\mathcal L_{\mathrm{sup}}
+0.2\mathcal L_{\mathrm{weather\ aux}}
+0.2\mathcal L_{\mathrm{anchor}}
+0.5\mathcal L_{\mathrm{KD}}.
\]

推理时单个 KD 学生不需要 teacher，也不需要运行十个 Normal 模型。只有显式选择 KD ensemble 时才运行多个 KD checkpoint。

---

## 5. 2024 held-out test：最终模型与主基线

下表中的每一行都写出完整结构。MAE、RMSE、\(R^2\)、ramp MAE 和 persistence skill 是 10 个独立单模型的 mean ± sd；ensemble 列是十个 checkpoint 的预测均值。

| 完整结构 | 参数量 | cluster MAE (MW) | cluster RMSE (MW) | \(R^2\) | 15 min ramp MAE (MW) | 相对 persistence 的 MAE skill | 10-model ensemble MAE (MW) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct joint d64：完整历史与未来 NWP 在单一直接 `state_conditioned` 网络中融合，无 \(B+C\) 分工 | 0.975M | 262.034 ± 2.530 | 385.704 ± 5.230 | 0.8974 ± 0.0028 | 52.558 ± 0.464 | 0.4207 ± 0.0056 | 250.389 |
| FullHistory correction：\(B(H)+C(H,W,\tau)\)，baseline/correction 都读完整 6 h 历史，anchor 0.2，无 KD | 1.287M | 257.323 ± 2.901 | 375.646 ± 5.317 | 0.9027 ± 0.0028 | 52.508 ± 0.564 | 0.4311 ± 0.0064 | **242.594** |
| Recent-state Normal：\(B(H)+C(S_{\mathrm{cluster},1h},W,\tau)\)，black-box correction，anchor 0.2，无 KD | 1.287M | 257.805 ± 2.361 | 381.071 ± 3.863 | 0.8998 ± 0.0020 | 52.182 ± 0.481 | 0.4300 ± 0.0052 | 244.687 |
| Recent-state consensus-KD：与上一行同结构，用 10-Normal consensus 做 station+cluster 蒸馏，KD 权重 0.5 | 1.287M | **250.966 ± 3.184** | **370.605 ± 4.521** | **0.9053 ± 0.0023** | **51.470 ± 0.265** | **0.4452 ± 0.0070** | 242.755 |

配对统计：

- KD 相对 Direct：单模型平均改善 11.068 MW，10/10 seeds 获胜；ensemble 改善 7.634 MW，按 UTC 日 bootstrap 95% CI 为 `[4.183, 10.995]`。
- KD 相对未蒸馏 Recent-state Normal：单模型平均改善 6.838 MW，10/10 seeds 获胜；ensemble 改善 1.932 MW，95% CI `[0.629, 3.247]`。
- KD 相对 FullHistory：单模型平均改善 6.356 MW，9/10 seeds 获胜；但 KD ensemble 比 FullHistory ensemble 差 0.161 MW，95% CI `[-1.764, 2.205]`，两者不可区分。

因此，当前推荐 KD 的核心原因是**提升单模型精度并把 ensemble 共识转入单模型**，不是因为 KD ensemble 显著超过所有模型。单个 KD 的 250.966 MW 仍高于 KD ensemble 的 242.755 MW，也高于 FullHistory ensemble 的 242.594 MW；“单模型无损替代十模型 ensemble”尚未成立。

---

## 6. 关键机制实验及其实际含义

### 6.1 DCT 频谱约束：否定硬频带分工

DCT-K 是把**联合 correction 的 24 步输出轨迹**投影到前 K 个 DCT-II 余弦基上。它限制的不是气象输入本身，也无法把某个频率唯一归因给天气或历史。

2023 validation、10 seeds：

| 结构 | 单模型 MAE (MW) | ensemble MAE (MW) |
|---|---:|---:|
| Direct joint d64，无 \(B+C\) 分工 | 262.089 ± 1.787 | 250.203 |
| Direct joint d72，近容量控制 | 263.595 ± 2.024 | 250.038 |
| \(B(H)+C(H,W)\)，全 24 个时间自由度 | **259.946 ± 1.916** | 244.229 |
| 同一 correction 仅保留前 4 个 DCT 模式 | 260.718 ± 1.402 | 244.784 |
| 前 7 个模式 | 260.203 ± 1.381 | 244.588 |
| 前 8 个模式 | 260.343 ± 1.676 | **243.714** |
| 前 12 个模式 | 259.929 ± 1.963 | 243.975 |

DCT-4 相对全频单模型变差 0.772 MW，95% CI `[0.150, 1.394]`。DCT-8 只有 ensemble 点估计较好，但单模型平均反而差 0.396 MW、只赢 5/10，seedwise 与 bootstrap 区间都跨 0。DCT-7/8/12 都没有稳定优于全频。

可支持的结论只有：**硬低频 correction 不是必要且 DCT-4 过窄。**不能据此说天气提供高频，也不能说 DCT-12 已与全频严格等效。

### 6.2 严格四格：天气是主增量，额外状态主要条件化天气

四格共享同一个完整历史基线、1,287,343 参数 backbone 和四槽输入，只对 correction 的 H/W 槽置零。2023 validation、10 seeds：

| correction 的完整可见输入 | 单模型 MAE (MW) | ensemble MAE (MW) |
|---|---:|---:|
| 只保留场站与目标时刻上下文：\(B(H)+C(context)\) | 411.962 ± 0.722 | 408.055 |
| 只增加额外历史状态：\(B(H)+C(H)\) | 411.609 ± 0.799 | 408.304 |
| 只增加未来 NWP：\(B(H)+C(W)\) | 274.231 ± 1.595 | 260.461 |
| 同时增加历史状态与未来 NWP：\(B(H)+C(H,W)\) | **259.946 ± 1.916** | **244.229** |

- 额外历史在无天气时只改善 0.353 MW，95% CI `[-0.188, 0.894]`；
- 额外历史在有天气时改善 14.285 MW，95% CI `[12.993, 15.576]`；
- 差中之差为 13.932 MW，95% CI `[12.599, 15.264]`，10/10 seeds 同方向；
- 历史条件化收益随 horizon 增强：0.25 h 为 -2.587 MW，0.75 h 起可靠为正，3 h 为 17.361 MW，6 h 为 29.195 MW。

正确解释是**预测误差上的非加性互补**：未来天气带来主增量，近期状态帮助模型确定天气到功率的映射。它不是天气对功率的物理因果效应识别，也不是神经网络函数分解的唯一证明。

### 6.3 冻结基线状态阶梯：需要运行水平、趋势和近期轨迹

所有组固定完全相同的 \(B_0(H)\)，只改变 correction 可见状态。2023 validation、10 seeds：

| correction 状态结构 | 单模型 MAE (MW) | ensemble MAE (MW) |
|---|---:|---:|
| 无额外功率状态，只看天气 | 300.716 ± 3.300 | 287.404 |
| 当前集群总功率 | 282.879 ± 3.500 | 268.894 |
| 当前 56 场站功率快照，并保留集群当前值 | 281.825 ± 3.461 | 268.268 |
| 当前集群总功率 + 最近 15 min ramp | 270.785 ± 4.128 | 256.105 |
| 最近四点场站轨迹 + 集群轨迹 | 265.759 ± 3.770 | 250.637 |
| 完整 24 点场站与集群历史 | **264.858 ± 3.337** | **249.067** |

主要增量：

- 当前集群 operating level 相对无状态改善 17.837 MW，10/10；
- 加 15 min ramp 再改善 12.094 MW，10/10；
- 最近四点相对 current+ramp 再改善 5.026 MW，10/10；
- 完整 24 点相对最近四点只改善 0.900 MW，95% CI `[-0.144, 1.945]`。

所以状态模块的主要内容是 operating level、近期趋势和 recent trajectory，而不是完整 6 h 长记忆。

### 6.4 空间快照 × 近期轨迹 2×2

同样冻结基线，只改变 correction 的功率状态分辨率：

| correction 状态 | 单模型 MAE (MW) | ensemble MAE (MW) |
|---|---:|---:|
| 集群当前值 | 282.879 ± 3.500 | 268.894 |
| 56 场站当前快照 | 281.825 ± 3.461 | 268.268 |
| 最近 1 h 集群轨迹复制到场站 | 267.214 ± 3.985 | 252.577 |
| 最近 1 h 各场站轨迹 + 集群轨迹 | **265.759 ± 3.770** | **250.637** |

- 集群近期轨迹相对集群快照：ensemble 改善 16.317 MW；
- 相同近期窗口下，场站轨迹相对集群轨迹：单模型改善 1.455 MW，95% CI `[0.434, 2.476]`；ensemble 改善 1.939 MW，CI `[0.799, 3.091]`。

这说明场站分辨率近期轨迹有小而稳定的 validation 增量，但主效应仍是近期时间轨迹。它没有证明地理图、信息传播或空间因果。还要注意：**当前最终 KD 候选使用的是集群近期轨迹，不是这里数值更好的场站近期轨迹。**如果论文强化“空间跨分辨率”，这是一个需要补齐 test/KD 对照的结构缺口。

### 6.5 显式 interaction 与 horizon operator：未找到稳定更好的融合算子

冻结基线下，不同 fusion 的 validation 结果：

| correction 内部结构 | 完整含义 | 单模型 / ensemble MAE (MW) |
|---|---|---:|
| black-box | 状态、天气、目标时间拼接后由统一 MLP 学习 correction | 267.214 ± 3.985 / 252.577 |
| additive | 显式天气主分支 + 显式状态主分支相加 | 267.053 / 253.458 |
| multiplicative | 天气主分支 + 状态与天气低秩乘法交互 | 266.261 / 251.946 |
| asymmetric modulation | 状态缩放天气表征后由共享 decoder 输出 | 266.879 / 253.185 |

乘法结构点估计最好，但相对 black-box 的 seedwise 与 ensemble 区间均跨 0；显式分支还出现大量异号抵消。因此不把分支输出解释成真实天气贡献和交互贡献。

进一步比较五种“状态控制位置”：

| 结构 | 状态控制对象 | 单模型 / ensemble MAE (MW) | 判断 |
|---|---|---:|---|
| Horizon-FiLM | 天气 feature 的 scale/shift | 266.106 / 251.467 | 长时有 ensemble 信号，但短时显著变差且 seedwise 不稳 |
| Gated multiplicative | state-weather interaction 强度 | 267.276 / 252.908 | 无优势 |
| Conditioned query | weather attention 的 query | 268.787 / 254.121 | 无优势，短中时变差 |
| Attention bias | state-weather compatibility score | 269.962 / 254.836 | 显著更差 |
| Conditional LoRA | weather-to-power feature mapping | 266.802 / 251.966 | 无稳定优势 |

所以默认保留统一 black-box correction。特别是 query/bias 结果不支持“状态的主要作用是改变模型读取哪个天气节点”。

---

## 7. 已有跨分辨率证据

### 7.1 时间跨分辨率：已有较强 test 支持

以下消融都在 2024 test、10 seeds 上完成。表中正值表示替代方案相对未蒸馏 Recent-state Normal 变差。

| 替代结构 | 实际改变 | ensemble 变差 (MW) | 95% CI (MW) | 结论 |
|---|---|---:|---:|---|
| Independent hourly nodes | 每个小时 token 独立处理，不拼接相邻小时差分 | **+7.317** | `[3.942, 10.971]` | 强烈支持编码小时节点间演化；不要误写成“状态直接调制 forcing path” |
| Nearest-hour query | 每个 15 min 目标只读取时间上最近的一个小时 token | +1.061 | `[-0.684, 2.734]` | 总体不显著；短时略好、长时明确更差 |
| No hourly power auxiliary | 去掉每个 NWP 小时 token 到对应小时场站功率的辅助监督 | **+2.188** | `[1.062, 3.282]` | 对未蒸馏基础模型有支持；最终 KD 中证据混合 |
| Frozen history baseline | 预训练历史基线完全冻结，不与跨分辨率 correction 联合适配 | **+7.340** | `[5.028, 9.662]` | coarse forcing 与历史基线需联合校准 |

Nearest-hour 的总体均值掩盖了时效差异：它在 0.25–1 h 改善 0.599 MW，但在 3.25–6 h 变差 2.891 MW且区间不跨 0。因此，完整 8 节点路径主要为中长时功率演化提供信息。

这些结果支持“小时路径编码 + 目标时刻查询”的价值，但仍有两个边界。第一，`independent` 变体比 Normal 少 8,256 个参数，虽然只占约 0.64%，仍应在 T0–T4 矩阵中做严格容量匹配。第二，现有实验没有把当前 attention 与**原始气象变量先线性插值到 15 min**的强基线做正式 10-seed、同参数比较。因此不能说当前方法已经优于常规时间插值。

### 7.2 空间跨分辨率：实现存在，优越性尚未成立

2024 test 空间消融：

- 去掉 wind geometry 后，ensemble 点估计反而改善 0.702 MW，但 CI `[-2.072, 0.619]` 跨 0；无法证明风向几何必要。
- 在都不使用 wind geometry 的干净比较中，fixed latent-space bilinear 相对 learned 4×4 的 ensemble 变差 1.101 MW，CI `[-0.585, 2.788]`；单模型 fixed bilinear 反而好 0.399 MW，区间也跨 0。

这里的 fixed bilinear 是：先对四个原生网格点分别做非线性 MLP 编码，再按场站经纬度对**latent token**做固定双线性加权。它不是把 14 个原始气象变量先双线性插值到场站再编码。

因此当前只能声称：模型实现了 forecast-oriented grid-to-farm adaptation；不能声称 learned 4×4 已优于 bilinear，更不能声称恢复了真实场站气象。

### 7.3 天气依赖性

2024 test 中，在控制月份、起报相位、NWP age 和当前功率分位后置乱未来天气：

- Recent-state Normal 的 MAE 平均恶化 287.918 MW；
- KD 的 MAE 平均恶化 299.749 MW。

这排除了模型忽略天气、只复制历史的退化解释。置乱可能产生分布外的历史—天气组合，所以恶化幅度不能解释为天气的物理因果贡献量。

### 7.4 预测图和扩展指标

仓库已有 314 张 PNG 和 314 张 PDF，全部来自 2024 test，包括：

- 固定随机样本；
- 每月中位误差样本；
- 相对 Direct 改善最大与退化最大的样本；
- 最大未来 ramp 和最大当前 ramp 样本；
- 逐 lead、regime、场站和误差分位图。

这些图可用于展示成功与失败案例，但按“最大改善”筛选的图只能作案例解释，不能作为总体性能证据。机器可读索引位于 `artifacts/paper_evidence_test_v1/cluster/tables/case_manifest.csv`。

KD 相对 Direct 的 test 指标不仅是 MAE 改善：10-seed 平均 RMSE 从 385.704 降至 370.605 MW，\(R^2\) 从 0.8974 升至 0.9053，ramp MAE 从 52.558 降至 51.470 MW，persistence skill 从 0.4207 升至 0.4452。说明收益并非只来自某一个 MAE 统计口径。

---

## 8. 空间功率信息共享与区域 composition

以下均为 2023 validation 探索，不是当前最终模型，也没有进入 2024 test 主结果。

### 8.1 跨场站近期状态共享

| 名称 | 完整结构 | 单模型 / ensemble MAE (MW) |
|---|---|---:|
| G0 aggregate recent | correction 只看最近 1 h 集群轨迹，并复制到各场站；不看其他场站的真实近期轨迹 | 258.563 ± 1.816 / 244.732 |
| G1 self duplicate | correction 读取各场站自身最近 1 h 轨迹；再增加一个与自身完全重复的第二通道，控制额外通道和参数 | 257.480 ± 1.793 / 243.916 |
| G2 geographic-K4 | 第二通道改为地理距离最近 4 个其他场站近期轨迹的聚合 | 257.205 ± 1.996 / 243.369 |
| G3 global-other | 第二通道改为除自身外所有场站近期轨迹的全局聚合 | 257.163 ± 2.124 / 243.434 |
| G4 weather-coherent-K4 | 第二通道改为未来天气表示最相似的 4 个其他场站近期轨迹聚合 | **256.858 ± 2.307 / 242.412** |

G4 相对 G1 的 ensemble 改善为 1.504 MW，bootstrap CI `[0.488, 2.494]`；但 seedwise 平均只改善 0.622 MW、CI 跨 0，且 Holm 校正不支持。长时点估计更好。

因此只能说“天气一致邻居可能为 ensemble center 提供小的长时信号”。它没有形成稳定单模型机制，也没有证明地理传播或动态邻接已有效。

### 8.2 区域功率分布 composition

为检验“集群总功率相同但场站间分布不同”是否是缺失状态，做了参数匹配实验：

| 完整结构 | 单模型 / ensemble MAE (MW) |
|---|---:|
| R0 matched control：增加同样模块和参数，但只输入冗余集群统计 | 258.258 / **242.496** |
| R1 handcrafted statistics：输入场站功率分位数、均值、方差和 ramp 分布统计 | 258.775 / 243.022 |
| R2 tiny DeepSets：对场站近期状态集合做小型 permutation-invariant 编码 | **258.093** / 242.668 |

R1、R2 相对参数匹配 control 都没有稳定优势。因此区域 composition 目前不应成为主线。

---

## 9. Ensemble、KD 与效率的统一解释

Recent-state Normal 在 validation 上的单模型—ensemble gap 为 13.831 MW，seed 间 prediction disagreement 为 103.214 MW，平均 signed-error correlation 为 0.8977。分歧和 ensemble gain 都随 horizon 增大，说明中长时的优化方差更大。

稳定化比较：

| 方法 | 实际操作 | validation 单模型 / ensemble MAE (MW) | 相对 Normal 单模型改善 |
|---|---|---:|---:|
| Recent-state Normal | 原始最佳 checkpoint | 258.563 ± 1.816 / 244.732 | — |
| EMA | 同一次训练的权重指数平均 | 253.494 ± 2.163 / 245.251 | 5.069 |
| Last-five checkpoint average | 同一次训练最近五个 checkpoint 权重平均 | 256.731 ± 2.535 / 247.410 | 1.832 |
| Plain KD 0.25 | 拟合 10-Normal 共识，KD 权重 0.25 | 254.438 ± 1.687 / 243.437 | 4.125 |
| Plain KD 0.5 | 拟合 10-Normal 共识，KD 权重 0.5 | **251.116 ± 1.617 / 242.899** | **7.447** |
| Disagreement-aware KD 0.5 | 在 teacher 分歧高的位置降低 KD 权重 | 252.396 ± 1.871 / 243.086 | 6.166 |

Plain KD 0.5 相对 Normal 在 validation 上 10/10 seeds 获胜；aware KD 比 plain KD 变差 1.280 MW。高分歧区域同时可能是 ensemble 平均最有价值的区域，简单把分歧当作 teacher 不可信度会丢失信号。

同一物理 GPU 的顺序基准：Direct 0.1195、Normal 0.3637、KD 0.3425 ms/origin。单 KD 相对顺序运行 10 个 Normal 的估计约快 10.62×，但这不是实际并行 ensemble serving benchmark，也不是等精度比较。KD 与 Normal 参数完全相同，所以不应称为结构压缩。

---

## 10. 科学问题如何从旧故事演进到跨分辨率故事

### 10.1 已被实验否定或需要降级的旧说法

1. **“天气低频、历史高频”应撤回。** DCT 只限制联合 correction，且 DCT-7/8/12 没有稳定超过全频。
2. **“状态必须显式控制天气取点/取时”应撤回。** Conditioned query 与 attention bias 没有提升，attention bias 还显著更差。
3. **“乘法就是已识别的物理交互”不成立。** 乘法只有弱点估计优势，显式分支不可唯一解释。
4. **“learned 4×4 空间采样优于 bilinear”尚未成立。** 现有区间跨 0，且 baseline 不是原始变量 bilinear。
5. **“一个 KD 学生已无损替代 ensemble”尚未成立。** KD 稳定提高单模型，但仍存在约 8.2 MW 的 single–ensemble gap。
6. **“复杂区域图或 composition 是必要模块”不成立。** 目前只有小的 ensemble 探索信号，没有稳定单模型证据。

### 10.2 现有证据真正指向的科学问题

第一性原理上，未来功率由两个因素共同决定：

- 历史功率是当前潜在运行状态的观测代理；
- 未来 NWP 是起报之后的外部强迫代理。

真实困难不在于给两者指定固定频带，而在于它们来自不同坐标系和分辨率：

- 历史状态：15 min、场站点位、只存在于过去；
- 未来强迫：小时、规则网格、存在于未来；
- 输出目标：15 min、场站点位、未来 6 h。

因此建议把科学问题正式写成：

> **如何在缺少场站级、15 min 未来气象真值的条件下，学习一个由近期运行状态约束的跨时空分辨率响应算子，把小时规则网格 NWP 转换为不规则风电场站的 15 min 功率演化？**

对应的数学对象不是气象重建器 \(W_{1h}\to \widehat W_{15min}\)，而是功率响应算子：

\[
\mathcal F:\big(H_{15min,\,farm},W_{1h,\,grid},\tau\big)
\longrightarrow Y_{15min,\,farm}.
\]

### 10.3 推荐技术路线

整条路线可在框架图中画成：

```text
过去 6 h、15 min 场站功率 H
├─ 完整历史基线 B(H) ──────────────────────────────────────┐
└─ 最近 1 h 集群状态 S_recent ── state encoder ────────┐    │
                                                        │    │
小时规则网格 NWP W                                      │    │
└─ 逐点气象 MLP                                         │    │
   └─ 固定 4×4 候选、内容自适应空间汇聚                 │    │
      └─ 8 个场站小时 token + 相邻小时差分              │    │
         └─ 真实相对时间 cross-attention                │    │
            └─ 24 个目标时刻特定 latent forcing ───────┤    │
                                                        │    │
目标 lead / 起报相位 / 场站 embedding ──────────────────┤    │
                                                        ▼    │
                                      unified correction C   │
                                                        │    │
                                                        └─ + ┘
                                                           │
                                              56 场站 × 24 步
                                                           │
                                                容量加权集群输出
```

这条路线的三个核心模块分别解决：

1. **空间支持错配：**规则网格到不规则场站的局地适配；
2. **时间采样错配：**8 个原生小时节点到 24 个目标时刻特定的 latent response；
3. **响应非平稳性：**同一未来天气在不同 operating level 和近期轨迹下对应不同功率修正。

KD 位于训练层，不是跨分辨率响应算子的组成部分：它利用同构 ensemble 共识提高单模型稳定性和部署性能。

### 10.4 推荐术语与主张强度

推荐使用：

- `state-conditioned native-resolution forcing adaptation`；
- `coarse-forcing-to-fine-power response operator`；
- `target-time-conditioned latent forcing response`；
- 中文可用“近期状态条件化的原生分辨率未来强迫适配”。

暂不使用：

- weather super-resolution / 气象超分辨率；
- reconstructed 15-min weather / 重建 15 min 气象；
- physics-identified interaction / 已识别物理交互；
- adaptive spatial sampling outperforms interpolation；
- history-high-frequency / weather-low-frequency decomposition。

### 10.5 与仓库内 Tyan-WP/PAMF 的边界

根据仓库 `thesis/Tyan-WP A Wind Power Foundation Model for Ultra-Short-Term.pdf` 中的方法定义，PAMF 的中心也是让功率状态与气象表示发生适配，因此二者在“气象—功率不能无条件拼接”这一高层思想上存在相似性。仅仅再增加一个 gate 或把 black-box MLP 改名，不能形成可靠区别。

当前路线真正可发展的区别在研究对象：PAMF 主要处理对齐后的历史功率与历史气象特征融合，而这里面对的是**未来小时网格 NWP 与未来 15 min 场站功率之间的异步、异空间支持映射**。当前模型显式包含 grid-to-farm 局地适配、8 小时节点到 24 个目标时刻的相对时间查询，以及历史基线加未来强迫 correction 的职责结构。

不过，这一差别要成为方法贡献，必须由后续 T0–T4 时间插值基线和空间 same-support 矩阵支撑；否则它目前只是一种更合适的问题表述，而不是已经证明优于 PAMF 类融合模块的技术结论。

---

## 11. 若把跨分辨率作为论文主线，还缺哪些实验

### 11.1 最高优先级：时间跨分辨率强基线

当前最大的论证缺口是没有回答：“复杂 target-time attention 是否真的优于常规插值？”应冻结同一 downstream correction 和参数预算，做 10-seed、2024 test 前先在 validation 冻结选择的比较矩阵。

| 编号 | 时间适配方式 | 要回答的问题 |
|---|---|---|
| T0 | **原始 14 变量先按有效时刻线性插值到 24 个 15 min 时刻，再做逐点/场站编码** | 最强常规工程 baseline；当前完全缺失 |
| T1 | 先编码小时 token，再在 latent space 按上下相邻小时做固定线性插值 | 区分收益来自非线性气象编码还是动态时间查询 |
| T2 | 只读取上下两个 bracketing token，用参数匹配 MLP | 检验局部跨分辨率映射是否足够 |
| T3 | nearest-hour | 已有结果，可复用为低复杂度下界 |
| T4 | 当前 full-path relative-time attention | 主模型 |

必须保证相同空间映射、状态输入、baseline、训练预算和监督，只改变时间适配。除总体 MAE外，至少报告整点目标、非整点目标、短中长时效三段和每个 lead。

### 11.2 证明模型真的使用了“连续相对时间”

当前 relative-time bias 存在，但还没有单独证明其价值。需要：

- 删除 \(\Delta t,|\Delta t|\) bias，只保留内容 attention；
- 在样本内打乱 NWP 时间戳但保持 token 内容顺序，或打乱相对时间而不打乱内容；
- 按 NWP age 15/30/45/60 min 分组；
- 分别比较恰落在 NWP 有效时刻附近和位于两个小时节点之间的目标。

如果 current model 只在非整点和较大 NWP age 下显著优于插值，跨时间分辨率故事会比单一总体 MAE更有解释力。

### 11.3 稀疏节点与缺失节点稳健性

用同一 checkpoint 或重新训练的匹配协议测试：

- 8 个小时节点变成每 2 h 一个节点；
- 随机缺失一个/两个中间节点；
- 只保留前半段或后半段路径；
- 节点时间发生 ±15/30 min 偏移。

这能验证模型学习的是跨分辨率响应，而不是依赖固定 8 节点模板。

### 11.4 空间 same-support 公平矩阵

当前 learned 4×4 与 fixed bilinear 的支持范围不同：前者 16 点，后者 4 点，所以即使出现差异也无法区分“更多邻域”与“自适应权重”。至少补齐：

| 候选范围 | 固定权重 | 学习权重 |
|---|---|---|
| 2×2 四角 | 原始/latent bilinear | learned 2×2 attention |
| 4×4 十六点 | 固定距离核或预定义权重 | 当前 learned 4×4 attention |

再增加两个顺序对照：

1. **raw bilinear -> point encoder：**先对 14 个原始气象变量双线性插值到场站，再编码；
2. **point encoder -> latent bilinear：**当前 fixed bilinear 实现。

这样才能区分非线性编码与插值是否可交换、learned 权重是否真正有价值。

### 11.5 4×4 固定范围与自适应范围

固定 4×4 目前只有工程直觉，没有系统消融。应比较 2×2、4×4、6×6 或基于公里半径的候选范围，并记录：

- 总体与逐场站 test MAE；
- 近海拔/复杂地形场站和普通场站差异；
- attention 熵、有效支持点数和距离分布；
- 训练时间与显存。

只有 4×4 在多 seed 上稳定优于 2×2且 6×6不再提升，才能说固定范围有实验支撑。若希望做真正“自适应范围”，建议先做稀疏 top-k 或多尺度 local/regional gate，但它应建立在上述 same-support 结果之后，而不是直接增加复杂图网络。

### 11.6 空间适配的监督边界

当前训练目标只有功率，没有场站未来实测气象。因此 learned spatial weights 只保证对最终功率预测有用，无法证明它们恢复了真实局地风场。

- 若能获得场站测风塔/机舱风速或再分析场站点真值，应增加空间映射本身的 wind speed/direction MAE；
- 若没有，只能把该模块称为 `forecast-oriented spatial adaptation`，并避免把 attention map 当成真实物理影响范围。

### 11.7 当前最终状态与空间状态证据未完全对齐

验证集 2×2 已显示“场站最近 1 h 轨迹”比“集群最近 1 h 轨迹”有约 1.455 MW 单模型、1.939 MW ensemble 的稳定增量，但当前 KD 使用后者。若跨空间分辨率成为主线，应做一个最小正式比较：

- 当前 aggregate-recent KD；
- station-recent KD；
- station-recent + 参数匹配 self-duplicate；
- 最有希望但仍属候选的 weather-coherent-K4 KD。

先用 3 seeds screening，再对通过门槛者做 10 seeds；最终必须以单模型为主，不要只按 ensemble 点估计选择。若 station-recent KD 没有复现，就保留当前 aggregate recent，并把空间功率共享明确降级。

### 11.8 新时间或新区域 pristine holdout

现有 2024 test 已用于多轮结构验证，严格意义上不再是长期未触碰数据。最终论文最好新增：

- 2025 或之后的时间外 holdout；或
- 未参与训练的区域/场站 holdout；或
- 不同 NWP cycle/系统的迁移测试。

跨分辨率算子的核心价值应体现在对 NWP age、季节、场站位置或 NWP 系统变化的稳健性，而不只是同一区域逐年随机性。

### 11.9 概率、不确定性与 extreme-ramp 不是当前主缺口

仓库已有 ramp MAE、ramp direction accuracy、extreme-ramp precision/recall、功率/ramp regime 和案例图。它们适合补强应用价值，但不会替代 T0–T4 与空间 same-support 对照。优先顺序应是：

1. 证明跨时间分辨率映射优于合理插值；
2. 证明 learned 空间映射是否优于同支持固定映射；
3. 再扩展概率预测、极端 ramp 或更复杂专家模型。

---

## 12. Horizon specialist 当前状态：不可写成正式结论

现有模型的季度 blocked cross-validation 中，最佳单一模型 KDNoAnchor 为 242.102 MW，三时段 convex blend 为 241.547 MW，只改善 0.555 MW，没有达到预注册的总体 1 MW或任一时段 2 MW门槛，因此高成本双专家训练已停止。

3-seed screening：

- L1 anchor taper：只让 anchor 在 3 h 内保持 0.2、3–4 h 退火、4 h 后为 0；相对 Normal 单模型改善 1.168 MW，3/3 wins，通过筛选；
- L2 nearest-to-full fixed view + scalar anchor：总体恶化 0.530 MW，短时恶化 4.085 MW，失败；
- L3 fixed view + tapered anchor：总体改善 1.510 MW，但短时恶化 2.060 MW，失败。

截至本文档更新时间，L1 的 10-seed formal 仍在运行，只有部分 seed 完成，不能升级为正式结果。这一阶段只说明：显式 nearest-to-full 天气视图切换没有通过筛选，horizon-dependent anchor 值得等待完整结果。它不改变当前主结论。

---

## 13. 证据等级与论文可写结论

| 结论 | 当前证据等级 | 论文表述建议 |
|---|---|---|
| 未来 NWP 在完整历史基线之外有大预测增量 | 强：严格四格 + test 天气置乱 | 可作为机制前提；注明是预测依赖，不是物理因果 |
| 近期状态主要帮助解释天气到功率的条件映射 | 强 validation 机制证据，test 结构间接支持 | 可作为核心机制表述 |
| operating level、趋势和最近 1 h 轨迹是主要状态 | 强 validation 机制证据 | 可解释为何 correction 只需 compact recent state |
| \(B+C\) 优于 Direct 不是单纯参数量效应 | 中强：d72 容量控制 + test | 可作为结构依据，不称唯一因果来源 |
| 小时节点间演化编码有用 | 强 test 证据 | 可作为时间跨分辨率模块贡献 |
| 完整路径主要服务中长时效 | 中强 test 证据 | 可写 horizon 分析 |
| target-time attention 优于原始线性时间插值 | 未验证 | 必须补 T0–T4 |
| learned 4×4 空间聚合优于 bilinear | 未验证 | 不能写成贡献，需 same-support 矩阵 |
| 场站近期轨迹优于集群近期轨迹 | 中强 validation 证据 | 可作为待整合发现，尚非最终 test 模型结论 |
| weather-coherent 邻居有效 | 弱到中：ensemble 有信号、seedwise 不稳 | 只作未来方向 |
| 显式乘法、FiLM 或 attention control 是主机制 | 不支持 | 不再作为主线 |
| KD 提高单模型精度与稳定性 | 强 validation + test | 可作为训练策略贡献 |
| 单 KD 无损替代 ensemble | 不支持 | 不得宣称 |

---

## 14. 建议的完整论文故事线

### 14.1 问题动机

超短期风电预测并不是普通的同分辨率多变量时序预测。场站功率以 15 min 记录在不规则点位上，未来 NWP 以小时提供在规则网格上，而目标又是场站级 15 min 功率。直接拼接迫使一个网络同时解决状态估计、空间位置错配、时间节点错配和功率响应，容易产生职责混淆与训练方差。

### 14.2 科学假设

未来 NWP 是外部强迫，近期功率是当前运行状态代理。相同的小时网格天气路径在不同 operating level 和近期趋势下对应不同的细粒度功率响应。因此需要学习的不是高分辨率气象本身，而是状态条件化的 coarse-forcing-to-fine-power response。

### 14.3 方法

1. 完整历史分支 \(B(H)\) 提供连续性和历史基线；
2. 逐点编码与场站局地聚合把规则网格 NWP 变为场站小时 token；
3. 相邻小时差分编码天气系统演化；
4. 基于真实相对时间的 target query 为每个 15 min 目标生成 latent forcing response；
5. 最近 1 h 状态与该 response 在统一 correction 中融合；
6. 弱 anchor 保持 baseline/correction 职责；
7. training-only ensemble consensus KD 降低单模型训练方差。

### 14.4 实验证据链

1. DCT 消融排除固定频带分工；
2. 严格四格证明天气是主增量、状态产生条件收益；
3. 冻结状态阶梯将状态压缩到 level + trend + recent trajectory；
4. 显式 fusion/horizon 矩阵表明无需复杂控制算子；
5. test 上 Direct、FullHistory、Recent-state、KD 确认总体性能；
6. test 时间消融支持 inter-hour path 和中长时完整天气路径；
7. KD 在 test 上 10/10 优于同结构未蒸馏模型；
8. 空间适配和跨场站机制保持克制，明确列出尚未完成的强对照。

### 14.5 最稳妥的贡献表述

现阶段可以写成三项：

1. 将异步、异空间支持的风电预测重新表述为“原生未来强迫到细粒度功率响应”的跨分辨率问题，而不是气象超分辨率；
2. 通过严格四格与冻结状态阶梯识别出“未来天气主增量 + compact recent-state 条件化”的预测机制，并据此构建 \(B(H)+C(S_{recent},W,\tau)\)；
3. 用同构 ensemble consensus KD 稳定这一结构，在 2024 test 上显著提高单模型 MAE、RMSE、\(R^2\) 和 ramp 指标。

在补齐 T0–T4 之前，不建议把“新型时间插值器”列为独立贡献；在补齐空间 same-support 矩阵之前，不建议把 learned spatial sampler 列为贡献。

---

## 15. 关键产物位置

- 2024 test 主结果与消融：`artifacts/paper_ablation_completion_v1/analysis_test/`
- 2024 test 图、扩展指标与场站结果：`artifacts/paper_evidence_test_v1/`
- DCT 与 correction 初始机制：`artifacts/incremental_mechanism_v2/`
- 严格四格、anchor 和初始化控制：`artifacts/conditional_forcing_validation_v1/`
- 冻结状态阶梯与空间×轨迹 2×2：`artifacts/state_forcing_ladder_v1/`
- 显式 interaction：`artifacts/explicit_interaction_matrix_v1/`
- horizon operator：`artifacts/horizon_control_matrix_v1/`
- ensemble/KD：`artifacts/ensemble_stabilization_v1/`
- 跨场站近期状态共享：`artifacts/cross_farm_context_v1/`
- 区域 composition：`artifacts/regional_composition_v1/`
- horizon specialist 进行中结果：`artifacts/horizon_specialist_v1/`
- 当前 KD 配置示例：`artifacts/ensemble_stabilization_v1/groups/kd_050/seed42/config.yaml`
- 数据形状与预处理声明：`data/model_cache_pad5_f8/manifest.json`
- 当前核心实现：`src/wind_forcing/models/components.py`、`src/wind_forcing/models/forced_evolution.py`、`src/wind_forcing/models/incremental_correction.py`、`src/wind_forcing/training.py`

---

## 16. 最终统一判断

当前工作已经从“尝试某种 fusion”推进到一个有实验依据的问题：**近期运行状态如何约束原生小时网格未来强迫到 15 min 场站功率的响应。**这个问题具有清楚的物理动机和工程价值，也与简单的同分辨率气象—功率融合不同。

但跨分辨率故事的成熟度是不对称的：

- **时间侧：**已有真实相对时间、小时差分路径、nearest/full path 和辅助监督的 test 消融，具备成为主线的基础；
- **空间侧：**当前模块实现合理，但 learned 4×4 相对 bilinear 的优越性未成立，固定 4×4 也没有完整范围消融；
- **状态侧：**机制证据最完整，但当前最终模型尚未吸收场站近期轨迹的小幅稳定增量；
- **稳定化侧：**KD 的单模型收益最可靠，但它应定位为训练与部署层，不要取代跨分辨率科学问题。

因此最合理的下一步不是继续搜索新 fusion 算子，而是补齐**原始时间插值强基线、连续相对时间消融、空间 same-support 矩阵和 station-recent KD**。这些实验一旦成立，故事就能从“状态条件化 correction”自然升级为“有机制证据支撑的跨分辨率未来强迫响应建模”；如果它们不成立，也能及时把跨分辨率降级为问题背景，而不损害现有状态条件化与 KD 的可靠结论。
