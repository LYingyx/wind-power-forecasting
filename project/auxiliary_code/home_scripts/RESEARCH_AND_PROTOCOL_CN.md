# 潜在高频代理天气接入原 CBH：研究设计与口径

## 研究问题

本批不把不存在的 15 分钟天气当成监督标签。已知量始终是每个发行周期的 7 个小时 NWP 节点和 14 个站点天气变量；模型只在相邻小时之间生成 3 个潜在中间状态，再用未来 24 个 15 分钟功率标签训练天气到功率的映射。中间状态的可识别性来自下游功率目标、小时锚点、跨尺度小时遮盖任务和模型先验，而不是来自真实 15 分钟天气。

因此主要问题是：**受小时天气约束的潜在 future-covariate adapter，能否提高未来 6 小时、24 个 15 分钟点的单站功率预测？** 代理天气首先按下游功率 MAE 评价。4h→1h 与 2h→1h 的天气遮盖结果只用于检查正则和尺度迁移行为，不能用来声称恢复了真实 15 分钟气象过程。

## 引入生成路线前的正式模型

交接文档确定的主结构是 **CBH：History-supported baseline + conditional future-weather correction**，不是旧探索目录里的 `P0I_CBH_SINGLE_*`：

1. history-only ForcedEvolution 从全 56 站历史得到未来轨迹 `b`；该轨迹在 correction 阶段冻结；
2. 显式构造 `Δb`，并保留完整历史功率 Transformer memory；
3. 小时 NWP 经 P0I 时间插值和天气投影；
4. condition projection 同时读取历史 summary、`b`、`Δb` 与 lead；
5. transfer adapter 融合 weather 与 condition，Transformer decoder 读取完整历史 memory；
6. readout 给出 correction，最终输出为 `b + correction`。

正式原结构对照 `CBH_ORIGINAL_56_SUBSET` 直接加载 seeds 42–51 的原 CBH checkpoint，在固定 10 站和完全相同的 2023 report 窗口上推理。`CBHSelected` 则从每个 checkpoint 抽取一个目标站实际使用的同一组模块、站点静态向量和条件偏置，并继续使用原 56 站 history-only baseline cache。100 个 station×seed 初始化前向的最大绝对差为 `3.1769276e-5`；这是 56 站批量图与单站 seed-vmap 图的 FP32 矩阵核差异，远小于预测误差尺度，训练 guard 固定为 `4e-5`。

## 生成后连接的模型

正式生成路径为：

`小时 NWP(7×14) → 相邻小时内 missing-only 代理生成器 → 15min covariates(24×14) → 原 CBH weather projection / condition adapter / Transformer decoder → b + correction → 功率(24)`

这条路径没有使用每帧单一 CLS token。当前输入在进入实验前已经从天气格点固定提取为 14 个物理变量；先压成一个 CLS 会丢掉变量级锚点，并使“整点逐变量不变”难以逐值审计。若以后直接使用完整空间 patch，适合保留多个 patch token，以全局 token 作为条件，而不是把 CLS 当成唯一需要生成的天气状态。

所有小时节点直接旁路生成器；生成器只输出 6 个小时区间各自的 3×14 个 missing residual。最终按实际 valid-time offset 查询 24 个目标位置，已知节点从原输入写回。随机生成器保存 5 个样本，下游点预测固定取 **5 个功率预测的中位数**，不是先平均天气再预测功率。

## 被比较的生成方法

所有方法共用同一 128 宽度条件 MLP field、相同端点条件、相同 residual scale、预训练预算和联合训练预算。下列实现是面向本任务的受控适配，不冒充对应论文的完整复现。

| 方法 | 实际改变 | 采样/作用 |
|---|---|---|
| `LINEAR` | 物理 valid time 上直接线性插值 | 不含生成器；也是同初始化、同单站微调的主控制 |
| `DET` | 从零状态直接回归相对线性插值的残差 | 1 次前向，检查随机路径是否必要 |
| `ICFM` | paired linear conditional flow，`x_t=(1-t)z+t y`，目标 `y-z` | 8 步 Heun；对应 I-CFM/初次 rectified-flow 类目标 |
| `VPCFM` | 三角 VP probability path，速度目标随 `sin/cos` 变化 | 与 I-CFM 区分 probability path |
| `SINE_SI` | 线性 path 加 `0.3 sin(πt)ε` 的 stochastic interpolant | 显式比较中间随机路径 |
| `VP_EPS` | VP forward path，epsilon 参数化 | 与 `VP_V` 共用 path，隔离训练 target 差异 |
| `VP_V` | 同一 VP path，v 参数化 | 避免把 sampler 差异误叫成新扩散模型 |
| `EDM` | Karras 预条件、log-normal sigma 训练、rho=7 schedule | 8 个 sigma 区间，Heun，17 NFE |
| `MEANFLOW` | 用 JVP identity 学区间平均速度，target stop-gradient | 1 NFE，检查少步生成的代价 |

DDIM 是已训练扩散模型的采样方式，因此没有被单列为训练模型。OT-CFM 会在 minibatch 中重配 source/target；这里每个 residual 已绑定站点、发行周期、物理时间和小时端点，任意跨条件 OT 会破坏语义，故没有为了凑名称而运行。SB-CFM、I²SB、DDBM 和 TimeBridge 的关键价值是数据依赖 prior/paired bridge；在没有真实 15 分钟目标时，可在未来作为更强的潜变量先验，但当前不能把 EDM 或 I-CFM 结果改名为 bridge 结果。

## 自监督代理与联合训练

生成器先在 2022 pilot 上做 4h→1h 人工遮盖：以第 0/4、1/5、2/6 个小时节点作为三组端点，预测每组中间的 3 个已观测小时节点。这提供跨尺度 residual 结构，不提供 1h→15min 真值。随后从原 CBH checkpoint 初始化 correction 路径，在同一单站 pilot 上最小化：

`power MAE + 0.1 × weather proxy loss`

功率 MAE 穿过可微 sampler、天气投影、condition adapter 和 decoder 回传至生成器。history-only ForcedEvolution 的 `b` 仍来自缓存并保持冻结。`LINEAR` 也从同一 CBH checkpoint 初始化并微调同一 correction 路径，所以“生成方法 − LINEAR”是隔离生成天气选择的首要比较。

## 站点、数据与统计

10 站为 `1043019, 1043010, 1043036, 1043007, 1043028, 1043033, 1043011, 1043058, 1043015, 1043061`，对应原 56 站索引 `17, 8, 34, 5, 26, 31, 9, 53, 13, 55`。筛选只使用 2022 的覆盖率、容量检查、非恒定信号、同期天气功率相关和地理/容量多样性；不使用爬坡、功率差分、验证集性能或生成模型结果。

正式 pilot 使用 2022 训练和 2023 validation/report，约为每月 3 个 hash 固定 UTC 日期。每站每模型运行 seeds 42–51。checkpoint 由独立 `val_stop` 子集的最低 CF MAE 选择，`report` 子集含 1632 个窗口；统计先在同一 seed 内对 10 站等权，再报告 10 seeds 的均值、样本 SD、最低/最高及 seed。配对差使用 station×seed 的 20,000 次 crossed bootstrap，并在 seed 级 10 站等权差上做 Wilcoxon；预列出的 17 个比较统一做 Holm 校正。

原 CBH 使用完整 2022 数据训练，其 checkpoint 曾用完整 2023 validation 选模；后续 adapter 又在 2022 pilot 上微调。因此 `CBH_ADAPT_* − CBH_ORIGINAL_56_SUBSET` 同时含单站 pilot 适配与天气代理变化，且 2023 report 不是盲测。生成方法之间与 `CBH_ADAPT_LINEAR` 的比较共享初始化、数据和选模协议，内部归因更干净。

## 已有命名 baseline 的位置

上一隔离批次已经完成以下现有模型，每项均为 10 站×10 seeds：

- 历史功率：DLinear、PatchTST；
- 天气 only：scikit-learn `HistGradientBoostingRegressor(loss='absolute_error')`；
- 历史功率 + 未来已知天气：TiDE、TFT、NBEATSx；
- TiDE 后接 DET、I-CFM、VP-CFM、Sine-SI、VP-epsilon、VP-v、EDM、MeanFlow。

它们的完整结果保存在 `temporal_sr_named_models_20260917_v2`。本目录会复制其聚合摘要作为背景，但不把 TiDE pilot 与完整 2022 训练的原 CBH 绝对差归因给某一个结构组件。旧 `P0I_CBH_SINGLE_*` 使用 PatchTST 替换 ForcedEvolution 且令 `Δb=0`，只保留为简化控制，不代表交接文档确定的原 CBH。

## 文献边界

CSDI 提供 observed/missing mask 的自监督条件扩散范式；TimeBridge 提供 spline/data-dependent prior 与 point-preserving sampler；I²SB/DDBM 提供 paired bridge；TorchCFM 区分 I-CFM、OT-CFM 和 SB-CFM；Stochastic Interpolants 统一了带中间噪声的路径；EDM 系统化了连续 sigma 预条件和 solver；MeanFlow 针对 1-NFE 平均速度。CorrDiff、AFM、PC-AFM 和近期天气 flow-matching 工作主要有真实高分辨率空间目标，不能直接证明本项目潜在 15 分钟状态的气象真实性。

这些文献对本任务最可迁移的结论是：只更新 missing residual、用数据相关先验缩短路径、硬写回观测点，并用下游任务与概率样本共同评价。没有 15 分钟天气真值不妨碍研究潜在代理天气对功率预测的价值，但会限制“恢复真实天气”“频谱正确”或“物理极值可靠”等表述。

论文与官方实现快照、版本和 SHA256 已保存在 `temporal_sr_named_models_20260917_v2/references` 与 `vendor`；详细方法综述见该目录的 `analysis/LITERATURE_SYNTHESIS_CN.md` 和 `analysis/METHOD_MATRIX_CN.md`。
