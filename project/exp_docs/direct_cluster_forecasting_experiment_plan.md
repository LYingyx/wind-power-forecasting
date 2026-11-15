# Aggregate-only 集群功率预测实验计划

## 1. 研究问题与数据边界

本轮研究任务固定为：使用集群过去功率和未来 8 个小时节点的区域 NWP，直接预测未来 6 h、24 个 15 min 集群总功率：

\[
(H_\Sigma,W)\rightarrow \hat P_{\Sigma,1:24}.
\]

- 2022 年训练，2023 年完成历史范围、结构和超参数选择；全部冻结后统一在 2024 年内部时间留出集评估。
- 模型训练和监督均不读取场站历史、场站未来或场站级标签。
- 本轮不做 KD、预测 ensemble，也不和旧场站任务约 242 MW 的结果比较。
- 每个 checkpoint 独立预测；最终只汇报 3-seed 或 10-seed 的 `mean ± SD`，不挑最好 seed。
- 主指标为 cluster MAE/MW，同时报告 RMSE、R²、NMAE、energy MAE、ramp MAE、persistence skill、逐 lead 和三个时距段结果。

## 2. Aggregate 数据集

缓存位于 `data/aggregate_cluster_v1/`，包含：

- `cluster_history_mw [N,24]`、`cluster_future_mw [N,24]`；
- 8 个小时节点的 14 通道区域天气场、样本时间、NWP cycle 映射和天气节点实际时间距离；
- 由 56 个场站容量与坐标离线双线性散射得到的归一化 `capacity_density [24,25]`。

功率以 6827.01 MW 归一化后进入模型，预测产物恢复为 MW。场站信息只用于一次性的容量密度构造，不进入 split 数据或训练 batch。天气数组优先以只读 hard-link 复用原缓存。manifest 保存来源指纹、时间范围、形状、单位与完整性信息。

## 3. 先回答“历史应该看多远”

历史承担两种不同职责，不能预设两者需要相同窗口：

\[
H_B\rightarrow B(H_B),\qquad S_C\rightarrow C(S_C,W,\tau).
\]

### 3.1 历史独立基线 \(B\)

3-seed（42、46、50）筛选 Persistence、DLinear-6h、参数匹配 TCN-1h/3h/6h 和 PatchTST-6h。TCN 始终接收长度 24 的张量，早期不可见位置置零并提供 mask，差分只在可见区间计算，因此参数量相同。

TCN-6h 与 1h/3h 中较优者补齐 seeds 42–51。如果 DLinear 或 PatchTST 在初筛中相对最优 TCN 至少改善 1 MW 且 3/3 获胜，则其中较优者也补齐 10 seeds，作为强 history-only baseline；主模型中的 \(B\) 仍选 TCN。

仅当长窗口相对短窗口平均改善至少 1 MW、至少 7/10 seeds 获胜且 paired 95% CI 不跨零，才认为更早功率有可靠价值。否则在 10-seed 最优均值 0.5 MW 内选择最短窗口。

### 3.2 天气修正状态 \(S_C\)

冻结历史基线和 R3 天气路线后，比较 current、1h、3h、6h、multi-scale 和 horizon-query 六种状态。所有版本使用相同 64 维 token encoder、两层时间编码器和相同参数量；固定窗口只改变 mask，horizon 版本用已有 lead embedding 查询，multi-scale 保留最近 1 h 的 15 min 点并把更早 5 h 聚合为小时 token。

六组先跑 3 seeds。随后补齐 SC-1h、最优固定窗口和满足 3-seed 改善条件的最佳结构化状态，至多三组 10 seeds。长历史或结构化状态只有同时满足以下条件才入选：相对 SC-1h 改善至少 1 MW、至少 7/10 获胜、paired 95% CI 上界小于 0，且任一时距段平均退化不超过 1 MW。若无可靠差异，则在正式候选最低 MAE 的 0.5 MW 内选择最短状态。

最终只冻结一种 \(S_C\)，不做历史状态和天气路线的笛卡尔积。

## 4. Aggregate-only baseline

### 4.1 单一信息源

- 选定的 history-only TCN；
- 通过筛选的最强 DLinear/PatchTST；
- 真正的 `W-Only`：R3 区域 CNN、容量池化和完整天气路径，不实例化历史编码器、不读取历史功率，也不以当前功率为输出起点。

### 4.2 直接 fusion baseline

所有 fusion baseline 使用冻结后的历史输入范围、R3 天气编码、相同 64 维表示、lead embedding 和 trajectory decoder；从随机初始化直接预测绝对功率，不使用 \(B+C\)、history checkpoint 初始化、persistence 输出捷径或 anchor。

比较 EarlyConcat、TargetConcat、LeadGate、PredictionBlend、CrossAttention、Bidirectional、TFT、ASTAFN-inspired 和 TimeXer-inspired 九种方法。先跑 3 seeds，validation 均值最好的三种补齐 10 seeds。

## 5. 主模型与天气路线

主模型固定为：

\[
\hat P_\Sigma=\max\{0,B(H_B)+C(S_C,Z^W,\tau)\}.
\]

\(B\) 从同 seed 的 history-only checkpoint 初始化并联合训练；anchor 权重为 0.2，3 h 内保持、3–4 h 线性衰减、4 h 后为 0。所有路线共享同一个 \(B\)、状态编码器、black-box correction 和训练预算，天气模块统一输出 `[B,24,64]`。

- R1：原始天气插值后 CNN；
- R2：CNN 编码后 latent interpolation；
- R3：容量池化后查询完整 8 节点路径；
- R4：目标时间控制节点 value；
- R5：动态区域空间池化；
- R6：每小时四个区域语义 token；
- R7：patch Transformer 加区域 token；
- R8：联合时空 patch 查询。

匹配控制为 R4-NoTime（相对时间输入置零）和 R8-Factorized（相同 patch、模块数和参数量下先时间后空间读取）。新路线相对 R3 在 3 seeds 上平均改善至少 1 MW、3/3 获胜且任一时距段不恶化超过 1 MW，才进入 10-seed；R4/R8 还必须通过对应匹配控制。最多保留两条新路线。

## 6. 状态—天气作用边界

在 R3 和最终 \(S_C\) 上比较：

- M0：\(B+C(\varnothing)\)；
- MS：\(B+C(S_C)\)；
- MW：\(B+C(W)\)；
- MFull：\(B+C(S_C,W)\)，复用 R3；
- MAdd：\(B+C_S(S_C)+C_W(W)\)。

MFull 优于 MW 只说明状态在历史基线外仍有增量；MFull 优于 MS 说明天气有额外增量；只有 MFull 进一步优于参数匹配的 MAdd，才支持“历史状态调制天气响应”，否则仅表述为信息互补。

## 7. 分阶段执行与冻结规则

1. **Stage 0**：审计 aggregate 缓存，跑完整单测、全架构 2-epoch smoke 和显存测试，记录代码树 SHA256。正式运行开始后不修改实现。
2. **Stage 1**：完成历史 baseline 的 3-seed/10-seed 选择，再完成 correction state 的 3-seed/10-seed 选择，冻结 \(H_B\) 和 \(S_C\)。
3. **Stage 2**：对 W-Only、九种 fusion、R1–R8 和机制矩阵做 3-seed 筛选，并只运行满足条件的匹配控制。
4. **Stage 3**：对 W-Only、最强三种 fusion、R3、至多两条新天气路线、五组机制及入选控制补齐 seeds 42–51。
5. **Stage 4**：根据 2023 validation 冻结全部方法后，在 2024 上统一推理；不得据此追加变体。

最终主模型只和 Stage 3 中最强的新 aggregate fusion baseline 比较。

## 8. 调度、审计与产物

统一启动命令：

```bash
bash shell/launch_aggregate_cluster_v1.sh 0,1,2,3,4,5,6,7
```

调度器按可用显存分配任务：49 GB 卡优先承担 R1、R7、R8、TFT、TimeXer 和匹配控制，24 GB 卡承担历史、状态、W-Only、CNN 路线和轻量 fusion；一卡一个进程，梯度累积保持每次更新等效 12 个 weather cycles。终端同时显示全局任务 tqdm 和每卡 train/validation tqdm。断点续跑只有在 `COMPLETE` 标记及 checkpoint、metrics、prediction 完整审计均通过时才复用。

产物统一位于：

- `artifacts/aggregate_cluster_v1/`：配置、checkpoint、validation/test 预测、指标、选择记录、代码指纹与汇总表；
- `exp_docs/direct_cluster_forecasting_experiment_plan.md`：本计划；
- `exp_docs/direct_cluster_forecasting_experiment_results.md`：最终精简结论。

结果文档只保留历史范围选择、3-seed 完整筛选均值、10-seed `mean ± SD` 主表、相对最强 aggregate baseline 的平均差值、机制边界、失败方向和最终推荐。
