# 气象—功率响应状态辨识实验计划

## 1. 要回答的问题

已有严格调制实验已经证明：模型会使用历史功率状态，但只用集群功率或历史预测 latent 仍不能稳定超过无状态天气模型。因此本轮不再搜索新的 gate、expert 或 FiLM，而是验证一个更基础的问题：

> 当前“天气如何转化成功率”的响应规律，能否从最近已经发生的盘古预报—实际集群功率配对中辨识出来？

任务仍是 aggregate-only 集群预测：输入集群历史功率和区域盘古天气，直接输出未来6 h的24个15 min集群总功率。不使用56场历史，不做KD或ensemble，模型选择只看2023 validation的多seed平均结果。

## 2. 固定不变的部分

- 2022训练、2023选择，全部冻结后才统一评估2024。
- 历史自主基线固定为已验证的3 h TCN。
- 未来天气固定为R2路线：原生小时天气先经过Regional CNN和静态容量池化，再在latent空间线性对齐到24个15 min目标时刻。
- 响应状态只通过rank-4零保持低秩变换调制未来天气latent；若未来天气latent为零，不同状态必须产生完全相同的correction。
- 从同seed完整R2-W checkpoint嵌套初始化。Adapter阶段冻结历史基线、天气编码器和天气解码器；Joint阶段仅以0.1倍学习率微调天气解码器。
- 训练预算与上一轮严格调制实验一致：Adapter最多30 epochs，Joint最多20 epochs，分别使用6/5轮early stopping patience。

## 3. 因果历史配对数据

对每个预测起点，缓存最近6个原生小时节点：

\[
(W^-_1,P^-_1),\ldots,(W^-_6,P^-_6).
\]

每个历史天气节点都满足：

\[
\text{issue time}\le\text{valid time}\le\text{forecast origin}.
\]

历史天气使用当时可获得的盘古预报，而不是事后更晚起报的天气；对应功率为该有效时刻已经实现的集群总功率。缓存位于 `data/aggregate_response_state_v1/`，manifest保存来源指纹、数组校验和及因果时间审计。

历史与未来天气均经过同一个冻结的same-seed R2天气编码器，并保存到 `artifacts/aggregate_response_state_v1/latent_cache/seed*/`。这既保持表示空间一致，也避免每个候选重复运行区域CNN。

## 4. 响应状态模型

六个历史节点形成以下token：

\[
c_k=[u^-_k,\Delta u^-_k,p^-_k,\Delta p^-_k,t_k,a_k],
\]

其中 \(u^-\) 是冻结R2编码器输出，\(t_k\) 是相对预测起点的时间，\(a_k\) 是历史天气预报年龄。统一的32维两层因果TCN将token编码为响应状态 \(r_t\)。所有候选输入宽度和参数量相同，缺失的信息源以零填充。

未来天气latent通过：

\[
\tilde u_q=\left[I+U\operatorname{diag}(g(r_t))V^\top\right]u_q^+,
\]

再由原R2天气解码器得到correction。控制器零初始化，因此训练第0步严格等于same-seed R2-W。

## 5. 实验矩阵

已有参照直接复用，不重训：

| ID | 状态信息 | 用途 |
|---|---|---|
| C0 | 无，现有A0-FT | 同联合训练预算的最强无状态参照 |
| C1 | 3 h历史基线latent，现有D2 | 只从输出历史编码状态的参照 |

新增候选：

| ID | 响应token来源 | 回答的问题 |
|---|---|---|
| C2 | 仅历史功率level与变化 | 只观察输出是否足够 |
| C3 | 仅历史天气level与变化 | 只观察输入是否足够 |
| R1 | 天气—功率level pair | 输入输出配对是否带来增量 |
| R2 | 天气变化—功率变化transition pair | 显式响应关系是否优于level配对 |
| R3 | R2 + support→query辨识损失 | 强迫状态解释新的历史响应是否更可辨识 |

R3把前3个transition作为support，后2个transition作为query；辅助损失权重固定为0.2。未来预测时使用的全部历史节点都位于预测起点之前或等于起点，不构成未来泄漏。

## 6. 分阶段执行与冻结规则

1. 对C2、C3、R1、R2、R3运行seeds 42、46、50的Adapter和Joint。
2. 根据2023 validation的3-seed平均MAE，保留最好的paired候选、最好的single-source候选；若R3获胜，同时保留R2作为无辨识损失控制。
3. 入选组补齐seeds 42–51，仍逐checkpoint独立训练和预测。
4. C0、C1和入选组按2023的10-seed平均MAE排序。只要平均值更低即可选择新模型；精确相同时保留更简单的已有模型。不用显著性门槛。
5. 写入 `selection/validation_freeze.json` 后才创建test latent并评估2024。2024不得用于追加候选或改超参数。

主指标为cluster MAE；同步报告RMSE、R²、Ramp MAE、0–1 h、1–3 h和3–6 h MAE。只汇报独立checkpoint的mean ± SD，不平均预测，不挑最好seed。

## 7. 机制诊断

对validation冻结的最佳响应候选，在validation与2024运行：

- power shuffle：打乱历史实际功率但保持天气与未来任务不变；
- weather shuffle：打乱历史天气但保持功率不变；
- pair shift：错开同一轨迹内天气—功率时间配对；
- response mean：所有样本使用训练集参考响应状态；
- support only：只用前3个transition推断未来响应状态；
- zero weather：将未来天气latent置零后改变响应状态，correction必须保持不变。

另报告当前功率五分位×未来10 m风速下降/稳定/上升×三个预测时段的45格 \(\Delta MAE\)。所有差值均为候选减A0-FT，负值表示响应状态更好，并生成validation和2024热图。

## 8. 调度与产物

启动命令：

```bash
bash shell/launch_aggregate_response_state_v1.sh 0,1,2,3,4,5,6,7 auto
```

- 49 GB GPU每卡3槽，batch为6个weather cycles、梯度累积2。
- 24 GB GPU每卡2槽，batch为3个weather cycles、梯度累积4。
- 两者均保持等效12个weather cycles/update。
- 总任务与每个worker均显式显示tqdm；每个任务使用文件锁和完整产物审计，只有审计通过才写COMPLETE。
- OOM时停止该卡继续领取任务，不静默修改科学配置。

统一产物根目录为 `artifacts/aggregate_response_state_v1/`，最终结果写入 `exp_docs/weather_power_response_state_experiment_results.md`。
