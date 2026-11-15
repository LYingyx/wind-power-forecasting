# Aggregate-only 集群功率预测关键结果记录（最终版）

更新时间：2026-09-03 21:40 CST

## 1. 完成状态与结论

- 整套 staged pipeline 已于 2026-09-03 17:07 CST 完成，并生成根级 `artifacts/aggregate_cluster_v1/COMPLETE`。
- 共完成并审计 207 个独立训练 checkpoint、150 个 2024 frozen-test 推理任务。
- R2状态补充矩阵于21:36完成，新增20个训练checkpoint、20个frozen-test推理和20个correction-state干预推理，均已审计。
- 严格状态—天气调制矩阵随后完成87个训练checkpoint、60个frozen-test标记、全局状态干预、零天气不变性和45格regime审计；根级 `artifacts/aggregate_strict_modulation_v1/COMPLETE` 已生成。
- 没有 KD、ensemble 或跨 seed 预测平均；表中均为 10 个独立 checkpoint 的 `mean ± SD`。
- 没有 OOM、Traceback 或无效完成标记。
- 2022 用于训练，2023 用于全部结构选择，2024 在结构冻结后统一评估。由于2024曾被既往工作分析，本文称其为“内部时间留出”，不称 pristine holdout。

纳入严格调制及同预算训练控制后，当前最终选择为：

\[
\boxed{
\text{3 h aggregate-history TCN baseline}
+\text{weather-driven residual correction}
+\text{R2 latent weather interpolation}
+\text{low-LR continued training (A0-FT)}
}
\]

最终A0-FT的10-seed平均MAE为：

- 2023 validation：`273.195 ± 2.435 MW`。
- 2024 内部时间留出：`270.325 ± 3.949 MW`。

它在结构选择用的validation上超过当前最强aggregate fusion baseline、R2-Full和旧R2-W。A0-FT不增加状态分支，只从同seed旧R2-W嵌套初始化，并在相同联合微调预算下以低学习率继续优化天气解码器。

## 2. 最终架构是什么

R2 对未来天气采用以下顺序：

1. 保留8个原生小时 NWP 节点，不先伪造15 min原始天气。
2. 每个小时节点的区域天气场先经过共享 Regional CNN，得到任务相关的空间 latent grid。
3. 使用由56个场站容量和坐标离线形成的 `capacity_density`，把latent grid池化成集群尺度的小时天气表示。
4. 在这些集群相关latent表示之间线性插值，得到24个15 min目标时刻的天气特征。
5. 过去3 h集群总功率通过共享TCN形成自主预测基线 \(B(H)\)。correction只读取天气latent和预测时距，不再重复读取当前功率。
6. 正式checkpoint从同seed R2-W嵌套初始化，再执行预注册的低学习率继续训练；模型拓扑仍为无额外状态修正路径的 `B(H)+C(W)`。

即：

\[
\hat P_{\Sigma,1:24}
=B(H_{3h})+C(\operatorname{Interp}(\operatorname{Pool}(\operatorname{CNN}(W))),\tau).
\]

这个结构的核心不是复杂MoE或全路径attention，而是正确安排跨分辨率处理顺序：

> 先在原生小时天气节点上提取与集群功率相关的非线性表示，再在任务latent空间完成小时到15 min的时间适配。

## 3. 10-seed主结果

### 3.1 2023 validation

| 类别 | 方法 | MAE / MW | RMSE / MW | \(R^2\) | Ramp MAE / MW |
|---|---|---:|---:|---:|---:|
| History-only | TCN-3h | 412.962 ± 0.834 | 611.160 ± 1.751 | 0.7632 ± 0.0014 | 61.131 ± 0.051 |
| Weather-only | W-Only | 372.493 ± 5.427 | 502.657 ± 6.473 | 0.8398 ± 0.0041 | 71.148 ± 0.416 |
| Fusion baseline | PredictionBlend | **293.217 ± 5.215** | 415.253 ± 8.498 | 0.8907 ± 0.0045 | **57.718 ± 0.738** |
| Fusion baseline | TFT | 294.138 ± 5.521 | 413.953 ± 8.192 | 0.8913 ± 0.0043 | 58.230 ± 0.803 |
| Fusion baseline | TargetConcat | 294.696 ± 4.209 | 413.464 ± 4.679 | 0.8916 ± 0.0025 | 58.166 ± 1.254 |
| Reference route | R3 Full-path | 280.738 ± 3.621 | 400.359 ± 5.198 | 0.8984 ± 0.0026 | 55.966 ± 0.457 |
| Candidate route | R1 Raw interpolation | 276.605 ± 2.396 | **394.736 ± 3.961** | **0.9012 ± 0.0020** | 58.413 ± 0.485 |
| R2 control | R2-Full state-weather | 276.114 ± 3.276 | 395.107 ± 6.170 | 0.9010 ± 0.0031 | 58.292 ± 0.274 |
| Base route | R2-W Latent interpolation | 275.363 ± 2.599 | 395.438 ± 3.816 | 0.9010 ± 0.0019 | 60.479 ± 0.642 |
| **Final route** | **A0-FT (R2-W continued training)** | **273.195 ± 2.435** | **391.839 ± 3.625** | **0.903 ± 0.002** | 60.102 ± 0.385 |

### 3.2 2024内部时间留出

| 类别 | 方法 | MAE / MW | RMSE / MW | \(R^2\) | Ramp MAE / MW |
|---|---|---:|---:|---:|---:|
| History-only | TCN-3h | 395.983 ± 0.929 | 590.380 ± 1.404 | 0.7596 ± 0.0011 | 57.979 ± 0.061 |
| Weather-only | W-Only | 369.500 ± 5.198 | 513.521 ± 8.683 | 0.8181 ± 0.0062 | 68.545 ± 0.307 |
| Fusion baseline | PredictionBlend | 289.344 ± 5.232 | 420.369 ± 10.683 | 0.8781 ± 0.0063 | **55.269 ± 0.626** |
| Fusion baseline | TFT | **288.727 ± 6.148** | 414.703 ± 8.294 | 0.8813 ± 0.0048 | 55.698 ± 0.829 |
| Fusion baseline | TargetConcat | 291.903 ± 2.161 | 420.787 ± 5.436 | 0.8779 ± 0.0032 | 55.713 ± 1.207 |
| Reference route | R3 Full-path | 276.284 ± 3.703 | 403.475 ± 5.792 | 0.8877 ± 0.0032 | **53.446 ± 0.356** |
| Candidate route | R1 Raw interpolation | 272.242 ± 4.466 | 396.265 ± 6.130 | 0.8917 ± 0.0033 | 56.088 ± 0.488 |
| R2 control | R2-Full state-weather | **270.330 ± 3.411** | **394.281 ± 6.578** | **0.8928 ± 0.0036** | 55.975 ± 0.335 |
| Base route | R2-W Latent interpolation | 272.717 ± 4.578 | 397.189 ± 7.017 | 0.8912 ± 0.0039 | 58.194 ± 0.736 |
| **Frozen final route** | **A0-FT (R2-W continued training)** | **270.325 ± 3.949** | **393.077 ± 6.612** | **0.893 ± 0.004** | 57.818 ± 0.465 |

## 4. 最终模型具体超过了多少

本文按2023 validation冻结baseline和最终架构，不根据2024重新选择。

| 比较 | Validation ΔMAE | 2024 ΔMAE | 说明 |
|---|---:|---:|---|
| A0-FT − PredictionBlend | **−20.022 MW** | **−19.019 MW** | 相对validation冻结的最强fusion baseline |
| A0-FT − TFT | −20.943 MW | **−18.402 MW** | TFT是2024数值最好的fusion baseline |
| A0-FT − R3 Full-path | **−7.543 MW** | **−5.959 MW** | latent插值与后续低学习率优化的共同差值 |
| A0-FT − R1 Raw interpolation | **−3.410 MW** | **−1.917 MW** | 两个时间切分均更低 |
| A0-FT − old R2-W | **−2.169 MW** | **−2.392 MW** | 同拓扑、同seed嵌套继续训练收益 |

负值表示A0-FT更好。相对PredictionBlend，A0-FT validation MAE降低约6.83%，2024降低约6.57%。

逐时距相对PredictionBlend的改善也不是由单一时段贡献：

| 时距 | Validation ΔMAE | 2024 ΔMAE |
|---|---:|---:|
| 0–1 h | −20.425 MW | −21.468 MW |
| 1–3 h | −24.948 MW | −25.882 MW |
| 3–6 h | −16.604 MW | −13.628 MW |

既有12个功率、历史变化、历史波动和未来风速变化分层针对R2-Full；其相对PredictionBlend的平均MAE差值全部为负，validation改善范围为 `12.860–23.604 MW`，2024为 `10.703–25.356 MW`。A0-FT尚未重复这套旧分层，因此不把这组数值冒充最终模型的分层结果。

需要保留的代价边界是：A0-FT主要优化总体轨迹MAE，其Ramp MAE没有优于所有baseline。validation上A0-FT比PredictionBlend高 `2.384 MW`，2024高 `2.549 MW`。因此不能声称所有辅助指标同时最优。

## 5. 历史范围与状态范围

### 5.1 历史基线

| 方法 | Validation MAE / MW | 2024 MAE / MW |
|---|---:|---:|
| TCN-3h | 412.962 ± 0.834 | 395.983 ± 0.929 |
| TCN-6h | **412.839 ± 0.498** | **395.669 ± 0.593** |

6 h只比3 h降低 `0.123 MW`（validation）和 `0.314 MW`（2024）。这个量级不足以抵消额外历史范围，因此最终模型保留3 h。结合3-seed的TCN-1h `416.405 MW`，更合理的解释是：集群历史从1 h扩展到3 h有实际价值，再扩展到6 h趋于饱和。

### 5.2 Correction状态

| 状态输入 | Validation MAE / MW | 2024 MAE / MW |
|---|---:|---:|
| Current | **280.908 ± 2.802** | **274.437 ± 3.421** |
| Recent 1h | 281.585 ± 2.710 | 274.865 ± 3.303 |

只用当前功率分别低 `0.677 MW` 和 `0.428 MW`。因此历史基线和天气修正采用不同信息范围：

\[
H_B=\text{过去3 h},\qquad S_C=\text{当前功率}.
\]

这使故事线更清楚：较长历史负责构造集群自主演化基线，修正分支只需要最小当前运行状态。

## 6. 天气架构探索得到的主线

3-seed筛选结果为：

| 路线 | 天气处理方式 | MAE / MW |
|---|---|---:|
| R1 | raw天气先插值，再CNN编码 | **275.338 ± 1.433** |
| R2 | 每个小时先CNN编码和容量池化，再latent插值 | 277.190 ± 2.690 |
| R3 | 容量池化后读取完整8节点路径 | 280.279 ± 3.558 |
| R4 | 目标时间控制节点value | 281.902 ± 1.844 |
| R5 | 动态区域空间池化 | 294.171 ± 0.612 |
| R6 | 每小时区域语义token | 298.866 ± 5.849 |
| R7 | 空间patch Transformer | 293.309 ± 0.938 |
| R8 | 联合时空patch查询 | 300.054 ± 4.579 |

R1和带状态的R2-Full补齐10 seeds后次序反转，R2-Full低 `0.491 MW`（validation）和 `1.912 MW`（2024）。进一步移除correction状态得到R2-W后，其validation又降低 `0.751 MW`。因此最终仍采用R2天气路线，并把分辨率转换放到集群任务相关表示中；是否保留状态则由专门消融独立决定。

R4–R8没有显示复杂时间控制、动态空间注意力或patch Transformer的优势。这些失败结果支持一个简洁结论：对于直接集群总功率预测，固定容量支持和局部CNN已经足够；更复杂的空间token化会增加优化难度，但没有转化成更低误差。

## 7. 状态—天气机制边界

该机制矩阵固定使用R3天气路线，因此用于解释R3，不应直接替代R2的专门消融。

| 机制 | Validation MAE / MW | 2024 MAE / MW |
|---|---:|---:|
| M0：基线 + 空修正 | 413.277 ± 1.063 | 396.354 ± 1.084 |
| MS：基线 + 状态修正 | 413.403 ± 1.119 | 396.614 ± 1.096 |
| MW：基线 + 天气修正 | **276.636 ± 4.053** | **273.117 ± 5.468** |
| MFull：R3状态—天气black-box修正 | 280.738 ± 3.621 | 276.284 ± 3.703 |
| MAdd：状态修正 + 天气修正 | 282.271 ± 3.781 | 276.468 ± 3.968 |

可稳妥使用的结论是：

1. 天气是中长时改善的主要来源；M0和MS接近history-only，而MW大幅降低误差。
2. 在R3上加入当前状态并没有超过MW：MFull比MW高 `4.102 MW`（validation）和 `3.167 MW`（2024）。
3. Black-box融合只比MAdd低 `1.533 MW`（validation）和 `0.184 MW`（2024），不适合把“状态调制天气”写成核心创新。
4. 当前最完整的论文故事应围绕“集群历史基线 + 原生小时天气的latent跨分辨率适配”，而不是围绕复杂状态—天气interaction。

R2专门消融现已完成。R2-W在validation比R2-Full低 `0.751 MW`，但2024高 `2.387 MW`；StateShift在两个split都使Full恶化约 `2.61 MW`。这说明Full确实利用了状态，但状态增量不能跨时间切分稳定复现。最终依照validation冻结R2-W，并把状态作用限定在3 h历史基线中。

## 8. 计算代价

| 方法 | Trainable params | Profiled GFLOPs/origin | 单origin延迟 | Peak MiB |
|---|---:|---:|---:|---:|
| PredictionBlend | 393,560 | 0.251 | 12.455 ms | 17.0 |
| R1 | 412,335 | 0.725 | 15.592 ms | 30.7 |
| R2-Full / 参数匹配R2-W | 412,335 | **0.253** | 19.139 ms | 17.7 |
| R3 | 412,335 | 0.254 | 17.243 ms | 17.7 |

R2比R1减少约65%的profiler计数FLOPs和约42%的峰值显存，但当前单样本实测延迟反而比R1高约3.55 ms，说明小算子调度开销仍存在。论文可以报告计算量优势，但不应把当前R2描述成延迟最优模型。

## 9. 最终论文出口

当前最顺畅的主线是：

> 集群未来6 h功率既不能只依赖历史惯性，也不能只依赖未来天气。过去约3 h的集群功率形成自主基线；天气保留在原生小时节点上完成非线性空间编码和容量支持聚合，再在任务latent空间适配到15 min预测时刻。严格调制实验进一步表明，状态路由会被网络使用，却不如同预算继续优化固定天气响应。因此最终采用简洁的 `B(H_3h)+C(W_latent,τ)`，并以嵌套低学习率继续训练得到A0-FT。该单模型在10个独立随机种子上超过所有重新训练的aggregate fusion baseline。

核心数值结论：

\[
\boxed{
\text{A0-FT validation MAE}=273.195\text{ MW},\quad
\Delta\text{vs best fusion}=-20.022\text{ MW}
}
\]

\[
\boxed{
\text{A0-FT 2024 MAE}=270.325\text{ MW},\quad
\Delta\text{vs frozen best fusion}=-19.019\text{ MW}
}
\]

原始矩阵保存在 `artifacts/aggregate_cluster_v1/analysis/`；R2状态补充结果保存在 `artifacts/aggregate_cluster_r2_state_v1/analysis/`；最终严格调制与训练控制保存在 `artifacts/aggregate_strict_modulation_v1/`。完整结果表位于 `exp_docs/direct_cluster_forecasting_experiment_results.md`。

<!-- R2_STATE_MECHANISM_START -->
## R2 correction状态专门消融（10 seeds）

- 本阶段按2023 validation平均MAE选择 **R2-W**；后续严格调制阶段最终选择更新为同拓扑的A0-FT。
- 相对PredictionBlend：validation `-17.853 MW`，2024 `-16.627 MW`。
- R2-Full − R2-W：validation `+0.751 MW`，2024 `-2.387 MW`。
- R2-Full − R2-Add：validation `-4.536 MW`。
- StateShift退化：validation `+2.611 MW`，2024 `+2.614 MW`。
- 最终处理：删除修正分支的重复状态输入，采用更简洁的 B(H)+C(W) 结构。
- Full/W在validation和2024方向反转，因此不把correction状态调制作为稳健核心贡献。
- 历史基线固定为过去3 h TCN，不再扩展历史窗口或多尺度状态搜索。
- R2-W运行时跳过未激活的state encoder，但保留参数键以兼容正式checkpoint。

完整记录见 `exp_docs/aggregate_cluster_r2_state_mechanism_results.md`。
<!-- R2_STATE_MECHANISM_END -->

<!-- STRICT_STATE_WEATHER_START -->
## 严格状态—天气调制（10 seeds）

- validation冻结结果：**A0-FT（R2-W低学习率继续训练）**。
- 最佳严格候选 E1 相对R2-W：validation `-0.730 MW`，2024 `-0.252 MW`。
- E1 相对A0-FT：validation `+1.439 MW`，2024 `+2.140 MW`；严格调制没有形成净增益。
- A0-FT相对旧R2-W：validation `-2.169 MW`，2024 `-2.392 MW`。
- E1相对旧R2-W的Ramp MAE差值：validation `-0.546 MW`，2024 `-0.477 MW`；这不是相对最终A0-FT的收益。
- 全局置乱状态的MAE变化：validation `+9.005 MW`，2024 `+10.468 MW`。
- 零天气latent不变性最大误差 `0.000e+00`；严格排除了state-only correction捷径。
- 完整记录：`exp_docs/aggregate_strict_state_weather_modulation_results.md`。
<!-- STRICT_STATE_WEATHER_END -->
