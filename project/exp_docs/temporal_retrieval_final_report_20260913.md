# 天气输入参数化与检索对照：最终实验报告（机制解释已更正）

> **2026-09-13 代码核查更正：** P0I 单 key softmax 恒为 1，天气读取不依赖 query；其多层线性天气投影可合并为与 A1 同形式的仿射输入层。十 seed 性能指标保留，但此前“已证明 target-conditioned weather projection/retrieval”的结论撤回。实际 checkpoint 的等价性检查误差约 3.55×10⁻¹⁵。详见[核查与动态方案可行性评估](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/dynamic_response_feasibility_20260913/FEASIBILITY_CN.md)。

**分析对象。** 本报告合并 `temporal_retrieval_20260913` 的 3-seed screening 和 `temporal_retrieval_confirmation_20260913` 的十 seed confirmation。训练、验证和测试均已完成，没有失败运行。confirmation 包含 P0I、P1I、N0、N1 四个模型，共 40 个 seed 运行、80 份 val/test 预测文件。

## 最终结论

P0I 相对 A1 的稳定性能收益已确认；现有对照尚未识别其具体原因。P0I 在输入端使用分开的 context/weather 投影、天气矩阵因子化与零初始化输出层，但天气分支本身不做 query-dependent retrieval，且输入映射的函数类与 A1 相同。不能由两者的性能差异推出检索机制或表示能力优势。

P0I 是可以冻结的实用候选：P1I、N1 与它的十 seed 差异未达到统计可区分性，可作为敏感性对照；这不等于已经通过等价性检验证明三者等价。P2I/N2 的 global retrieval 未通过验证集 screening，因此没有证据要求扩大该方向。现有实验也没有直接检验独立 native-weather trajectory encoder 是否有效。

## 十 seed confirmation 的正式结果

主指标为 56 场站等权 CF MAE。CF MSE/RMSE/R² 同样在所有场站、origin 和 24 个 forecast lead 上 pooled 计算；MW MAE/RMSE 是单场站物理功率误差的平均，不是总功率 MAE。所有数值是十个单模型 seed 指标的平均，不是 ensemble。

| 模型 | 2023 CF MAE | 2024 CF MAE | 2024 CF MSE | 2024 CF RMSE | 2024 pooled R² | 2024 逐站平均 R² | 2024 MW MAE / MW RMSE | 越界比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 clean Transformer | 0.084744 | 0.082737 | 0.017497 | 0.132275 | 0.703872 | 0.675667 | 9.920 / 19.071 | 4.15% |
| **P0I** 插值 pointwise | **0.084205** | **0.081970** | **0.017121** | **0.130849** | **0.710228** | **0.682958** | **9.823 / 18.851** | 5.35% |
| P1I 插值 local-2 | 0.084187 | 0.082032 | 0.017151 | 0.130961 | 0.709732 | 0.682401 | 9.828 / 18.854 | 5.45% |
| N0 native pointwise | 0.084283 | 0.082199 | 0.017220 | 0.131225 | 0.708558 | 0.681167 | 9.846 / 18.874 | 4.19% |
| N1 native local-2 | 0.084201 | 0.082058 | 0.017142 | 0.130925 | 0.709888 | 0.682554 | 9.828 / 18.833 | 5.47% |

相对 A1，P0I 的验证 CF MAE 减少 0.000540，即约 0.637%；2024 减少 0.000767，即约 0.927%。2024 单站平均 MW MAE 减少约 0.097 MW（97 kW），CF MSE 减少约 2.15%，pooled R² 增加约 0.00636。

P1I、N0、N1 也都在十个 seed 中相对 A1 的验证 MAE **10/10 胜出**。P0I、P1I、N0、N1 的验证和测试 paired bootstrap 95% CI 均为负；完整 CI 保存在 `confirmation_paired_metrics.csv`。

| 比较 | 数据集 | Δ CF MAE | 胜出 | seed bootstrap 95% CI | paired-t 95% CI |
|---|---|---:|---:|---:|---:|
| P0I − A1 | 2023 val | −0.000540 | 10/10 | [−0.000711, −0.000365] | [−0.000751, −0.000328] |
| P0I − A1 | 2024 test | −0.000767 | 10/10 | [−0.000944, −0.000596] | [−0.000979, −0.000556] |
| P1I − A1 | 2023 val | −0.000558 | 10/10 | [−0.000704, −0.000409] | [−0.000736, −0.000379] |
| P1I − A1 | 2024 test | −0.000705 | 10/10 | [−0.000901, −0.000524] | [−0.000935, −0.000475] |
| N0 − A1 | 2023 val | −0.000462 | 10/10 | [−0.000629, −0.000308] | [−0.000657, −0.000266] |
| N0 − A1 | 2024 test | −0.000539 | 10/10 | [−0.000764, −0.000329] | [−0.000803, −0.000274] |
| N1 − A1 | 2023 val | −0.000544 | 10/10 | [−0.000713, −0.000372] | [−0.000751, −0.000337] |
| N1 − A1 | 2024 test | −0.000679 | 10/10 | [−0.000877, −0.000488] | [−0.000916, −0.000442] |

P0I、P1I、N1 之间没有可靠的性能排序。P1I−P0I 的验证 Δ 为 −0.000018、测试 Δ 为 +0.000062；N1−P0I 的验证 Δ 为 −0.000004、测试 Δ 为 +0.000088，所有区间均跨零。N0 相对 P0I 的验证 Δ 为 +0.000078，2024 为 +0.000229，说明 native pointwise 不如插值 pointwise。

N1 相对 N0 有小幅局部读取收益：验证 Δ=−0.000082，2024 Δ=−0.000141；但幅度远小于相对 A1 的主收益，因此只能作为弱支持性结果，不能把 local retrieval 定义为主创新。

## 时距结果

A1 的 2023/2024 horizon CF MAE 分别为 0.048823/0.045753（0–1 h）、0.082491/0.079729（1–3 h）和 0.098221/0.097071（3–6 h）。四个 confirmation 模型都保持同样趋势：改善在 1–6 h 更明显，0–1 h 变化较小。这支持“未来 forcing representation 在远期更重要”的经验观察，但没有证明具体的物理因果路径。

P0I 的 horizon 结果为：

| 数据集 | 0–1 h | 1–3 h | 3–6 h |
|---|---:|---:|---:|
| 2023 P0I | 0.048706 | 0.082000 | 0.097507 |
| 2024 P0I | 0.045582 | 0.079035 | 0.096055 |
| 2023 相对 A1 | −0.000117 | −0.000491 | −0.000714 |
| 2024 相对 A1 | −0.000171 | −0.000694 | −0.001016 |

P0I 的绝对收益主要出现在 1–6 h。N1、P1I 的时距曲线相近，没有显示 global retrieval 才能获得的特殊远期增益。

## 真实结构和机制解释

A1 是：

\[
\bar W_\tau=Interp(W_{1:M},\tau),\qquad
q_\tau=E([\bar W_\tau,c_\tau]).
\]

P0I 改为：

\[
q_\tau=E_c(c_\tau),\qquad
F_\tau=q_\tau+Attn(q_\tau,E_w(\bar W_\tau),E_w(\bar W_\tau)),
\]

其中 attention mask 强制每个 target 只读取自己的插值 weather token。随后仍使用原来的 history encoder、Transformer decoder 和 persistence residual：

\[
M_H=Encoder(H),\qquad
X=Decoder(F,M_H),\qquad
\hat{CF}_\tau=CF_t+Head(X_\tau).
\]

但 pointwise mask 下，每个 target 只剩一个可用 key，softmax 权重恒为 1。因此上式可以化简为：

\[
F_\tau=E_c(c_\tau)+M_W\bar W_\tau+d_W.
\]

这里没有 query-dependent 的天气权重，连续的天气线性投影也没有新增非线性表达能力。完整未来 token 可合并为与 A1 的 `Linear([weather, context])` 相同形式的仿射层。target context 仍进入后续 decoder，但当前结果只确认了所训练 P0I 参数化的性能收益，不能区分因子化、初始化和优化过程各自的贡献。

P1I 允许每个 target 读取最近两个插值 token；N1 允许读取最近两个 native token；P2I/N2 则分别允许全部 24/7 个 token。screening 中 P2I/N2 均未通过 2023 验证门槛，说明 global retrieval 没有可确认的增量。

H0/H1/H2 的 screen 结果为：

| 变体 | 2023 CF MAE | 2024 CF MAE |
|---|---:|---:|
| H0 forcing-only | 0.110018 | 0.105610 |
| H1 当前 strict additive | 0.093117 | 0.089245 |
| H2 full decoder | 0.084237 | 0.081842 |

H2 接近 A6，但 H1 的历史支路是 pooled-history MLP，weather 支路也是轻量 readout，没有和完整 decoder 做表达能力匹配。因此 H2−H1 不能被解释为已经证明的 history–forcing non-separability。后续若还要验证 interaction，必须重新实现 lead-specific、完整历史轨迹的 additive control，并匹配参数、训练预算和天气 direct path。

## 已验证和未验证的逻辑链

| 命题 | 判定 |
|---|---|
| 所训练 P0I/P1I/N0/N1 相对 A1 的性能收益 | **已确认，四个结构十 seed 均稳定改善** |
| P0I 证明 target-conditioned weather retrieval | **撤回**；单 key 权重恒为 1，天气读取与 query 无关 |
| 改善来自第八个 NWP 节点 | **不成立**；N0/N1 只用七个 native 节点仍改善 |
| 改善主要是 endpoint clamp 修复 | **不成立**；上一轮四种 origin phase 均改善 |
| native representation 显著优于 interpolated representation | **未确认**；N1 与 P0I 十 seed 差异跨零，2024 N1 反而略差 |
| local retrieval 必须优于 pointwise | **未确认**；P1I 与 P0I 无显著差异，N1 相对 N0 只有小幅收益 |
| global future trajectory retrieval 更好 | **未支持**；P2I/N2 未通过 validation screening，不能解释成证明其永远无效 |
| full decoder 证明了不可分离 history–forcing interaction | **未确认**；H1 additive control 表达能力不足 |
| forcing variability 越大，retrieval gain 越大 | **未确认**；此前仅有支持性趋势，未通过严格验证 |

## 预测审计和限制

confirmation 的 80 份 val/test 预测均通过 sample index、origin_unix、目标形状和有限性检查。反标准化目标与缓存 CF 的最大差异为约 (5.8\times10^{-8})，指标复算和原始导出指标一致到浮点误差范围。

P0I 的越界比例为验证 4.98%、测试 5.35%，高于 A1 的约 4.08%/4.15%；N0 的越界比例较低，N1 较高。所有主结果都使用未裁剪预测，没有把边界后处理混入模型比较。后续若研究 [0,1] bounded output，应对 A1 和候选模型统一使用同一后处理规则。

2024 已在此前多轮实验中查看，因此应称跨年复评，不能称全流程盲测。十 seed 降低了随机初始化不确定性，但不能消除多轮 validation-based architecture search 的选择偏差。模型改进幅度约为 0.5%–0.9% CF MAE，属于稳定但有限的改进。

## 定稿建议

当前建议冻结 **P0I** 作为论文主模型，P1I 和 N1 作为 representation sensitivity ablations。选择 P0I 是因为它是 pointwise target-query 中最简单的版本；P0I、P1I、N1 十 seed 的性能差异没有统计可区分性，不应人为声称某一个明显胜出。

当前可以报告的结论是：

> 对于相同历史与天气输入，采用分开的 context/weather 投影和因子化天气输入路径训练的 P0I，在当前协议下稳定优于 A1；但该输入映射可合并为与 A1 同形式的仿射层，收益原因仍需通过参数化、初始化与优化对照识别。

这足以作为性能事实，尚不足以单独支撑“目标条件化检索”这一方法创新。也不能声称 native-resolution weather 优于插值表示、global retrieval 必要，或新的模型已经证明 history–weather 非可分离物理响应。

完整数据文件：

- [confirmation aggregate metrics](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_retrieval_confirmation_20260913/confirmation_aggregate.csv)
- [confirmation paired metrics and CI](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_retrieval_confirmation_20260913/confirmation_paired_metrics.csv)
- [verified prediction audit](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_retrieval_confirmation_20260913/confirmation_verified.csv)
- [screening analysis](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_retrieval_20260913/SCREEN_ANALYSIS_CN.md)
- [run directory](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_retrieval_confirmation_20260913)
