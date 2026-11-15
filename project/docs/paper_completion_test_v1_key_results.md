# Paper completion v1：held-out test 关键结果

## 协议与完成性

- 划分：2022 train、2023 validation、2024 test；checkpoint 只由 validation 选择。
- 正式统计：seed 42--51，共 10 个随机种子；每个 test 产物包含 28,782 个共同 forecast origin、24 个 lead（0.25--6 h）。
- 新增消融：11 组 × 10 seeds = 110/110 完成；每项均有 `metrics_test.json`、`predictions_test.npz` 和 `TEST_COMPLETE`。
- 稳健性：weather permutation 100/100 完成。
- 可视化：314 张 PNG 和 314 张 PDF，全部来自 test split。
- 全量日志扫描：无 Traceback、RuntimeError 或 CUDA OOM。

## 主结果

| 模型 | 单模型 cluster MAE，mean ± sd (MW) | 10-seed ensemble MAE (MW) |
|---|---:|---:|
| Direct | 262.034 ± 2.530 | 250.389 |
| FullHistory | 257.323 ± 2.901 | 242.594 |
| Normal | 257.805 ± 2.361 | 244.687 |
| KD | **250.966 ± 3.184** | 242.755 |

按 UTC 日做 10,000 次 block bootstrap 后：

- KD 相对 Direct：ensemble 降低 7.634 MW，95% CI [-10.995, -4.183]；单模型均值降低 11.068 MW，10/10 seeds 获胜。
- KD 相对 Normal：ensemble 降低 1.932 MW，95% CI [-3.247, -0.629]；单模型均值降低 6.838 MW，10/10 seeds 获胜。
- KD 相对 FullHistory：ensemble 差值为 +0.161 MW，95% CI [-1.764, 2.205]，不能区分；但单模型均值降低 6.356 MW，9/10 seeds 获胜。

核心定位：KD 的可靠价值是显著提高单模型精度和跨 seed 稳定性；KD 的十模型 ensemble 与 FullHistory 十模型 ensemble 相当。单个 KD 的平均 MAE 仍高于 Normal/FullHistory ensemble，因此当前证据不支持“一个 KD 已无损压缩 ensemble”，也不支持宣称 KD ensemble 显著超过 FullHistory ensemble。

## 结构消融

差值均为 ablation/替代方案减去 Normal；正值表示变差。`fixed_bilinear` 使用干净对照 `no_wind_geometry`。

| 变体 | ensemble 差值 (MW) | 95% CI (MW) | Holm p（seedwise） | 结论 |
|---|---:|---:|---:|---|
| independent forcing path | +7.317 | [3.942, 10.971] | 5.93e-5 | 受状态调制的 forcing path 必要 |
| no auxiliary weather loss | +2.188 | [1.062, 3.282] | 0.0479 | 辅助气象监督必要 |
| frozen history baseline | +7.340 | [5.028, 9.662] | 4.39e-4 | baseline 与修正项必须联合适配 |
| nearest-hour query | +1.061 | [-0.684, 2.734] | 1.000 | 总体无可靠差异 |
| no wind geometry | -0.702 | [-2.072, 0.619] | 0.530 | 无法证明 wind geometry 必要 |
| fixed bilinear vs no geometry | +1.101 | [-0.585, 2.788] | 1.000 | 无法证明学习式空间取点优于固定双线性 |
| no anchor | -0.843 | [-2.694, 0.992] | 1.000 | anchor 的总体收益不成立 |

Horizon 结果揭示了总体均值掩盖的作用边界：

- nearest-hour 在 0.25--1 h 改善 0.599 MW，但在 3.25--6 h 变差 2.891 MW；完整未来气象路径主要服务长时域。
- independent forcing 在 1.25--3 h、3.25--6 h 分别变差 7.211、9.657 MW；状态调制的收益集中在中长时域。
- 去掉辅助气象损失在 3.25--6 h 变差 3.297 MW；冻结 baseline 在该区间变差 12.710 MW。
- 去掉 anchor 在 0.25--1 h 和 1.25--3 h 分别变差 1.294、1.718 MW，却在 3.25--6 h 改善 3.262 MW。因此 anchor 更像 horizon-dependent bias controller，不应表述为全时域必需组件。

由此，最稳固的结构故事是“状态调制气象作用 + 气象辅助监督 + 联合适配”，而不是 wind geometry、学习式空间采样或统一强 anchor。

## KD 目标消融

- `kd_station_only`、`kd_cluster_only`、`kd_no_aux_weather`、`kd_no_anchor` 均未同时通过 seedwise、ensemble bootstrap 和 Holm 校正的严格判据。
- `kd_cluster_only` 的单模型均值比 KD 差 3.232 MW，10/10 seeds 均更差，Holm p=0.00163；但 ensemble 差值仅 +0.219 MW，95% CI [-0.650, 1.038]。这支持“station target 有助于单模型稳定性”，但不足以宣称其提高 ensemble 上限。
- `kd_no_anchor` 数值上 ensemble 改善 1.132 MW，整体 CI [-2.599, 0.316]；只在 3.25--6 h 有明确改善（2.621 MW）。可作为后续 horizon-adaptive anchor 的候选，不应替换当前默认配置。
- 当前应保留联合 station + cluster distillation 作为稳妥默认，但不把具体 KD target 拆分写成已被完全证实的独立贡献。

## 天气依赖与效率

- Weather permutation 后，Normal 的 MAE 平均恶化 287.918 MW，KD 恶化 299.749 MW；这排除了模型忽略天气、只复制历史状态的退化解释。该试验证明依赖性，不等价于因果识别。
- 同一物理 GPU 7 上顺序基准：Direct 0.1195、Normal 0.3637、KD 0.3425 ms/origin。KD 与 Normal 参数量相同（1,287,343），不应宣称结构压缩。
- 顺序执行 10 个 Normal 成员的估计延迟为 3.6367 ms/origin；单个 KD 相对该估计约 10.62× 加速。此值是 sequential estimate，不是实际 ensemble serving benchmark，也不是等精度比较：单 KD 仍未达到十模型 ensemble 的 MAE。

## 最终结论

当前证据支持的论文主线是：历史状态不是单独提供“高频”，天气也不是只提供“低频”；有效机制是天气作用受近期状态调制，并通过联合训练在中长预测时域逐步接管。KD 能把教师信息转化为更准确、更稳定的单模型，但尚未无损达到 ensemble 上限。空间几何、学习式取点和固定 anchor 只能作为实现选择或后续方向，不能作为当前主要创新点。

## 产物

- 消融统计：`artifacts/paper_ablation_completion_v1/analysis_test/`
- 主模型配对比较：`artifacts/paper_ablation_completion_v1/analysis_test/reference_comparisons.json`
- test 图表与表格：`artifacts/paper_evidence_test_v1/cluster/`、`artifacts/paper_evidence_test_v1/station/`
- weather permutation：`artifacts/paper_evidence_test_v1/weather_robustness_analysis/`
- 同卡效率结果：`artifacts/paper_evidence_test_v1/efficiency/`
