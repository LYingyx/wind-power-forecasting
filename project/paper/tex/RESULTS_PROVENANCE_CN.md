# 手稿证据与图表来源

权威审查目录：`../../runs/single_farm_review_20261018/`。本稿所有数值基于其中已核验的 CSV/预测分析；文件哈希见 `data/single_farm/sources.json`。未用图像生成模型制造数值结果。

| 手稿内容 | 审查目录中的来源 | 解释约束 |
|---|---|---|
| 源站主表 | `tables/paper_source_main.csv` | 全 56 站，全年有效窗口；本站/区域输入分组 |
| 四格比较 | `tables/sharing_2x2.csv`、`sharing_2x2_contrasts.csv` | BF16，SL/IL/SC/IC；RMSE/R² 不补造 |
| 条件消融 | `tables/condition_ablation_fp16.csv`、`condition_actual_inputs.csv` | 新四臂 FP16；CB 含 b 与 Δb |
| 输出形式 | `tables/matched_output_fp16.csv`、`matched_output_contrasts.csv` | 匹配 FP16，微小均值增量区间跨零 |
| 天气编码 | `tables/weather_ablation_fp32.csv`、`weather_factorial_contrasts.csv` | 同组 FP32；归档因子区间为等周敏感性 |
| 区间权重 | `tables/source_e2c_week_bootstrap.csv`、`equal_week_sensitivity.csv` | 有效 origin 与等周是不同统计目标 |
| 六个代表源站 | `tables/paper_source_representatives_2024.csv`、`farm_selection.json` | 训练描述量 PAM，回顾性规则 |
| 场景与时段 | `tables/scenario_summary.csv`、`scenario_e2c_minus_e0.csv` | 仅区域 E2C 对 E0；未来爬坡仅诊断 |
| 迁移 | `tables/paper_target_learning_curve.csv`、`target20_paired_contrasts.csv` | 共同五种子；先站内平均；H1 验证标签未计预算 |
| 完整源站附录 | `tables/source_all56_per_farm.csv` | 全 56 站；不按代表性聚类大小加权 |
| 独立 pilot | `tables/independent_baselines_pilot_only.csv` | 3156/3349 起点，只有 2023，不混排 |
| 八组数值图 | `figures/01_*.pdf` 至 `08_*.pdf` | 原始数据生成，保存为矢量 PDF |

## 实现核对

- 本站参考：`artifacts/cross_farm_shared_specific_20260920_v1/code/train_sl_seed.py` 及其 snapshot `forced_evolution.py`。第二历史编码器保留参数形状、改读本站历史。
- CBH 及条件输入：`artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_transfer.py`。
- 点式天气对齐：同 snapshot 的 `p0i_legacy.py`。先插值，再投影；每个提前量只允许一个天气 token。
- E2C：`artifacts/column_weather_20260928_v1/code/column_model.py` 及 `configs/E2C_seed42.yaml`。memory.mean，原 14 天气 + 高程的可逆坐标，非线性替换 native projection。
- 图 1：`../../output/imagegen/single_farm_framework_v19/`，使用 bundled imagegen CLI，模型请求参数 `gpt-image-2.5`；一轮 generate、两轮 edit。最终按实际计算路径核查，没有程序绘制或覆盖连线。完整 prompt 和生成记录已复制到 `figures/imagegen_provenance/`。

## 结果审计

此前审查重算 40 个 E0/E2C 预测归档，MAE 最大差约 5.4e-10；448 行季节结果最大差约 9.9e-10。案例 Transformer 曲线使用保存 checkpoint 的 CPU FP32 推理，不等同于原 BF16 全年逐样本复评。完整审查状态和来源哈希在审查目录的 `audit_status.json`、`prediction_verification.json`、`table_sources.json` 等文件。
