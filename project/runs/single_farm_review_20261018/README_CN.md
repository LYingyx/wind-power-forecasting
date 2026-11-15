# 单风场方向重审交付

先读 [REVIEW_CN.md](REVIEW_CN.md)。建议主线为“历史轨迹条件化天气修正的单风场预测”，允许跨站训练；将本站部署与区域历史辅助明确分开，迁移作为适配收益及边界实验。

另有 [结果章节 PDF 预览](RESULTS_CHAPTER_PREVIEW_CN.pdf)，已通过独立 XeLaTeX 编译；它是修订候选章节，不是已同步重写全文的投稿版本。

## 关键表格

| 内容 | 入口 |
|---|---|
| 全年有效缓存：全部共享 baseline、local-only CBH、区域版本 E2C | [paper_source_main.csv](tables/paper_source_main.csv) |
| 独立单站 pilot；不可与全年主表混排 | [independent_baselines_pilot_only.md](tables/independent_baselines_pilot_only.md) |
| 56 站完整逐站指标 | [source_all56_per_farm.csv](tables/source_all56_per_farm.csv) |
| 6 个源代表站 | [paper_source_representatives_2024.md](tables/paper_source_representatives_2024.md) |
| 四臂条件消融 / 实际字段输入检查 | [condition_ablation_fp16.md](tables/condition_ablation_fp16.md) / [condition_actual_inputs.md](tables/condition_actual_inputs.md) |
| 天气编码器与因子控制 | [weather_ablation_fp32.md](tables/weather_ablation_fp32.md) / [weather_factorial_contrasts.md](tables/weather_factorial_contrasts.md) |
| 共享参数 × 在线区域输入 | [sharing_2x2.md](tables/sharing_2x2.md) |
| direct/correction 输出锚点 | [matched_output_fp16.md](tables/matched_output_fp16.md) |
| 分场景结果 | [scenario_summary.md](tables/scenario_summary.md) / [scenario_e2c_minus_e0.md](tables/scenario_e2c_minus_e0.md) |
| 20 目标站迁移主表 | [paper_target_learning_curve.md](tables/paper_target_learning_curve.md) |
| 迁移的配对差与区间 | [target20_paired_contrasts.md](tables/target20_paired_contrasts.md) |
| 全部 15 个目标站 scratch baseline | [target20_scratch_all10seeds.md](tables/target20_scratch_all10seeds.md) |
| 4 个目标代表站 | [paper_target_representatives.md](tables/paper_target_representatives.md) |
| 有效 origin 与等周统计的敏感性 | [source_e2c_week_bootstrap.md](tables/source_e2c_week_bootstrap.md) / [equal_week_sensitivity.md](tables/equal_week_sensitivity.md) |

主要汇总表均有同名 `.csv`、`.md`、`.tex`；大型明细只输出 CSV。缺失值为空，不代表零误差。所有 MAE 采用 CF 单位，0.001 相当于额定容量的 0.1 个百分点；不是相对改善 0.1%。

## 图件

每幅图均有同名矢量 PDF 和 PNG。

1. [训练期特征选站](figures/01_farm_selection.png)
2. [源站 baseline 对比](figures/02_source_baseline_comparison.png)
3. [24 个预测 lead](figures/03_forecast_horizons.png)
4. [季节、时段、运行状态](figures/04_scenario_contrasts.png)
5. [迁移学习曲线与强预训练对照](figures/05_transfer_learning_curve.png)
6. [6 个源代表站：普通/历史爬坡案例](figures/06_source_forecast_examples.png)
7. [龙湖失败边界案例](figures/07_failure_forecast_examples.png)
8. [4 个目标代表站的学习曲线](figures/08_target_medoid_learning_curves.png)

场景图是 E2C 对 E0 的区域历史版本比较；不是尚未执行的 local-only 最终模型实验。示例曲线的 Transformer 为本次 FP32 CPU 推理，仅作描述，不替换正式历史 BF16 分数。

## 复跑

使用当前统一项目自带环境，无需安装包。命令按顺序运行；原始数据只读，输出覆盖本目录的派生表/图。最后两步分别生成图表与统计权重敏感性。

```bash
cd /shared_data_ssd2/liying/wind_cluster_project_20261018/project
.runtime/chronos/bin/python runs/single_farm_review_20261018/scripts/audit_tables.py
.runtime/chronos/bin/python runs/single_farm_review_20261018/scripts/audit_condition_inputs.py
.runtime/chronos/bin/python runs/single_farm_review_20261018/scripts/analyze_predictions.py
.runtime/chronos/bin/python runs/single_farm_review_20261018/scripts/predict_case_baseline.py
.runtime/chronos/bin/python runs/single_farm_review_20261018/scripts/finalize_review.py
.runtime/chronos/bin/python runs/single_farm_review_20261018/scripts/audit_interval_weighting.py
```

`analyze_predictions.py` 需读取数 GiB 归档，会持续打印进度。主指标是各单模型误差的 seed 均值，不是先集成预测再算误差。

来源：[table_sources.json](table_sources.json)。核验：[audit_status.json](audit_status.json)、[prediction_verification.json](prediction_verification.json)、[case_inference.json](case_inference.json)、[final_verification.json](final_verification.json)。

补实验：[NEXT_EXPERIMENTS_CN.md](NEXT_EXPERIMENTS_CN.md)。本轮完成的是审查、重汇总、预测复算与少量案例推理，没有启动新训练。旧论文需按本报告同步任务、模型版本和结果；不能把 E2C 数值直接放进仍描述 SLAST 的正文。
