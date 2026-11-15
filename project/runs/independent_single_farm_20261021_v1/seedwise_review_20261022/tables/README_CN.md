# 表格字段与读取方法

所有CSV使用UTF-8 BOM；csv.gz是gzip压缩CSV，可用pandas.read_csv直接读取。数字保留源浮点精度；各表均不跨风场或随机种子取均值。

| 文件 | 每行是什么 | 重要字段 |
|---|---|---|
| all_trained_results.csv | 一次训练job、一个评价年份；18460行 | family、station_id、seed、split、mae、rmse、bias、width/lr/weight_decay、source_job_id |
| all_tuning_results.csv | 一次调参或调参参考job的2023结果；540行 | 通过source_job_id中的c0–c3及实际配置区分试验，seed=40/41 |
| formal_and_ablation_results.csv | 正式/消融/正式B0及持续性、E10派生视图 | row_kind区分trained/deterministic/derived_reuse；actual_family给出实际结构 |
| selected_baseline_resolved.csv | E10对既有基线结果的逐seed复用；1120行 | family=SELECTED_BASELINE不代表新网络；actual_family和source_job_id可追溯 |
| deterministic_results.csv | 一站一年的持续性预测；112行 | 没有seed，未复制成多个训练运行 |
| 2023/2024_mae/rmse_by_seed.csv | 一个风场、一个模型，种子42–51分别为列 | 空值=未运行，不是0，也不是结果被剔除 |
| architecture_dictionary.csv | 一个本轮模型缩写 | inputs、structure、condition、output、caveat、source；完整解释见ARCHITECTURES_CN.md |
| job_configs_and_costs.csv | 一个训练job或确定性模型 | 最优epoch、实际训练epoch、参数量三种口径、阶段耗时、GPU、精度、哈希、文件路径 |
| paired_results.csv | 同一风场/seed/年份的候选—对照配对 | *_candidate和*_reference是两方原值；*_gain=reference−candidate；*_direction为improved/worsened/tie |
| improvement_counts_by_seed.csv | 一组对比、一个seed、一个年份、一个指标 | improved/worsened/ties为56个风场的方向次数；unavailable表示无可用指标 |
| improvement_counts_by_farm.csv | 一组对比、一个风场、一个年份、一个指标 | n_pairs是该场实际共有seed数（5或10），不补齐未运行seed |
| improvement_counts_all_pairs.csv | 一组对比、一个年份、一个指标 | 仅计数，不计算误差均值；分母为实际风场×seed配对数 |
| cross_year_paired_results.csv | 同一风场/seed/指标在两年的改善方向 | gain_2023、gain_2024、方向及both_improved/2023_only/2024_only/neither_improved |
| lead_results.csv.gz | 一个实际模型/seed/年份/lead | lead_step=1–24，lead_hours=0.25–6，mae/rmse |
| scenario_results.csv.gz | 一个实际模型/seed/年份/场景 | scenario、n_origins、mae/rmse/bias；场景可以重叠 |
| lead/scenario_paired_results.csv.gz | 一项设计对比的逐lead或逐场景配对 | 保留两方原始值和差额；n_origins_candidate/reference相同 |
| lead/scenario_improvement_counts_by_seed.csv | 一项设计、一个seed、一个lead/场景的配对方向计数 | 无样本指标记入unavailable，不算相同 |
| daily_results_utc.csv.gz | 一个实际模型/seed/年份/UTC日期 | 原始abs_error_sum/squared_error_sum，n_origins，n_forecast_values=n_origins×24，及该次运行当天MAE/RMSE |
| training_epoch_results.csv.gz | 一个训练job的一个epoch | train_mae是训练过程中累计损失；val_mae/best_val_mae是训练选择用归约值，不与审计后float64复算指标混用 |
| forecast_cases_per_seed.csv.gz | 一个固定窗口、一个模型、一个seed、一个lead的预测 | observed/prediction/error/absolute_error；直接来自该job预测数组，E10标记实际来源 |
| case_selection.csv | 一个固定案例窗口 | origin_index、origin_local、source等；posthoc_failure明确表示事后案例 |
| selected_hyperparameters.csv | 每个正式家族的冻结配置 | 已去掉旧选择用跨seed均值分数，保留实际配置及选定索引 |
| selected_family_by_farm.csv | 每场原2023协议选定的基线family | 不依据2024重新挑选 |
| source_metrics_hashes.csv | 一个原始训练job的来源指纹 | 源metrics路径、SHA256、checkpoint SHA256 |
| historical_experiment_index.csv | 一个历史实验/资料目录 | 只作旧协议入口，不混入本轮独立结果 |

CF是容量因子；MAE/RMSE越小越好，bias=预测−观测的有符号误差。gain严格大于0即改善，不设百分比或数值容差。基线家族均为当前任务适配实现，结构见字典。

total_parameters包括冻结及未激活模块；trainable_parameters指requires_grad=True；active_gradient_parameters指最后训练阶段实际有梯度的参数。seconds_training是该job阶段耗时，CBH阶段不含前置B0训练。不要将一个字段替代另一个字段。

数据例子：`pd.read_csv('all_trained_results.csv', float_precision='round_trip')`；读取压缩明细可用同一方法并加`chunksize=100000`。
