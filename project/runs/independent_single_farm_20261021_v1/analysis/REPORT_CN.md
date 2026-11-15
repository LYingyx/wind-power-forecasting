# 独立单风场实验自动报告

更新时间：2026-10-21T22:52:40.463009+08:00

训练阶段完成 9500/9500；运行 0；失败 0。

**全部训练阶段完成。**

只有 56 站及预定全部种子齐全的家族进入正式汇总表。完整家族：B0, CB, CH, CLIMATOLOGY, CNEUTRAL, CNN_W, DIRECT, DLINEAR_FC, E0, E2C_CBH, GRU_H, NBEATSX_FC, NEUTRAL_LAST, PATCHTST_H, PERSISTENCE, SELECTED_BASELINE, TCN_FC, TFT_FC, TIDE_FC, TIMEXER_FC, TRANSFORMER_FC

本轮任务是各站独立训练、本站输入的 E2C-CBH；所有命名基线为文档注明的适配实现。2024 是跨年复评。

入口文件：

- `main_table_complete_only.csv`：完整模型主表；`farm_table_complete_only.csv`：全部风场。
- `paired_contrasts_complete_only.csv`：匹配效应量、风场/种子区间、7 日块区间及探索性等价分析。
- `lead_table_complete_only.csv`、`horizon_bands_complete_only.csv`：提前期结果。
- `scenario_table_complete_only.csv`：季节、当地时段、当前功率、已观测爬坡及未来诊断分组。
- `validation_selected_baselines.csv`：只按 2023 选择的逐站基线；有完整候选集时生成。
- `case_selection.csv`、`forecast_cases.csv`、`figures/`：固定案例和 PDF/PNG 图。
- `training_cost.csv`：分阶段成本；主模型部署训练成本需加同站同种子的 B0。并发墙钟耗时不是独占 GPU 活跃时间。

置信区间跨零不等于排除了某种机制；仅 90% 区间完全落在 ±0.001 CF 内时，报告该探索性容差下的等价。不得将事后未来爬坡诊断表述为可部署条件。

完整家族 2024 主表（MAE）：

- NEUTRAL_LAST: 0.086925，seed std 0.000447
- CB: 0.087390，seed std 0.000427
- E2C_CBH: 0.087447，seed std 0.000411
- E0: 0.087509，seed std 0.000264
- CNEUTRAL: 0.087652，seed std 0.000286
- CH: 0.087657，seed std 0.000490
- DIRECT: 0.087720，seed std 0.000181
- SELECTED_BASELINE: 0.087904，seed std 0.000161
- TRANSFORMER_FC: 0.088033，seed std 0.000192
- TIDE_FC: 0.088208，seed std 0.000226
- TCN_FC: 0.088698，seed std 0.000287
- TFT_FC: 0.088766，seed std 0.000277
- NBEATSX_FC: 0.101288，seed std 0.000151
- TIMEXER_FC: 0.102728，seed std 0.000109
- DLINEAR_FC: 0.103594，seed std 0.000062
- GRU_H: 0.107541，seed std 0.000016
- CLIMATOLOGY: 0.107574，seed std 0.000059
- B0: 0.107659，seed std 0.000035
- PATCHTST_H: 0.108847，seed std 0.000033
- CNN_W: 0.110460，seed std 0.000386
- PERSISTENCE: 0.112763，seed std 0.000000

预先指定比较的结果：

- CB − CNEUTRAL: ΔMAE=-0.000262，95% 风场/种子区间 [-0.000538, -0.000031]；40/56 站前者更好。
- E2C_CBH − CH: ΔMAE=-0.000099，95% 风场/种子区间 [-0.000181, -0.000019]；38/56 站前者更好。
- CH − CNEUTRAL: ΔMAE=+0.000005，95% 风场/种子区间 [-0.000219, +0.000277]；31/56 站前者更好。
- E2C_CBH − CB: ΔMAE=+0.000168，95% 风场/种子区间 [-0.000117, +0.000541]；23/56 站前者更好。
- DIRECT − E2C_CBH: ΔMAE=+0.000162，95% 风场/种子区间 [-0.001008, +0.002038]；39/56 站前者更好。
- CNEUTRAL − NEUTRAL_LAST: ΔMAE=+0.000727，95% 风场/种子区间 [+0.000380, +0.001145]；9/56 站前者更好。
- E2C_CBH − E0: ΔMAE=-0.000062，95% 风场/种子区间 [-0.000503, +0.000376]；26/56 站前者更好。
- E2C_CBH − CLIMATOLOGY: ΔMAE=-0.020015，95% 风场/种子区间 [-0.022349, -0.017567]；54/56 站前者更好。
- E2C_CBH − TRANSFORMER_FC: ΔMAE=-0.000586，95% 风场/种子区间 [-0.001111, -0.000103]；36/56 站前者更好。
- E2C_CBH − TIDE_FC: ΔMAE=-0.000762，95% 风场/种子区间 [-0.001423, -0.000197]；37/56 站前者更好。
- E2C_CBH − NBEATSX_FC: ΔMAE=-0.013841，95% 风场/种子区间 [-0.015089, -0.012504]；56/56 站前者更好。
- E2C_CBH − TIMEXER_FC: ΔMAE=-0.015281，95% 风场/种子区间 [-0.017200, -0.013277]；54/56 站前者更好。
- E2C_CBH − TFT_FC: ΔMAE=-0.001319，95% 风场/种子区间 [-0.001841, -0.000850]；47/56 站前者更好。
- E2C_CBH − TCN_FC: ΔMAE=-0.001251，95% 风场/种子区间 [-0.001680, -0.000821]；46/56 站前者更好。
- E2C_CBH − DLINEAR_FC: ΔMAE=-0.016147，95% 风场/种子区间 [-0.017809, -0.014314]；56/56 站前者更好。
- E2C_CBH − GRU_H: ΔMAE=-0.020094，95% 风场/种子区间 [-0.022400, -0.017606]；55/56 站前者更好。
- E2C_CBH − PATCHTST_H: ΔMAE=-0.021400，95% 风场/种子区间 [-0.023756, -0.018909]；55/56 站前者更好。
- E2C_CBH − CNN_W: ΔMAE=-0.023013，95% 风场/种子区间 [-0.026845, -0.019931]；56/56 站前者更好。
- E2C_CBH − B0: ΔMAE=-0.020212，95% 风场/种子区间 [-0.022517, -0.017738]；55/56 站前者更好。
- E2C_CBH − PERSISTENCE: ΔMAE=-0.025316，95% 风场/种子区间 [-0.027583, -0.022896]；56/56 站前者更好。
- E2C_CBH − SELECTED_BASELINE: ΔMAE=-0.000457，95% 风场/种子区间 [-0.000876, -0.000034]；38/56 站前者更好。
