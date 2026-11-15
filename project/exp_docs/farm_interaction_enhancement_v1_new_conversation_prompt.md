# 新对话 Prompt：Farm-level History–Weather Interaction 增强实验

请继续完成并启动以下实验。不要重新从头设计，也不要只给计划；先审计现有实现，修复剩余问题，通过测试和 GPU smoke test 后，一次性启动全部实验，并持续检查到确认任务正常运行。

## 工作目录

```text
/storage/liying/new_try/wind_cluster_forced_evolution
```

## 研究目标

在固定 56 个风场的单场预测任务上，增强并严格验证 History–Weather Interaction。历史功率表示风场当前已实现的发电条件，未来 Pangu 表示外部气象 forcing。核心检验是联合模型是否稳定优于等参数、严格可分离的 Independent-Add 对照：

\[
F(H,W) \stackrel{?}{=} F_H(H)+F_W(W).
\]

这轮只研究三个方向：

1. Temporal interaction：保留历史轨迹，让不同 forecast lead 读取不同历史信息。
2. Forcing-change representation：同时使用绝对天气和相对 forecast origin 的天气变化。
3. Minimal site context：给 response/interaction branch 加入最小场站上下文，并用 station ID 仅作异质性诊断上界。

## 固定实验协议，不得改动

- 固定已有 56 个风场，不要求 held-out-farm 或 zero-shot 泛化。
- 2022 年训练，2023 年 validation 做全部模型选择，2024 年是冻结测试。
- 在写出 validation freeze 文件之前，严禁运行、读取或据此选择任何 2024 结果。
- 天气输入只使用 raw Bilinear Pangu；本轮不做 4×4、GNN 或其他空间结构。
- 不做 cluster-aware fine-tuning，不做 station selection，不加入 ramp loss。
- 每一个 Interaction 模型都必须有：
  - 等参数的 matched Independent-Add control；
  - 严格可分离性，即 Add 输出中不能出现任何 history–weather 联合运算；
  - 相同数据、训练 schedule、seed 和参数预算。
- Screening seeds：`42,44,46,48,50`。
- Formal seeds：`42,43,44,45,46,47,48,49,50,51`。
- 先 5 seeds screening，再只对合法 winner/final combination 补齐 10 seeds。
- 使用 GPU `0–7`，每卡配置 8 个 worker slot，通过实时显存准入动态调度。用户希望高吞吐，但不得在显存不足时强行保证 8 个同时 active 的进程。
- ID embedding 只能作为 diagnostic upper bound，不能参与最终主模型选择。
- Ramp 报告必须严格区分：
  - station-level CF ramp；
  - physical-sum cluster MW ramp；
  - 15 min、1 h、3 h 三种间隔。

## 已确定的实验矩阵

### A：Temporal interaction

- `A1`：lead-conditioned multi-scale history summary，简单历史提取版本。
- `A2`：future weather/lead query over history tokens 的 cross-attention 版本。

### B：Forcing transition

- `B1`：forecast-origin-relative weather delta + linear residual encoder。
- `B2`：absolute weather + adjacent delta + origin-relative delta + lightweight TCN。

### C：Site context

- `C1`：response-specific metadata residual。
- `C2`：response-specific metadata FiLM。
- `C3`：station-ID FiLM，只作诊断上界。

注意：现有 raw Bilinear station query 内部已经使用 latitude、longitude 和 capacity。C1/C2 测试的是额外的 response-specific context，不是首次引入 metadata。源数据中没有 elevation，不要虚构。

## 当前已完成的实现

已修改：

- `src/wind_forcing/models/local_state_identifiability.py`
  - 新增 `response_history_mode`：`pooled_state`、`lead_summary`、`weather_cross_attention`。
  - 新增 `forcing_transition_mode`：`legacy`、`origin_delta_linear`、`origin_delta_tcn`。
  - 新增 context modes：`metadata_residual`、`metadata_film`、`id_film`。
  - 新增 9-feature history summary、lead-conditioned summary projection、history cross-attention。
  - IAdd 的 cross-attention query 只能使用 lead；Int 可以使用 weather + lead，借此保持 Add 严格可分离。
  - 新增基于前两个 Pangu nodes 的 forecast-origin weather 外推。
  - 新增 linear/TCN forcing-transition encoders。
  - 新增 response-specific metadata/ID residual/FiLM。
  - 输出 `history_response_attention`。
  - legacy 默认设置不实例化任何 optional module，以保持旧 checkpoint 兼容。

- `src/wind_forcing/training.py`
  - 导出 sample-level `history_response_attention_mean`，已对 station/head 维平均。

已新增：

- `configs/farm_interaction_enhancement_v1.yaml`
- `shell/run_farm_interaction_enhancement_v1_job.sh`
- `scripts/launch_farm_interaction_enhancement_v1.py`
- `scripts/analyze_farm_interaction_enhancement_v1.py`
- `tests/test_farm_interaction_enhancements.py`

## 已完成验证

以下测试已通过：

```bash
/data/liying/anaconda3/envs/chronos/bin/python -m pytest -q \
  tests/test_local_state_identifiability.py \
  tests/test_farm_interaction_enhancements.py
```

结果为 `40 passed`。

以下也已通过：

- 修改文件的 `python -m py_compile`；
- 新 shell 脚本的 `bash -n`；
- analyzer 的 training q90 ramp thresholds smoke；
- global `sample_index` 到 val/test local cache position 的映射；
- 既有 M0 prediction 上 station/cluster 的 15 min、1 h、3 h ramp 指标计算。

Analyzer 曾错误地用 global sample ID 直接索引 split array，现已改为使用 `split/sample_index.npy` 和 `np.searchsorted`。

已有 baseline reuse symlink：

- `artifacts/farm_interaction_enhancement_v1/reused/m0_iadd`
- `artifacts/farm_interaction_enhancement_v1/reused/m0_int`

## Launcher 当前设计

`scripts/launch_farm_interaction_enhancement_v1.py` 应执行：

1. 对 7 个 IAdd/Int candidate pair 跑 5-seed screening。
2. 只使用 2023 station metrics 选择 A/B/C physical winners。
3. 合法 direction winner 补齐到 10 seeds。
4. 构造并训练 A+B。
5. 在 A+B 上重新筛选 C1/C2/C3。
6. 合法 physical context 和 ID diagnostic 补齐 10 seeds。
7. ID 模型不得成为 primary finalist。
8. 写出 `selection/validation_freeze.json`。
9. freeze 后才运行 2024 inference。
10. 调用 analyzer 并在全部完成后写 `COMPLETE`。

当前 selection gates：

- `Int − IAdd` 的 seed mean `< 0`，且至少 4/5 seeds 获胜；
- `candidate Int − parent Int` 的 seed mean `< 0`，且至少 4/5 seeds 获胜；
- mean RMSE 不恶化；
- 任何 horizon 的 MAE 恶化不得超过 `0.0005 CF`。

## Analyzer 应生成的完整结果

- `analysis/seed_metrics.csv`
- `analysis/formal_comparisons.csv`
- `analysis/history_attention.csv`
- `analysis/complete_baselines.csv`
- `analysis/summary.json`
- `exp_docs/farm_interaction_enhancement_v1_results.md`

必须覆盖：

- station MAE/RMSE（CF）；
- 0–1 h、1–3 h、3–6 h station MAE；
- physical-sum cluster MAE（MW）；
- station 和 cluster RampMAE：15 min、1 h、3 h；
- ramp direction accuracy；
- extreme-ramp precision/recall/F1，threshold 必须来自冻结的 2022 q90；
- per-farm improvement coverage；
- paired bootstrap/confidence intervals；
- history attention summaries；
- 完整 baseline 和 target model 表格，不能只报告 winner。

## 当前尚未完成，必须先处理

实验尚未正式启动。启动前请逐项审计并修复：

1. shell 参数数量与 launcher invocation 是否一致；
2. stage/group/run path 是否在 launcher、job shell、analyzer 三者中一致；
3. screening → formal symlink 是否正确、幂等，是否可能覆盖已有 artifact；
4. analyzer 是否会重复统计 screen/formal 相同 seed；
5. 2024 test 是否被 freeze 文件严格阻断；
6. ID 是否在所有路径上都不可能成为 primary；
7. A 或 B 无 eligible winner 时，combination 流程是否安全退化；
8. optimizer parameter groups 是否包含所有新增 module；
9. A+B+C 组合下 IAdd 是否仍严格可分离且与 Int 等参数；
10. resume/restart 是否幂等，已成功 run 不重训、失败 run 可恢复。

需要补充自动测试：

- launcher screening/winner gate；
- resume/idempotence；
- test-before-freeze protection；
- analyzer paired CI 和 ramp 公式；
- 完整 A+B+C 的参数匹配与 Add separability；
- `history_response_attention` 与 `save_prediction_archive` 的兼容性；
- legacy checkpoint strict loading 和输出兼容性。

然后运行更完整的测试集：

```bash
/data/liying/anaconda3/envs/chronos/bin/python -m pytest -q \
  tests/test_local_state_identifiability.py \
  tests/test_farm_interaction_enhancements.py \
  tests/test_farm_level_history_weather.py \
  tests/test_local_state_cluster_transfer_analysis.py
```

如测试文件名称或位置已有变化，用 `rg --files tests` 查明真实路径，不要跳过相应覆盖。

## GPU smoke test 与正式启动

在正式启动前，必须对 A2/B2/C2 的 IAdd 和 Int 分别做至少一个 batch 的 CUDA forward/backward smoke，使用 `cycles_per_batch=3`，记录每种模型峰值显存，并确认无 NaN、无 OOM、optimizer 确实更新新增模块。

此前 sandbox 内 smoke 报 `RuntimeError: No CUDA GPUs are available`，需要用可访问 GPU 的执行权限重试，不能把它误判为代码失败。

先确保脚本可执行：

```bash
chmod +x shell/run_farm_interaction_enhancement_v1_job.sh
```

确认安全后，只启动一个总 launcher：

```bash
/data/liying/anaconda3/envs/chronos/bin/python \
  scripts/launch_farm_interaction_enhancement_v1.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --slots 8
```

用 tmux 或等价的持久后台方式启动，避免会话结束导致任务被杀。不要重复启动第二个总 launcher。

启动后立即核验并报告：

- launcher session/PID；
- run manifest 和 code snapshot；
- 初始生成的 configs/logs；
- GPU 0–7 的 worker 分布；
- 当前 active/queued 数量；
- OOM/traceback 状态；
- 没有覆盖已有结果；
- 实验总数、已完成数、失败数和初步 ETA。

## GPU 资源现状与调度要求

上次检查时 GPU 0–7 均有其他用户任务：

- GPU 0–3 约 48 GB；
- GPU 4–7 约 24 GB；
- 每卡已有使用约 9.5–15.9 GB。

因此 `--slots 8` 表示每卡八个排队 worker slot，不等于无视显存强行同时跑八个训练。调度器必须依据实时空闲显存准入，以避免影响已有任务和制造 OOM 风暴。

## 结果解释边界

不要混淆 station ramp 和 cluster ramp：

\[
\frac{1}{N}\sum_i |r_i|
\neq
\left|\sum_i C_i r_i\right|.
\]

现有严格结果是：旧 `HW-Int` 改善 station MAE/RMSE 和 physical-sum cluster RampMAE，但没有改善相对 equal-parameter IAdd 的 station-level 15-minute ramp CF MAE。新实验必须分别报告所有 ramp 定义，不能笼统写“ramp 改善”或“ramp 没改善”。

## 执行要求

- 先审计代码和补测试，再做 CUDA smoke，再启动；但不要停留在方案讨论。
- 使用 `rg` 搜索代码；文件修改用 `apply_patch`。
- 保留用户已有更改，不重置或删除现有 artifact。
- 每个阶段给出简短进度；运行超过一分钟时持续汇报，不要长时间失联。
- 如果发现会改变科学比较公平性的实现错误，先修复并测试，再启动。
- 最终必须确保 baseline 与所有目标模型结果完备，而不仅是只给最优模型。

现在从检查工作区、launcher/analyzer 和现有测试开始，继续实施并启动实验。
