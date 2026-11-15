# Held-out test 证据补全命令（2026-09-01）

## 协议

- train（2022）用于参数优化；validation（2023）用于 early stopping、结构选择和机制消融决策。
- 所有候选与实验矩阵冻结后，test（2024）只用于最终报告。
- 论文主表、预测轨迹、场站图、天气置乱和部署效率统一标记为 `split=test`。
- 旧目录 `artifacts/paper_evidence_v1/` 是 validation 开发诊断，不删除但不用于论文最终图。
- 新的测试产物统一写入 `artifacts/paper_evidence_test_v1/`，不会覆盖旧图。

## 当前产物审计

截至本次检查：结构消融完成 7/70，KD 消融完成 40/40；test prediction 为 0。旧 validation 目录包含 198 张集群 PNG、51 张场站 PNG，以及相同数量的 PDF。这 249 组图只能作为开发诊断。

随时重新检查：

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/audit_paper_completion.sh
```

## 阶段 1：补完结构消融训练

已修复 correction forcing/spatial ablation 改变 history-baseline checkpoint contract 的问题。launcher 会自动复用所有 `COMPLETE` 任务，只续跑缺失或失败的任务：

```bash
bash shell/launch_paper_ablation_architecture.sh 0,2,5,7
```

KD 40/40 已完成，不要重复启动。如审计显示缺失才运行：

```bash
bash shell/launch_paper_ablation_kd_targets.sh 1,3,4,6
```

结构训练完成后，先形成不接触 test 的 validation 机制结论：

```bash
bash shell/analyze_paper_ablation_completion.sh
```

## 阶段 2：导出冻结参考模型的 2024 test 预测

该矩阵导出 Direct、FullHistory、Normal、KD 的 10 seeds，共 40 次纯评估。第一个参数仍是实际物理 GPU：

```bash
bash shell/launch_paper_test_cluster_exports.sh 1,3,4,6
```

阶段 1 和阶段 2 可以在不同终端并行，但物理 GPU 列表不得重叠。

## 阶段 3：test 主图与场站图

集群预测导出完成后生成约 198 张 test PNG 及对应 PDF：

```bash
bash shell/generate_paper_cluster_evidence.sh --case-count 16
```

场站图默认固定 seed42，导出 Direct、Normal、KD 共 3 次 test 评估：

```bash
bash shell/launch_paper_evidence_exports.sh 0,2,5
bash shell/generate_paper_station_evidence.sh --case-count 16
```

若论文需要 3-seed 场站稳定性，显式运行：

```bash
WIND_PAPER_EVIDENCE_EXPORT_SEEDS=42,43,44 \
bash shell/launch_paper_evidence_exports.sh 0,2,5

bash shell/generate_paper_station_evidence.sh --seeds 42,43,44 --case-count 16
```

## 阶段 4：test 天气置乱负对照

必须先完成阶段 2，因为分析需要未置乱 test prediction。正式口径为 Normal/KD × 10 model seeds × 5 permutations，共 100 次 test 评估：

```bash
bash shell/launch_paper_weather_robustness.sh 0,1,2,3,4,5,6,7
bash shell/analyze_paper_weather_robustness.sh
```

## 阶段 5：预注册消融的 test 复核

仅在 110 个消融训练全部完成、且不再依据 test 修改结构之后运行。它对 11 组 × 10 seeds 做 test 纯评估：

```bash
bash shell/launch_paper_ablation_test_evaluation.sh 0,1,2,3,4,5,6,7
bash shell/analyze_paper_ablation_test.sh
```

validation 分析决定机制，test 分析只验证冻结结论是否外推，不允许据此再挑结构。

## 阶段 6：效率与统一绘图

```bash
bash shell/launch_paper_efficiency_benchmarks.sh 1,4,7
```

当 cluster export、station export 和 weather permutation 均完成后，可并行重新生成全部 test 图：

```bash
bash shell/launch_paper_evidence_plots.sh cluster,station,weather
```

最终检查：

```bash
bash shell/audit_paper_completion.sh
```

所有 launcher 都默认显示总 job `tqdm` 和各 GPU 的 batch/epoch `tqdm`，并通过 `TEST_COMPLETE` 支持续跑。
