# 论文消融、有效性证据与可视化命令

> 2026-09-01 更新：本文件早期版本的绘图流程使用 validation。论文最终图现已改为 held-out test，并写入独立目录。请以 [PAPER_TEST_COMPLETION_COMMANDS.md](PAPER_TEST_COMPLETION_COMMANDS.md) 的顺序和命令为准；`artifacts/paper_evidence_v1/` 仅保留为开发诊断。

## 1. 实验边界

所有新增训练仍只使用 2022 train 和 2023 validation，不触碰 2024。正式口径固定 seeds 42--51、相同训练预算和 early stopping。已有的 input factorial、state ladder、anchor、fusion operator、regional composition 和 KD 权重结果直接复用，不再重复训练。

新增实验分为两组：

### A. 结构与损失消融（Normal，无 KD）

除注明的嵌套对照外，共同参考为已有 `weak_anchor_recent_1h`。新增候选如下：

| 组 | 唯一主要变化 | 回答的问题 | 优先级 |
|---|---|---|---|
| `nearest_hour` | full NWP read → nearest node | 完整 8 节点时间上下文是否必要 | 必做 |
| `independent_weather_path` | 移除 hourly difference path | 显式天气变化编码是否必要 | 必做 |
| `no_aux_weather` | 去掉小时场站天气辅助监督 | 辅助头是否真正贡献最终预测 | 必做 |
| `no_anchor` | 去掉弱 baseline anchor | 可辨识性约束是否损害/改善准确率 | 必做 |
| `frozen_baseline` | 联合可训练 B → 冻结 B0 | joint adaptation 是否必要 | 必做 |
| `no_wind_geometry` | 去掉风向/位移几何 | 空间查询中的风几何是否必要 | 次要 |
| `fixed_bilinear` | 在 `no_wind_geometry` 上 learned query → 固定双线性 | learned spatial projection 是否必要；这是嵌套对照 | 次要 |

除 `fixed_bilinear` 与 `no_wind_geometry` 作嵌套比较外，其余组都与 Normal 比较。不允许从七组中挑最低 MAE 后宣称新模型；它们只用于判断原组件是否必要，并在 architecture family 内做 Holm 校正。之所以采用嵌套比较，是因为固定双线性取样本身不能使用 learned wind-geometry 特征，直接与 Normal 比会同时改变两个因素。

### B. 最终 KD 消融

共同参考为已有 `kd_050`：

| 组 | 唯一主要变化 | 回答的问题 | 优先级 |
|---|---|---|---|
| `kd_station_only` | KD 只监督 56 站输出 | 场站共识是否足够 | 必做 |
| `kd_cluster_only` | KD 只监督集群输出 | 集群共识是否足够 | 必做 |
| `kd_no_aux_weather` | 去掉天气辅助监督 | 辅助头在最终 KD 中是否仍必要 | 必做 |
| `kd_no_anchor` | 去掉 weak anchor | KD 是否已替代 decomposition anchor | 次要 |

已有 `Normal`、KD 0.25、KD 0.5、EMA、checkpoint average、uncertainty-aware KD 构成稳定化主消融。另有 rotating single-teacher control，用于区分“普通 teacher regularization”和“ensemble consensus”。

## 2. GPU 训练命令

以下逗号分隔值均是实际物理 GPU 编号，不是进程内重新编号。脚本会让每个任务只看到指定 GPU，并自动复用 `COMPLETE` 任务。

所有 launcher 默认显式显示两层 `tqdm`：第 0 行是整个并行矩阵的已完成 job 数；随后每行对应一个 GPU 槽位，显示当前组、seed、epoch/batch。每个任务的完整 stdout/stderr 仍同步保存到其 `run.log`。如在非交互调度系统中只想保留总 job 进度条、隐藏逐 batch 进度，可显式设置 `WIND_TQDM_VISIBLE=0`；正常手工运行无需设置。

结构消融，例如使用物理 GPU 0、2、5、7：

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/launch_paper_ablation_architecture.sh 0,2,5,7
```

正式启动前仅检查解析出的 GPU、组、seed 和总任务数（不会创建训练任务）：

```bash
bash shell/launch_paper_ablation_architecture.sh 0,2,5,7 --dry-run
```

只跑必做的五组：

```bash
WIND_PAPER_ARCH_GROUPS=nearest_hour,independent_weather_path,no_aux_weather,no_anchor,frozen_baseline \
bash shell/launch_paper_ablation_architecture.sh 0,2,5,7
```

KD target/loss 消融：

```bash
bash shell/launch_paper_ablation_kd_targets.sh 1,3,4,6
```

KD 矩阵预检：

```bash
bash shell/launch_paper_ablation_kd_targets.sh 1,3,4,6 --dry-run
```

只跑两个最关键 KD target 对照：

```bash
WIND_PAPER_KD_GROUPS=kd_station_only,kd_cluster_only \
bash shell/launch_paper_ablation_kd_targets.sh 1,3,4,6
```

两套 launcher 可在两个 tmux 窗口同时启动，但不要给它们重复的物理 GPU。

训练完成后统计：

```bash
bash shell/analyze_paper_ablation_completion.sh
```

只分析已经完成的组：

```bash
bash shell/analyze_paper_ablation_completion.sh --skip-missing
```

### Single teacher vs ensemble consensus（推荐补完）

teacher archives 已生成；以下命令会续跑未完成的 single-teacher 和 full-history-teacher controls：

```bash
bash shell/launch_consensus_controls.sh 0,2,5,7
```

完成后：

```bash
PYTHONPATH=src:. /data/liying/anaconda3/envs/torch241/bin/python \
  scripts/analyze_consensus_controls.py
```

## 3. 不依赖总体 MAE 的有效性证据

### 3.1 直接使用现有预测生成集群图

该步骤不使用 GPU。默认比较 Direct、Normal、KD，读取全部 10 seeds，生成 PNG+PDF、CSV 和 JSON。默认会产生约 100 张以上 PNG：

```bash
bash shell/generate_paper_cluster_evidence.sh
```

减少或增加每类案例数：

```bash
bash shell/generate_paper_cluster_evidence.sh --case-count 16
```

图和表包括：

- 逐 lead MAE、bias、persistence skill、seed disagreement；
- absolute-error ECDF 和 extreme-error survival；
- deterministic reliability、残差 hexbin、1/3/6 h observed-predicted hexbin；
- 逐日滚动误差、逐日胜率/差值分布、6 h energy error；
- current power × recent ramp 条件热图；
- month × horizon 稳定性热图；
- large-ramp precision-recall 与 ramp fidelity；
- ensemble-size saturation；
- baseline/correction 随时效的误差和幅度；
- 固定 seed42 和 ensemble 两套轨迹图：最大改善、最大退化、最大未来 ramp、最大当前 ramp、固定随机案例和每月中位误差案例。

案例选择规则记录在 `case_manifest.csv`。可按排版效果挑图，但论文必须保留类别名称，不能把最大改善案例写成随机案例。

### 3.2 场站级图像

先用 GPU 导出场站预测。默认只导出固定 seed42，约束存储量；物理 GPU 可任意指定：

```bash
bash shell/launch_paper_evidence_exports.sh 0,2,5
```

如需 3 seeds 的场站证据：

```bash
WIND_PAPER_EVIDENCE_EXPORT_SEEDS=42,43,44 \
bash shell/launch_paper_evidence_exports.sh 0,2,5
```

随后生成场站图：

```bash
bash shell/generate_paper_station_evidence.sh
```

3-seed 导出对应：

```bash
bash shell/generate_paper_station_evidence.sh --seeds 42,43,44
```

输出包括 56 站地理增益图、按增益排序柱图、station × horizon 热图、容量-增益关系、场站 ramp 指标、物理范围违反率、严格聚合误差，以及最佳/最差场站的高 ramp 轨迹案例。

### 3.3 正确 NWP 的置乱负对照

默认在 Normal/KD 的 10 seeds 上各做 5 次分层天气置乱，共 100 次纯评估。置乱保持 UTC 月、起报相位、NWP age 和当前功率分层，不是简单全局打乱：

```bash
bash shell/launch_paper_weather_robustness.sh 0,1,2,3,4,5,6,7
```

轻量版本（3 model seeds × 2 permutations）：

```bash
WIND_PAPER_ROBUSTNESS_MODEL_SEEDS=42,43,44 \
WIND_PAPER_ROBUSTNESS_PERMUTATIONS=1701,1702 \
bash shell/launch_paper_weather_robustness.sh 0,2,5
```

评估完成后：

```bash
bash shell/analyze_paper_weather_robustness.sh
```

### 3.4 部署效率

对物理 GPU 1、4、7 上的 Direct、Normal、KD 做统一预热和 100 次 forward benchmark：

```bash
bash shell/launch_paper_efficiency_benchmarks.sh 1,4,7
```

结果包含参数量、checkpoint 大小、单 batch/origin 时延、吞吐和峰值 allocated memory，并给出 KD single 相对顺序执行 10-member Normal ensemble 的估算加速比。跨 GPU 型号不同时不要直接比较；最好只传同型号 GPU，或只传一个 GPU 串行跑三组。

## 4. 并行生成全部图

集群图可立即生成；station/weather 必须分别等其导出/评估完成：

绘图脚本也会用 `tqdm` 显示 prediction archive 载入、bootstrap 以及 PNG/PDF 图像生成数量。

```bash
bash shell/launch_paper_evidence_plots.sh cluster
```

三个证据集都准备好后并行生成：

```bash
bash shell/launch_paper_evidence_plots.sh cluster,station,weather
```

默认产物根目录：

```text
artifacts/paper_ablation_completion_v1/
  groups/ analysis/ logs/

artifacts/paper_evidence_v1/
  cluster/ station_exports/ station/
  weather_robustness/ weather_robustness_analysis/
  efficiency/ logs/
```

## 5. 推荐论文选图顺序

优先选择：lead skill/MAE 曲线、逐日 paired delta、power-ramp regime heatmap、large-ramp PR、weather permutation、station map、一个固定随机轨迹和一个 high-ramp 轨迹。Reliability、error survival、ensemble-size saturation、correction magnitude 和更多案例放补充材料。最大改善与最大退化必须成对展示。
