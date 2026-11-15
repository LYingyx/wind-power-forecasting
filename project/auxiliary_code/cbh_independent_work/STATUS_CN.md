# CBH 逐风场多种子与完全独立训练实验

## 已有结果与本批目的

此前已经完成 `CBH_SHARED_56` 的完整逐风场评估。原始 `metrics.json` 均保存 56 个场站的 `per_station_mae_cf` 和 `per_station_mae_mw`，覆盖 seeds 42–51、2023 validation 和 2024 cross-year evaluation。旧报告只展示总体均值与改善站数；本目录已将它们展开为完整逐站表，而非重新估算：

- `analysis/shared_per_station_seed_long.csv`：28,000 行，2 splits × 25 models × 56 sites × 10 seeds；
- `analysis/shared_per_station_summary.csv`：逐模型、逐 split、逐站 mean/SD/bootstrap CI/min/max；
- `analysis/shared_per_station_comparisons.csv`：逐站配对消融、seed wins、bootstrap CI、Wilcoxon/Holm；
- `status/SHARED_EXTRACTION_AUDIT.json`：来源文件、行数、CF/MW 一致性和总体均值复算审计。

本批新增 `CBH_INDEPENDENT_SITE`，回答逐风场完全独立参数能否超过当前共享模型。

## 参数共享边界

`CBH_SHARED_56` 不是 56 个独立模型。冻结 history-only ForcedEvolution 中，下列可学习模块在风场间共享：station/cluster history encoder、target-time embedding、history query、response layers、trajectory decoder、output head及 weather/static embedding。其 baseline 还读取由全场站功率聚合得到的 `cluster_history`。

CBH correction 中共享：history projection/encoder、weather projection、P0I alignment projection、condition projection、transfer adapter、Transformer decoder、readout 和 lead-dependent `condition_bias`。场站差异来自静态位置/容量、本站历史、本站局地天气和本站 baseline trajectory。

`CBH_INDEPENDENT_SITE` 对每个“场站 × seed”构造完整独立的 history baseline 和 CBH correction。baseline 的 cluster-history 分支只读取本站历史，任何阶段都不读取其他场站功率；不同场站和不同 seed 之间不共享可学习 tensor。为了只检验参数共享效应，经纬度/容量的固定归一化常数仍沿用 56 站训练集定义。

## 正式协议

- 风场：全部 56 个；
- seeds：42–51；
- 2022 train，2023 validation/early stopping/checkpoint selection，2024 cross-year evaluation；
- 第一阶段：独立 history-only ForcedEvolution；
- 第二阶段：冻结该场站同 seed 的 baseline，训练独立 CBH correction；
- AdamW，lr 2e-4，weight decay 1e-4；
- 80 epoch 上限，patience 10，BF16，MAE；
- weather 训练使用 12 cycles/batch，validation 使用 24 cycles/batch；
- 无输出裁剪、EMA、checkpoint averaging、auxiliary loss、爬坡损失或爬坡指标。

## 预检与执行

`status/PREFLIGHT_AUDIT.json` 已验证：

- 单站模型只看到 `[batch, 1, 24]` 的功率历史；
- cluster 分支逐值等于本站历史；
- 不构造其他场站功率 tensor；
- 两个 seed 的参数 storage 完全分离；
- 单站静态向量覆盖后所有 state 有限；
- train/val/test 天气时间偏移严格递增；
- 局地天气缓存与原 P0I 双线性读取逐值一致；
- 烟雾测试完成 baseline/CBH forward、backward、checkpoint、2023/2024 inference，冻结 baseline hash 不变。

正式队列使用 64 个主 worker，即 8 GPUs × 8 processes。GPU 6 启动时有外部进程占用约 21.55 GiB，因此该卡的 OOM 任务由 GPU 0/1/3/4/5/7 上额外的 12 个 rescue worker 原子领取；这些卡各有 10 个本实验进程。GPU 6 上提前退出的 4 个 worker 已由 4 个显存门控 worker 补齐：它们在空闲显存不足 1.2 GiB 时等待，资源释放后才原子领取任务，避免反复 OOM。`watch_and_finalize.sh` 每分钟写入 `status/PROGRESS.json`，达到 560/560 后自动执行完整审计、聚合与中文报告生成。

最终产物将写入：

- `analysis/independent_per_station_seed_long.csv`
- `analysis/independent_per_station_summary.csv`
- `analysis/independent_vs_shared_comparisons.csv`
- `analysis/independent_overall_summary.csv`
- `analysis/independent_job_audit.csv`
- `analysis/REPORT_CN.md`
- `status/INDEPENDENT_FINAL_AUDIT.json`

负的 `CBH_INDEPENDENT_SITE − CBH_SHARED_56` 表示逐站独立训练更好。正式结论必须等 560 个作业全部通过审计后再写，不能用运行中的部分结果替代。
