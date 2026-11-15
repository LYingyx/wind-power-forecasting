# Raw-Bilinear Local-to-Global 实验

固定原始14变量2×2双线性场站采样、3 h共享局地历史基线和latent-linear时间适配，只比较聚合前局地转换、局地辅助监督、状态空间尺度和轻量场站异质性。主任务始终是未来24个15 min集群总功率。

筛选使用seeds 42/46/50。先比较cluster-only与`cluster + 0.5×station capacity-factor`监督；随后从相同L1-M checkpoint恒等初始化L1-MFT、aggregate-state和local-state。只有local-state同时优于两个控制才筛选ID16和per-station affine。正式阶段固定为无状态、aggregate-state和最佳local-state三类并补齐42–51。

所有模型按2023 validation的10-seed平均MAE选择，2024只在冻结后统一运行。无KD、无预测ensemble、无GNN、无learned interpolation、无ramp loss。状态只能乘法缩放非零天气latent，不存在state-only correction路径。
