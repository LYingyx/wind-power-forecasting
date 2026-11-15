# 方法与实现来源清单

| 组件 | 版本/commit | 原始来源 | 本批使用边界 |
|---|---|---|---|
| DLinear | Time-Series-Library `d7632da3156add2df094044661619977be979d60` | https://github.com/thuml/Time-Series-Library；AAAI 2023, arXiv:2205.13504 | 只调用历史功率路径 |
| PatchTST | 同上 | https://github.com/thuml/Time-Series-Library；ICLR 2023, arXiv:2211.14730 | 只调用历史功率路径；`patch_len=4,stride=2` |
| TiDE | NeuralForecast 3.2.2 wheel，源码 hash 见 `tsl_provenance.json` 外的 vendor 文件 | https://github.com/Nixtla/neuralforecast；arXiv:2304.08424 | `futr_exog_list` 显式放入未来天气，关闭单变量 temporal decoder 的 LayerNorm |
| TFT | NeuralForecast 3.2.2；Lim et al., arXiv:1912.09363 | https://github.com/Nixtla/neuralforecast | 未来天气经 future variable-selection/network 输入 |
| NBEATSx | NeuralForecast 3.2.2；arXiv:2104.05522 | https://github.com/Nixtla/neuralforecast | identity+exogenous basis，未来天气作为 `futr_exog` |
| Conditional Flow Matching | TorchCFM 1.0.7，MIT；Tong et al. arXiv:2302.00482 | https://github.com/atong01/conditional-flow-matching | ICFM、OT/SB 数学和官方路径核对；条件 OT 暂不伪造成 paired 训练 |
| Stochastic Interpolants | Albergo et al. arXiv:2303.08797 | https://arxiv.org/abs/2303.08797 | SINE_SI 只作 paired residual 的随机路径对照 |
| MeanFlow | Geng et al. arXiv:2505.13447 | https://github.com/gnobitab/MeanFlow | 1-NFE 平均速度的探索性代理；JVP target 按原文 stop-gradient |
| EDM | Karras et al. arXiv:2206.00364 | https://github.com/NVlabs/edm | `c_in,c_out,c_skip`、rho=7、Heun；时序 field 是本批适配 |
| CSDI | Tashiro et al. arXiv:2107.03502 | https://github.com/ermongroup/CSDI | 保存官方条件 mask 代码；下一阶段天气缺失模型候选 |
| TimeBridge | Park et al. arXiv:2408.06672v3 / KDD 2026；官方代码 commit `023a0aa` | https://github.com/JinseongP/TimeBridge | bridge prior 和 point-preserving 设计依据；未把 EDM 结果冒称 TimeBridge |
| I²SB / DDBM | arXiv:2302.05872 / 2309.16948；NVIDIA 代码快照 | https://github.com/NVlabs/I2SB | 端点桥与观测约束候选；当前 1h→15min 无真值，先不报告伪结果 |
| AFM | Fotiadis et al., ICML 2025, PMLR 267 | https://proceedings.mlr.press/v267/fotiadis25a.html | 两阶段 encoder+FM、adaptive noise 的结构启发；当前单站 latent adapter 另列 control |

所有 HTML/PDF/XML 和 GitHub API 响应放在 `references/raw`、`references/fulltext`，下载失败记录在对应 `*_status.json`，不会把缺失文件当作已复现代码。
