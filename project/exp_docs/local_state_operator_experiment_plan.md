# Local State × Interaction Operator 实验

本轮只隔离 state-weather interaction operator。现有 L2/L3 的真实实现为
`h * (1 + 0.25*tanh(g(s)))`，因此记为 G-RMul/L-RMul并直接复用，不重复训练纯乘法模型。

新增 G/L-Int 与 G/L-Cat，使用 seeds 42/46/50 筛选。Int定义为
`h + 0.25*(M([h,s])-M([0,s]))`，严格满足零天气不变性；Cat定义为
`h + 0.25*M([h,s])`，仅作为允许state-only路径的表达能力上界。所有新增operator末层
零初始化并从相同seed的L1-M checkpoint嵌套继续训练20 epochs。

晋级要求：相对各自控制平均MAE改善至少0.5 MW且2/3 seeds获胜；batch-roll或reference
至少一种平均恶化0.5 MW且2/3同向；RMSE恶化不超过3 MW，任一horizon恶化不超过5 MW。
local correspondence另要求station-roll平均恶化0.25 MW且2/3同向。通过者补齐42–51，
只按2023 validation冻结；随后统一评估2024。

固定原始14变量双线性场站采样、56场站、共享天气/功率encoder与decoder、最近1 h状态、
`cluster + 0.5*station CF`相对loss、无ramp/KD/ensemble。明确不运行station embedding、
station adapter、GNN或aggregate+local双状态。
