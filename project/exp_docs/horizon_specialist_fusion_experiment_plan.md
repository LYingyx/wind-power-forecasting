# 分时段天气视图实验计划

日期：2026-09-01  
状态：已冻结并开始执行

## 一、给老师的简要说明

当前模型用最近一小时功率帮助理解未来天气，但从 15 分钟到 6 小时都采用同一种天气读取方式和同样强度的历史约束。已有结果显示：短期只读取最近天气略有帮助，长期却需要完整天气轨迹；历史基线约束在短中期有保护作用，在长期可能过强。

本实验只回答两个问题：

1. 能否让模型短期侧重最近天气、长期逐渐转向完整天气轨迹？
2. 能否让历史基线约束在短中期保持、在长期逐渐减弱？

所有模型继续使用最近一小时状态和已经验证最稳的状态—天气融合方式。不再重复多尺度历史、注意力控制、FiLM、LoRA 等已经没有稳定优势的结构搜索。

## 二、数据和结论边界

- 2022：训练。
- 2023：验证、模型选择和 early stopping。
- 2024：模型冻结后测试。
- 正式实验为 seeds 42--51 共 10 次独立训练。

2024 已在既往工作中被多次分析，因此它是内部测试集，不再称为完全未见的外部测试集。新模型的结构和选择规则只能根据 2023 冻结，查看新测试结果后不得继续调参。

最终结果以单模型为准：10 seeds 用于估计一次训练通常能达到的均值、波动和胜率，不把十个蒸馏模型平均后的性能作为主结果。最终部署 checkpoint 在查看测试结果前，选择 validation MAE 最接近十个 seeds 中位数的代表性模型。

## 三、模型改动

### 1. 轻量天气视图门控

共享同一个天气编码器，分别得到最近时刻天气表示和完整未来天气表示：

\[
W_{\rm mix}=(1-g(\tau))W_{\rm near}+g(\tau)W_{\rm full}.
\]

固定门控为：1 小时内只用最近天气，1--3 小时线性切换，3 小时以后使用完整天气。混合后的天气仍进入同一个 black-box 修正网络，因此几乎不增加参数。

### 2. 分时效 anchor

原 anchor 权重统一为 0.2。新方案在 3 小时内保持 0.2，3--4 小时线性下降，4 小时后为 0。Anchor 仍只是训练损失，不是预测专家。

### 3. 双修正专家

代码同时实现 near/full 两个独立修正头，以及 full/full 同参数控制。但该路线必须先通过零训练互补性审计；审计失败就不训练，避免用复杂模型追逐不足 1 MW 的组合空间。

## 四、预注册实验矩阵

| ID | 天气视图 | Anchor | 修正头 |
|---|---|---|---|
| R0 | full | scalar 0.2 | 单头，复用现有 Normal |
| L1 | full | horizon taper | 单头 |
| L2 | fixed near→full | scalar 0.2 | 共享单头 |
| L3 | fixed near→full | horizon taper | 共享单头 |
| L4 | learned monotonic | horizon taper | L3 通过后才运行 |
| H0 | full/full | horizon taper | 双头参数控制 |
| H1 | near/full | horizon taper | 异质双专家 |
| H2 | near/full learned | horizon taper | H1 通过后才运行 |

先用 seeds 42、46、50 筛选。通过条件为：相对 R0 平均改善至少 1 MW、至少 2/3 seeds 获胜、任一时段不恶化超过 1 MW。通过者再运行 seeds 42--51 正式验证。

## 五、统计和最终判断

主指标为 cluster MAE，同时保存 RMSE、R²、station MAE、逐 lead、三时段、爬坡和运行效率。

单模型主统计：

- 10-seed 单模型 MAE 均值与标准差；
- 同 seed 配对差异、95% CI 和胜场；
- 逐 lead 和分时段也先在每个 seed 的单 checkpoint 上计算，再跨 seed 汇总；
- Holm 多重比较校正；
- validation 中位 seed 对应的单个代表模型表现。

Ensemble 均值、按 UTC 日 bootstrap 和成员分歧只作为训练方差与 teacher 诊断，不进入结构或蒸馏学生的最终通过判定，也不作为最终模型性能。

轻量结构的机制性 superiority 结论仍需要 10-seed paired CI 支持优于 Normal。根据 2026-09-01 在查看正式 validation 和新 test 结果前确定的单模型优先口径，若候选未确认 superiority，但同时满足：相对 Normal 的 2 MW 非劣界限、至少半数 seeds 获胜、三个时段的平均退化均不超过 1 MW，则允许继续做 KD，以检验可部署单模型；此时必须明确写成“安全非劣候选”，不得改称结构显著提升。最终蒸馏单模型与当前 KD 单模型比较，同样采用 2 MW 非劣界限。

## 六、执行顺序和停止规则

1. 用已有 validation 预测做季度阻塞交叉验证的互补性审计。
2. 实现并测试 L1/L2/L3 和双专家接口。
3. 执行 3-seed 筛选。
4. 只有固定模型通过才训练 learned gate。
5. 只有筛选通过的一个轻量候选进入 10-seed 正式验证。
6. 正式结构成立后才建立同构 ensemble teacher，并训练 10 个独立 KD 学生；最终仍评价单个学生分布。
7. 全部规则冻结后统一评估 2024。

以下情况立即停止对应路线：互补组合总体改善不足 1 MW且任一时段不足 2 MW；L3 未通过则不做 L4；H1 不优于 H0 则不做 H2；测试结果不用于创造新变体。

## 七、代码和产物

- 模型接口：`weather_view_mode`、`weather_gate_mode`、`specialist_mode`。
- 损失接口：`anchor_schedule_mode`。
- 实验根目录：`artifacts/horizon_specialist_v1/`。
- 分析：`stage0/`、`analysis_screening/`、`analysis_val/`、`analysis_test/`。
- 图像：`figures/`。
- 日志：`logs/`。

所有 launcher 接受物理 GPU 列表并在真实终端显示任务级和 batch 级 tqdm；完整标记、checkpoint、metrics 和 predictions 同时存在时才允许跳过任务。
