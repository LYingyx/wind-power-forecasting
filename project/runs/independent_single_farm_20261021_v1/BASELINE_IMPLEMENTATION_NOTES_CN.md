# 基线实现说明与边界

所有模型独立训练，预测 24 步，使用统一对齐的本站天气；主表名称应加 `adapted`。源码：`code/models.py`、冻结的 `code/baselines_original.py`。本轮没有下载官方仓库，也未进行官方结果复现；因此不使用“官方实现”“达到/超过所有 SOTA”等表述。

| 模型 | 当前结构 | 任务适配/限制 |
|---|---|---|
| Transformer_FC | 两层历史 Transformer、两层未来天气 decoder、位置编码、末值锚点 | 常规 encoder-decoder；历史24、未来24，14天气通道和提前期/phase |
| TiDE_FC | 历史和天气 dense 编码、三层残差 MLP、联合 decoder 和线性 history skip | 紧凑 TiDE 风格；不是官方完整 temporal decoder 配置 |
| NBEATSX_FC | 四个 generic backcast/forecast block，每块重新接收剩余历史和外生轨迹 | 修正旧 backcast 没有反馈的问题；没有另设 interpretable basis stack |
| TimeXer_FC | 内生历史 patch＋可学 global token、外生变量 token、两层内生 self-attention 和 global cross-attention、展平头 | 外生 token 编码已发布的未来24步天气；单变量历史、实例归一化；不额外使用 phase 标量 |
| TFT_FC | 天气变量选择、两层历史/未来 LSTM、因果多头注意力、两组门控残差、点预测头 | 修正旧版无注意力；没有完整官方静态变量 selection/GRN 体系；不额外使用 phase 标量；点 MAE 替代分位数目标 |
| TCN_FC | 完整24步历史卷积和展平摘要，天气与历史摘要联合多膨胀卷积 | 修正旧版只看末次功率；未来天气已知，因此未来序列允许双向卷积 |
| DLinear_FC | 5点滑动均值 trend/season 分解＋两条线性预测＋全未来天气线性回归 | 外生回归是任务扩展；无天气 GELU/MLP；无隐藏宽度，以weight decay作为第二搜索维度 |
| GRU_H | 两层 GRU、历史/相对位置输入、末值参考的 MLP 输出 | 纯历史基线；不额外使用 phase 标量 |
| PatchTST_H | 6个不重叠4点 patch、三层 Transformer、展平输出、实例归一化 | 单变量情形；无多通道设置、无预训练；不额外使用 phase 标量 |
| CNN_W | 14通道未来天气、多尺度时序卷积、逐步头 | 纯天气基线；完全不使用功率历史和 phase 标量 |

主模型的静态站点向量和地形高度对单站模型是常量；各站可学偏置不共享。各家族对于已知 phase/时间特征的编码并不完全一致，严格的“唯一变量改变”结论只来自 CB/CH/CNEUTRAL/DIRECT/NEUTRAL_LAST 等匹配主干消融，不能把异构基线比较当成纯机制识别。

不使用旧 Swin_W：原代码实际为全局注意力加 shift，不足以称 Swin。主协议选择已有计划允许的 CNN_W。

超参数搜索次数相同不代表 FLOPs、架构容量或调参范围完全一致；成本表必须一并报告。若用于强调顶会级 SOTA 竞争，应补官方实现的外部稳健性验证，不能用改名掩盖此限制。
