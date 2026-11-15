# Fig. 5 子图说明补充方案

## 当前问题

当前中文 caption：

图 5. 逐预测步误差及差值。本文模型降低 24 步平均误差，起始预测步相对 Weather-only 的改善尤为明显。

当前英文 caption：

Fig. 5. Horizon-wise errors and error gap. The proposed model reduces the 24-step average error, with the most pronounced improvement over Weather-only at the initial forecast horizons.

问题：

图中包含三个子图：

- (a) Horizon-wise MAE
- (b) Horizon-wise RMSE
- (C) Error gap: Weather-only MAE - Ours MAE

但 caption 没有明确说明 (a)、(b)、(C) 分别表示什么，尤其是 (C) 的差值方向没有在中文 caption 中解释。读者需要自己从图内英文标题判断，不够正式。

## 建议改法

只改 Fig. 5 caption，不改正文段落。

### 中文拟改

逐预测步误差及差值。(a) MAE 随预测步变化；(b) RMSE 随预测步变化；(C) Weather-only 与本文方法的 MAE 差值，正值表示本文方法误差更低。本文模型降低 24 步平均误差，起始预测步相对 Weather-only 的改善尤为明显。

### 英文拟改

Horizon-wise errors and error gap. (a) MAE over forecast horizons. (b) RMSE over forecast horizons. (C) MAE gap between Weather-only and the proposed method, where a positive value indicates a lower error of the proposed method. The proposed model reduces the 24-step average error, with the most pronounced improvement over Weather-only at the initial forecast horizons.

## 理由

1. 图内子标题是英文，中文论文 caption 需要给出中文解释。
2. (C) 的差值方向很重要，必须说明“正值表示本文方法更优”。
3. 只改 caption 最小，不影响表格、正文结构和页数。

## 拟执行动作

如确认：

1. 修改 `sections_short/experiments_zh.tex` 中 Fig. 5 caption。
2. 同步修改 `sections_short/experiments_en.tex` 中 Fig. 5 caption。
3. 编译中英文短版。
