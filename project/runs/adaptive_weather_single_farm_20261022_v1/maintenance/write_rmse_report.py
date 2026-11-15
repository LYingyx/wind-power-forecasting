"""Write a review from complete local results; remote coverage remains explicit."""
from pathlib import Path
import pandas as pd,json
R=Path(__file__).resolve().parents[1];OUT=R/'analysis/rmse_review_20261024'
def read(n):return pd.read_csv(OUT/n,float_precision='round_trip')
def md(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'
c=read('all_comparisons_by_seed.csv');actual=read('all_actual_results.csv');hz=read('horizon_comparisons_by_seed.csv');sc=read('scenario_comparisons_by_seed.csv');diag=read('residual_pair_diagnostics.csv')
def stat(a,b,year=2024):
 g=c[(c.candidate==a)&(c.reference==b)&(c.year==year)]
 return [int(g[x].sum()) for x in ['n_farms','mae_better','rmse_better','mae_improve_rmse_worse']]
report='''# RMSE完整核查：当前独立单风场实验

日期：2026-10-24。主表范围为20261021与20261022两轮独立训练结果。P60002、P60003免密连接和远端存储核查已完成，两机共享盘与当前机器相同；更早历史结果的独立复核见[跨服务器盘点与历史RMSE补充](remote_inventory/REPORT_CN.md)。历史共享训练和不同数据协议不混入本轮排名。

结论：相对点天气Transformer和TiDE，新主模型ST11的2024 RMSE确实在更多风场×seed配对上退步，且不只是少数极端站点。它也不是“比所有基线都差”；相比多数较弱基线仍有优势。W0和ST00已存在MAE/RMSE取舍，自适应前端没有消除，并在不少配对上进一步加重它。

口径：全部按相同风场、seed、年份配对。下面的560是56风场×10seed，不是560个独立风场；280是56×5。每个seed保留实际值，不跨风场/seed平均，也不设改善幅度门槛。计数只描述方向，不能代表效应大小或独立性显著性。原值和逐seed表见文末。

读取合并后44个命名配置/对照、34,272条风场×seed×年份记录（含确定性持续性和旧选定基线复用）；调参结果不作为正式模型排名。另直接重算4,200份焦点模型原始预测，核验4,760组残差配对。没有复制原始预测。

## 1. “RMSE更差”是否成立

MAE胜出和RMSE胜出分别统计，不能互相替代。

'''
rows=[]
for a in ['E2C_CBH','W0','ST00','ST11']:
 for b in ['TRANSFORMER_FC__W0','TIDE_FC__W0','SELECTED_BASELINE']:
  rows.append([a,b,*stat(a,b)])
report+=md(['候选','参考','配对数','MAE更好','RMSE更好','MAE更好但RMSE更差'],rows)
report+='ST11：本轮完整空间+时间自适应提取前端，接NEUTRAL_LAST骨干；W0：同骨干固定双线性取点+线性时间插值；ST00：同参数规模的静态空间/时间评分；E2C_CBH：上一轮冻结历史轨迹条件模型。Transformer/TiDE的`__W0`为原点天气基线。SELECTED_BASELINE沿用旧2023选择，不重新用2024择优。\n\n'
report+='ST11对Transformer的每个seed如下。只有seed50的RMSE胜出风场超过一半，其余9个seed均不足一半；这仍不等价于每个风场都退步。\n\n'
g=c[(c.candidate=='ST11')&(c.reference=='TRANSFORMER_FC__W0')&(c.year==2024)].sort_values('seed')
report+=md(['seed','风场数','MAE更好','RMSE更好','RMSE更差','MAE更好/RMSE更差'],[[int(r.seed),int(r.n_farms),int(r.mae_better),int(r.rmse_better),int(r.rmse_worse),int(r.mae_improve_rmse_worse)] for r in g.itertuples()])
report+='按实际风场检查，56站中34站在多数seed上RMSE更差，7站的全部10个seed更差，1站全部10个seed更好。\n\n'
report+='## 2. 不同基线与消融要分开看\n\n'
refs=['CNN_W','DLINEAR_FC','NBEATSX_FC','TIMEXER_FC','TFT_FC','TCN_FC','GRU_H','PATCHTST_H','B0','W0','ST00','TRANSFORMER_FC__S1']
report+=md(['ST11参考对象','配对数','MAE更好','RMSE更好','MAE更好/RMSE更差'],[[b,*stat('ST11',b)] for b in refs])
report+='不能用击败较弱基线掩盖对Transformer/TiDE的短板，也不能把对强基线的失败推广为对所有基线都失败。\n\n'
controls=[('S2','W0'),('S3','S2'),('S4','S3'),('T2_7','T1_7'),('T2_8','T1_8'),('ST11','ST00'),('ST11','ST_NO_H'),('H1','W0'),('W0','H2'),('W0','CLIM_LAST')]
report+=md(['候选','对照','配对数','MAE更好','RMSE更好','MAE更好/RMSE更差'],[[a,b,*stat(a,b)] for a,b in controls])
report+='S2是训练固定描述生成空间权重、实际天气作值；S3改为样本历史/天气生成权重；S4增加风向几何。T1/T2是静态/自适应点位置时间注意力。ST_NO_H仅在提取权重中固定历史；H1增加冻结b/Δb条件但保持末值锚点；H2固定历史编码输入；CLIM_LAST用训练天气均值。这些正负效应都应保留，不能预设“自适应一定更好”。\n\n'
report+='## 3. 为什么MAE改善而RMSE退步\n\n'
report+='公式是MAE=mean(|e|)，RMSE=sqrt(mean(e²))，这里的mean只在单个模型的预测样本和lead内计算。大量普通样本的小幅改善可以降低MAE，同时较大误差的恶化会抬高RMSE。已用原始预测确认这种分布变化，不能仅凭两个总指标猜测。\n\n'
report+='ST11对Transformer有181个“MAE改善但RMSE退步”的配对：其中173个绝对误差中位数更低，159个绝对误差99%分位数更高；163个中心化误差方差增加。于是普通样本精度提升与尾部恶化并存。在全部352个RMSE退步配对中，329个中心化误差方差增加，313个方差增量大于全局偏差平方的增量。这里使用MSE=Var(e)+Bias(e)²的精确分解，指误差分散度，不是跨seed的不确定性或因果归因。\n\n'
report+='对同一配对的每个样本，用两模型绝对误差的较大值分成[0,0.1)、[0.1,0.2)、[0.2,0.4)、[0.4,∞)的CF区间，分别累加平方误差差；分区之和严格等于总MSE差。这些阈值只用于解释误差尺度，不是判定提升的门槛。181个取舍配对中，160个在≥0.4区间的平方误差贡献增加；分解明细完整保存，不把某一分区当作独立因果。\n\n'
report+='### 远时效与场景\n\n'
g=hz[(hz.candidate=='ST11')&(hz.reference=='TRANSFORMER_FC__W0')&(hz.year==2024)]
report+=md(['时效','配对数','MAE更好','RMSE更好'],[[k,int(x.n_farms.sum()),int(x.mae_better.sum()),int(x.rmse_better.sum())] for k,x in g.groupby('horizon')])
report+='352个RMSE退步配对中，336个在3–6小时段有正的MSE恶化贡献；181个取舍配对中相应为168个。优势集中在短时效，但不能据此证明末值锚点是唯一原因：Transformer/TiDE也含末值/历史跳连，骨干还存在编码、激活、dropout、参数规模等差别。\n\n'
g=sc[(sc.candidate=='ST11')&(sc.reference=='TRANSFORMER_FC__W0')&(sc.year==2024)]
report+=md(['场景','配对数','MAE更好','RMSE更好'],[[k,int(x.n_farms.sum()),int(x.mae_better.sum()),int(x.rmse_better.sum())] for k,x in g.groupby('scenario')])
report+='季节、功率区间和爬坡标签彼此可能重叠，不可把表中数量相加。future_ramp使用未来真值，只是诊断切片，不能作为部署时已知输入。冬季与未来爬坡段较弱，但非爬坡段也存在RMSE短板，不能归咎于爬坡这一种情况。\n\n'
report+='### 训练目标与跨年变化\n\n'
report+='两轮候选和基线都以MAE训练、按2023验证MAE选择最佳epoch；本轮前端选型也以MAE为主。MAE偏向条件中位数，平方损失偏向条件均值，因此当前协议不保证RMSE最优。但所有基线同样MAE训练，损失函数只能解释“为什么不保证RMSE胜出”，不能单独解释“为什么Transformer更好”。目前没有MAE对MSE/混合损失的受控重训证据，也没有保存全部epoch模型来证明改选RMSE一定有效。\n\n'
report+=md(['ST11对Transformer','配对数','MAE更好','RMSE更好','MAE更好/RMSE更差'],[[year,*stat('ST11','TRANSFORMER_FC__W0',year)] for year in [2023,2024]])
report+='其中193个配对在2023 RMSE更好、到2024变差，反向变好只有29个。说明跨年泛化是必须正视的一部分；可能涉及气象、功率状态或运行分布变化，但现有比较没有证明某个具体漂移原因。2024已经用于复评，不能继续称作未查看的盲测。\n\n'
report+='### 不是简单裁剪即可修复\n\n'
report+='对双方同时做[0,1]裁剪的诊断中，原先ST11 RMSE较差的352个配对全部仍较差；其MSE恶化在双方预测都处于[0,1]的样本上也全部为正。越界不是该差距的充分解释。还发现真值CF最大约1.46384，因此[0,1]并非已验证的真值支持范围，不能直接把裁剪当作合法修复。应核对容量口径、单位及超额出力/数据质量；这里没有修改正式预测或分数。\n\n'
report+='## 4. 固定代表风场的实际值示例\n\n'
report+='以下使用原先按训练描述量选定的6个代表站和seed42，并非按2024输赢挑选。单位是本站容量归一化功率；更多seed见完整原值表。\n\n'
t=actual[actual.year==2024];x=t[t.arm=='ST11'].merge(t[t.arm=='TRANSFORMER_FC__W0'],on=['site','seed'],suffixes=('_c','_r'));x=x[x.site.isin([6,8,14,16,43,46])&x.seed.eq(42)].sort_values('site')
report+=md(['风场','seed','ST11 MAE','TF MAE','ST11 RMSE','TF RMSE'],[[z.station_name_c,42,*[f'{getattr(z,k):.9f}' for k in ['mae_c','mae_r','rmse_c','rmse_r']]] for z in x.itertuples()])
report+='双龙风场seed42体现MAE/RMSE取舍；到保风场seed42两指标都退步；大岗、马力、王府该seed两指标改善。这些例子不能替代完整56站×seed比较。\n\n'
report+='## 5. 对论文与下一步的判断\n\n'
report+='目前不宜声称“全面优于强基线”或“自适应提取带来稳定提升”。当前证据更支持：历史与逐次天气有价值；静态空间提取在一些站点有增量；自适应模块收益不稳定；短时效典型误差与远时效大误差之间存在取舍。贡献措辞应随这些证据调整。\n\n'
report+='若后续继续优化，先在验证期比较MAE、MSE和预定的混合损失，并让W0/ST00/ST11与Transformer使用同等预算；把checkpoint选择指标作为独立因素，才能识别训练目标与结构的作用。先核查超1真值和运行状态数据，再考虑极端/爬坡的处理；不要在2024上反复挑模型后宣称泛化优势。本次只分析已有结果，没有新增训练或改动正式指标。\n\n'
report+='## 6. 数据入口\n\n'
report+='远端补充：旧共享E2C相对同期共享Transformer的2024 RMSE在479/560个风场×seed配对中更好；旧Physics Gate和Level/Shape相对同期clean Transformer分别为136/168、131/168。前者有区域历史和精度差异，后两者同时改变结构与训练损失，不能当作当前独立ST11的成绩或纯损失因果证据。原值、来源与去重范围见[跨服务器补充报告](remote_inventory/REPORT_CN.md)。\n\n'
for fn,desc in [('all_actual_results.csv','全部实际值与来源job'),('all_comparisons_by_seed.csv','全部模型与各基线逐seed方向计数'),('all_paired_actual_results.csv.gz','相同风场/seed的配对原值'),('horizon_comparisons_by_seed.csv','分时效逐seed计数'),('scenario_comparisons_by_seed.csv','分场景逐seed计数'),('cross_year_direction_counts.csv','2023至2024方向变化'),('residual_statistics_per_farm_seed.csv','每次模型的误差分位数、偏差、方差及范围'),('residual_pair_diagnostics.csv','配对残差、时效和越界贡献'),('squared_error_partition_per_farm_seed.csv','共同误差尺度分区的可加分解'),('target_range_by_farm.csv','真值范围核查'),('CHECK.json','原预测重算与分解验收')]:
 report+=f'- [{desc}]({fn})\n'
(OUT/'REPORT_CN.md').write_text(report)
entry=R.parents[1]/'RMSE_REVIEW_20261024_CN.md'
entry.write_text('# 当前单风场模型RMSE完整核查\n\n'+
'已核查当前服务器可访问的两轮独立实验全部正式结果，并对4200份焦点模型原始预测重算。ST11相对Transformer的2024 RMSE在352/560个配对中较差，但不能推广为比所有基线都差。主要证据是普通误差改善、较大误差加重，短时效优势到远时效减弱，以及2023到2024的泛化变化。\n\n'+
'[完整报告与逐风场逐seed原值](runs/adaptive_weather_single_farm_20261022_v1/analysis/rmse_review_20261024/REPORT_CN.md)。\n\n'+
'P60002/P60003免密连接已成功，两机共享存储中的当前实验是同一份结果。更早历史工程中保留了MSE基线、Physics Gate、Level/Shape和clean重训对照；部分汇总早已归档，本次补充原始预测复算及逐seed配对。历史共享E2C及MSE候选存在RMSE优势，但不能替代当前独立ST11的结论。详见[跨服务器盘点与历史RMSE补充](runs/adaptive_weather_single_farm_20261022_v1/analysis/rmse_review_20261024/remote_inventory/REPORT_CN.md)。\n')
print('WROTE',OUT/'REPORT_CN.md')
