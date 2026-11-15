from pathlib import Path
import sys,json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'));from settings import now,atomic_json
A=json.loads((R/'analysis/FINAL_ANALYSIS.json').read_text());audit=json.loads((R/'analysis/FINAL_AUDIT.json').read_text());S={x['group']:x for x in A['summary']};C={(x['candidate'],x['reference']):x for x in A['comparisons']}
lines=['# 共享预训练 + 逐站个性化：最终分析','',f'分析时间：{now()}。训练结果最后生成于 {audit["last_training_timestamp"]}；1,100/1,100 项训练成功。','',
'## 结论','',
'逐站微调在 2023 选择集上同时改善 W0 和 TF 的 MAE、RMSE，且 5 个 seeds 的方向一致；全参数微调平均优于只调输出层。当前最佳平均指标来自 TF + SEL + 区域实况 + 全参数微调。W0 没有达到优于同设定 TF 的目标，但两骨干的细小差距受 seed 45 显著影响，不能把平均排序写成跨种子的稳定架构优势。','',
'## 协议与完整性','',
'- 55 站（按原先质量规则剔除龙湖并重算区域输入），5 seeds：42–46。2022 更新参数；2023 选择检查点并评价，排除 11 月；本轮未评价 2024。',
'- W0 与 TF 都使用 SEL、区域实况和相同站点信息。各自从已冻结的共享检查点出发，统一 AdamW、学习率 2e-5、最多 30 轮、patience 6、MAE 损失。',
'- P0 共享模型直接预测，550 个既有站点-seed 基准；P1 全参数微调、P2 仅输出 MLP + 本站嵌入，各 2 骨干 × 55 站 × 5 seeds，共 1,100 项训练。',
'- 每站 29,647 个评价起报点，24 个 15 min 时效。宏平均为各站、各 seed 的 MAE 或 RMSE 等权平均，不是混池平方误差再开方。',
'- 每个任务核对了 manifest、源检查点和输出检查点 SHA256、样本数、指标有限值与聚合关系、训练轮数，以及从全部逐轮日志独立重算最优检查点。冻结代码和源数据哈希一致。','',
'## 主表','', '|骨干|方式|MAE|RMSE|相对各自共享 MAE / RMSE|','|---|---|---:|---:|---:|']
labels={'P0_SHARED':'共享，不微调','P1_FULL':'逐站全参数微调','P2_HEAD':'输出层 + 本站嵌入微调'}
for family in ['W0','TF']:
 for mode in labels:
  g=family+'_'+mode;s=S[g];delta='—' if mode=='P0_SHARED' else f'{C[g,family+"_P0_SHARED"]["mae"]["change_percent"]:+.3f}% / {C[g,family+"_P0_SHARED"]["rmse"]["change_percent"]:+.3f}%'
  lines.append(f'|{family}|{labels[mode]}|{s["mae"]:.6f}|{s["rmse"]:.6f}|{delta}|')
lines+=['','## 微调收益与风场覆盖','', '|比较|MAE 改善站数|RMSE 改善站数|高功率 RMSE 变化|5 seeds 双指标均改善|','|---|---:|---:|---:|---|']
for family in ['W0','TF']:
 for mode in ['P1_FULL','P2_HEAD']:
  c=C[family+'_'+mode,family+'_P0_SHARED'];lines.append(f'|{family} {mode} 对 P0|{c["mae"]["wins"]}/55|{c["rmse"]["wins"]}/55|{c["high_power_rmse"]["change_percent"]:+.3f}%|是|')
lines+=['',
'这些比较均通过原先设定的选择集门槛：两个指标改善至少 0.1%、各自严格过半站改善、高功率 RMSE 退化不超过 0.1%。TF 全参数的高功率 RMSE 改善 1.316%，W0 改善 0.876%，不是通过牺牲高功率总体表现换取平均改善。',
'数值精度提示：W0 P1 的原始 MAE 胜场为 50/55，其中 2 站仅为复算浮点差异；按绝对差大于 1e-7 计为 48/55。W0 P1 相对 P2 的 MAE 胜场相同口径由 41/55 变为 40/55。预设判定未改，所有微调收益结论均不受影响。', '',
'全参数优于头部微调：W0 的额外改善为 MAE 0.285%、RMSE 0.246%；TF 为 0.327%、0.231%。每个 seed 都是双指标改善。这个结果支持在当前训练预算下优先使用全参数微调，不证明冻结内部响应的 P3 一定无效，也不证明尚未测试的 LoRA 无效。','',
'## 训练代价与适配幅度','', '|组别|有效可训练参数|占全模型参数|选中第 0 轮|最优轮数中位数|选中第 30 轮|','|---|---:|---:|---:|---:|---:|']
for family in ['W0','TF']:
 for mode in ['P1_FULL','P2_HEAD']:
  s=S[family+'_'+mode];lines.append(f'|{family} {mode}|{s["trainable_parameters"]:,}|{100*s["trainable_parameters"]/s["total_parameters"]:.2f}%|{s["epoch0_count"]}/275（{s["epoch0_percent"]:.1f}%）|{s["selected_epoch_median"]:.0f}|{s["selected_epoch30_count"]}/275|')
lines+=['',
'W0 P1 在 7 个站的全部 5 seeds 都退回第 0 轮；TF P1 没有这样的站。说明本站适配并非普遍必需，保留不微调候选是合理的。全参数最优轮数中位数只有 W0 1 轮、TF 2 轮，符合在共享模型附近作小幅适配的现象，但不能据此断言具体物理机理。',
'P2 参数很少（W0 约 1%、TF 约 2.4%），仍保留全参数 MAE 收益的约 60% 和 69%，RMSE 收益的约 60% 和 72%。若部署重视存储与更新成本，它是有竞争力的方案。P2 有 W0 23/275、TF 13/275 项在第 30 轮达到最好，部分任务仍可能受预算限制；相同学习率/轮数并不等于各方式都已调到最优。','',
'## W0 与 TF 的公平比较','', '|比较：W0 对 TF|MAE 变化|RMSE 变化|W0 MAE / RMSE 胜场|高功率 RMSE 变化|','|---|---:|---:|---:|---:|']
for mode in labels:
 c=C['W0_'+mode,'TF_'+mode];lines.append(f'|{labels[mode]}|{c["mae"]["change_percent"]:+.3f}%|{c["rmse"]["change_percent"]:+.3f}%|{c["mae"]["wins"]}/55、{c["rmse"]["wins"]}/55|{c["high_power_rmse"]["change_percent"]:+.3f}%|')
lines+=['',
'全参数微调后 W0 的 MAE 与 RMSE 均未超过同样微调的 TF；头部微调结果也一样。按两骨干各自统一选择集宏平均 MAE 选择，均应选 P1。不能比较“微调后的 W0”与“未微调的 TF”来声称优势。','',
'## 差距的种子敏感性','', '|seed|P1：W0 相对 TF MAE|P1：W0 相对 TF RMSE|','|---|---:|---:|']
c=C['W0_P1_FULL','TF_P1_FULL']
for i,seed in enumerate(range(42,47)):lines.append(f'|{seed}|{c["mae"]["per_seed_change_percent"][i]:+.4f}%|{c["rmse"]["per_seed_change_percent"][i]:+.4f}%|')
lines+=['',
'种子 45 对平均差距影响很大。只作敏感性检查、去掉该 seed 后，W0 对 TF 的 MAE/RMSE 差距由 +0.1655%/+0.3750% 缩至 +0.0155%/+0.0732%，均不足 0.1%。这不是剔除 seed 的依据，主表仍保留全部 5 seeds。也不能反向用剩余 seeds 证明两模型严格等效。',
'同时配对重采样 seed 与风场后，P1 的 W0 对 TF 差距区间为 MAE [-0.1687%, +0.5325%]，RMSE [-0.1573%, +1.0307%]，都跨 0。仅对风场重采样会低估共享 seed 带来的不确定性。这里重采样仍未校正风场空间相关、检查点选择偏差，属于描述性敏感性分析，不是独立显著性证据。',
'因此可靠的结论是“微调对两个骨干都有一致收益；TF 平均最好；W0 没有证明双指标优越性”。目前还不足以断言 TF 架构跨 seed 稳定优于 W0。','',
'## 时效与高功率','', '|全参数微调相对共享模型|0–1 h MAE / RMSE|1–3 h MAE / RMSE|3–6 h MAE / RMSE|','|---|---:|---:|---:|']
for family in ['W0','TF']:
 c=C[family+'_P1_FULL',family+'_P0_SHARED'];lines.append('|'+family+'|'+ '|'.join(f'{c["horizons"][hz]["mae"]:+.3f}% / {c["horizons"][hz]["rmse"]:+.3f}%' for hz in ['0_1h','1_3h','3_6h'])+'|')
lines+=['',
'两个骨干的微调收益都随时效增长，尤其 MAE；这与个性化校准更影响远时效的现象一致，但不能仅凭这张表确定是功率曲线、天气响应、运行状态还是其他偏差被校准。',
'P1 的 W0 对 TF：0–1 h 为 -0.505%/-0.334%，1–3 h 为 +0.085%/+0.283%，3–6 h 为 +0.321%/+0.495%。W0 仍有短时优势，远时效落后。高功率上 W0 相对自身改善，但相对 TF 仍差 0.920%，两个比较应区分。',
'逐站也有失败例：P1 相对各自共享模型，W0 的马力风电 RMSE +0.290%、来福 +0.197%、镇赉 +0.196%；TF 的马力 +0.136%、新立 +0.058%。均为按 MAE 选模后的 RMSE 退化，完整结果见 per_farm_final.csv。','',
'## 证据边界与贡献判断','',
'1. 第 0 轮共享检查点参与逐站 2023 MAE 选模，故验证 MAE 原则上不会比第 0 轮差；微调后的高胜率不能当成未经选择的泛化证据。RMSE 未参与选模、改善方向一致是有价值的筛选信号，但同样来自已反复用于选择的 2023。',
'2. 本轮直接支持的是“共享模型可以通过逐站微调进一步改善选择集指标”。没有同协议、同 55 站/区域输入的从零独立训练对照，不能宣称共享预训练 + 个性化优于独立训练。',
'3. 没有 P3 或 LoRA，也没有微调后的历史 × 天气 2×2、区域去除/错配。一般微调收益不能替代两个机制复验，不能为此前失败的参考轨迹方案提供证据。',
'4. 全参数平均更好是当前方案下的结果；头部微调更省参数，且部分任务达到轮数上限。仍不能据此证明风场差异必须进入内部天气响应，也不能证明分工式 P3 会成功。','',
'## 建议的下一步（本次只分析，未新增训练或读取 2024）','',
'优先冻结 P0/P1/P2、现有 5 seeds 的全部逐站检查点与评估代码，按相同 55 站、相同区域输入规则进行 2024 探索性复评（剔除 11 月），不根据 2024 重选微调轮次或方式。这一步可以复用现有模型，只需推理；先确认微调收益能否跨年保留。2024 已在项目历史中查看过，仍不能称为未触碰的最终测试。',
'若跨年保持收益，再用预先写定的规则扩到 10 seeds，判断两骨干排序是否稳定；不要因 seed 45 不利而删除它。若要主张共享预训练优势，补同协议的从零单站训练对照。之后才值得针对 P3/LoRA 做小规模、同预算筛选。',
'若论文继续采用全参数个性化模型，历史 × 天气交互和区域实况的去除/错配应在个性化后的最终模型及同样个性化的 TF 上复验，不能直接沿用微调前机制结论。','',
'## 输出文件','',
'- FINAL_AUDIT.json：1,100 项完整性、SHA256 与选模重算。',
'- PARAMETER_INTEGRITY.json：实际保存参数的冻结与本站嵌入核验。',
'- FINAL_ANALYSIS.json：增益、seed 敏感性与分时效结果。',
'- per_farm_final.csv / per_seed_final.csv：330 条逐站均值 / 30 条逐 seed 均值。',
'- per_station_seed.csv：1,650 条原始站点-seed 指标（含共享对照）。',
'- horizon_summary.csv / lead_summary.csv：18 条时效段 / 144 条逐时效均值。',
'- figures/personalization_gains.png / .pdf：时效增益图。','']
(R/'analysis/FINAL_REVIEW_CN.md').write_text('\n'.join(lines));atomic_json(R/'analysis/FINAL_DECISION.json',dict(timestamp=now(),complete=True,audit_passed=True,personalization_selection_gate_passed=True,chosen_modes={'W0':'P1_FULL','TF':'P1_FULL'},lowest_average_metrics='TF_P1_FULL',W0_over_same_personalized_TF_passed=False,seed_sensitivity_material=True,results_selection_year=2023,test2024_evaluated=False,matched_independent_baseline_available=False,new_training_started=False,next_recommended='freeze existing checkpoints and evaluate exploratory2024; then seed stability and matched independent controls'))
# Scientific export; preserves the per-farm/per-seed aggregation used in the report.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader((R/'analysis/lead_summary.csv').open()));series={g:{m:np.array([float(x[m]) for x in rows if x['group']==g]) for m in ['mae','rmse']} for g in S}
x=np.arange(1,25)/4;fig,axes=plt.subplots(2,2,figsize=(10,6.8),sharex=True,constrained_layout=True)
colors={'W0':'#b34d29','TF':'#226caa'}
for col,metric in enumerate(['mae','rmse']):
 ax=axes[0,col]
 for family in ['W0','TF']:
  for mode,style,label in [('P1_FULL','-','full'),('P2_HEAD','--','head')]:
   y=100*(series[family+'_'+mode][metric]/series[family+'_P0_SHARED'][metric]-1);ax.plot(x,y,color=colors[family],ls=style,label=family+' '+label,lw=1.8)
 ax.set_title(metric.upper()+': fine-tuning vs shared');ax.set_ylabel('Error change (%)');ax.legend(fontsize=8);ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2)
 ax=axes[1,col]
 for mode,style,label,color in [('P0_SHARED',':','Shared','#777777'),('P1_FULL','-','Full','#7a4e9b'),('P2_HEAD','--','Head','#32835a')]:
  ax.plot(x,100*(series['W0_'+mode][metric]/series['TF_'+mode][metric]-1),ls=style,label=label,color=color,lw=1.8)
 ax.set_title(metric.upper()+': W0 vs TF, matched mode');ax.set_ylabel('Error change (%)');ax.set_xlabel('Forecast horizon (h)');ax.legend(fontsize=8);ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2)
fig.suptitle('2023 selection set | 55 farms x 5 seeds | negative = lower error',fontsize=11)
folder=R/'analysis/figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'personalization_gains.png',dpi=180);fig.savefig(folder/'personalization_gains.pdf');plt.close(fig)
print('Final report and plots saved')
