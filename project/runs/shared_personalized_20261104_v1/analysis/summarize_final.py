from pathlib import Path
import sys,json,csv,collections
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'));from settings import atomic_json,now
manifest=json.loads((R/'manifest.json').read_text());groups=collections.defaultdict(list);audit=json.loads((R/'analysis/FINAL_COMPLETENESS_AUDIT.json').read_text());gate=json.loads((R/'analysis/GATES.json').read_text())
for s in manifest:
 if s['stage'] in ['baseline','mechanism']:groups[s['group']].append(json.loads((R/'jobs'/s['id']/'metrics.json').read_text()))
def val(g,m):return float(np.mean([d['metrics'][m] for d in groups[g]]))
def change(a,b,m):return 100*(val(a,m)/val(b,m)-1)
seedrows=[];hzrows=[]
for g,ds in groups.items():
 for d in sorted(ds,key=lambda x:x['spec']['seed']):
  m=d['metrics'];seedrows.append(dict(group=g,seed=d['spec']['seed'],checkpoint_epoch=d['checkpoint_epoch'],parameters=d['parameter_count'],mae=m['mae'],rmse=m['rmse'],high_power_rmse=m['high_power_rmse']))
  for hz,x in m['horizons'].items():hzrows.append(dict(group=g,seed=d['spec']['seed'],horizon=hz,mae=x['mae'],rmse=x['rmse']))
for name,rows in [('per_seed_metrics.csv',seedrows),('per_seed_horizon_metrics.csv',hzrows)]:
 with (R/'analysis'/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs={}
for c in gate['comparisons']:pairs[(c['candidate'],c['reference'])]=c
lines=['# 共享训练与参考轨迹机制：最终结果与验收','',f'汇总时间：{now()}。最后训练结果生成于 {audit["last_training_result"]}。','',
'## 完成与数据口径','',
'- 170/170 项训练全部完成：80 项调参、25 项参考模型（4 折 OOF + 全年模型，各 5 seeds）、25 项共享基线、40 项机制与对照。13 个正式组均齐全 5 seeds（42–46）。',
'- 55 个风场，2022 年训练、2023 年选择和评价（剔除 11 月及缓存既定边界），每站 29,647 个评价起报点。按预先冻结的功率质量规则剔除龙湖风电场，同时重算区域输入；本轮未进行 2024 模型评价，也未读取 2024 功率值进行筛选。',
'- 逐站 MAE/RMSE 先跨 seeds 平均，再等权跨站平均。RMSE 为各站各 seed 的 RMSE 均值，不是对跨站、跨 seed 全部平方误差直接开方。所有指标来自同一个按 2023 MAE 选择的检查点。',
'- 已核验所有成功标记、指标形状与有限值、样本数、检查点 SHA256、参考预测与 OOF 缓存 SHA256、数据冻结 SHA256，并独立重算全部验收比较。',
'- 汇总故障已修复：NumPy bool_ 不能直接写入 JSON。仅将判定转换为 Python bool；没有修改训练、数据、指标公式或判定门槛。此前报告停留在 147 项，不代表后续训练失败。','',
'## 主要指标','', '|组别|含义|MAE|RMSE|','|---|---|---:|---:|']
labels={'TF_BASE':'共享调优 TF，BASE 输入','TF_R3':'共享调优 TF + SEL + 区域实况','W0_R3':'共享 W0 + SEL + 区域实况','TIDE_R3':'共享调优 TiDE，同输入','TFT_R3':'共享调优 TFT，同输入','M1_CB':'W0 + 参考轨迹条件','M2_CBH':'M1 + 历史摘要','M3_ANCHOR':'M1 改用参考轨迹锚点','M4_WRONG_CB':'M1 的异日轨迹对照','M4H_WRONG_CBH':'M2 的异日轨迹与摘要对照','M5_TF_CB':'TF + 参考轨迹条件','M5W_TF_WRONG_CB':'TF 的异日轨迹对照','M5N_TF_NEUTRAL':'TF 的同参数中性条件对照'}
for g in labels:lines.append(f'|{g}|{labels[g]}|{val(g,"mae"):.6f}|{val(g,"rmse"):.6f}|')
lines+=['','## 预设判定','', '以下变化率均为“候选 / 对照 − 1”，负值更好；胜场数为跨 5 seeds 平均后比较的风场数。', '', '|候选 / 对照|MAE 变化|MAE 胜场|RMSE 变化|RMSE 胜场|高功率 RMSE 变化|通过全部门槛|','|---|---:|---:|---:|---:|---:|---|']
for (a,b),c in pairs.items():lines.append(f'|{a} / {b}|{c["mae"]["change_percent"]:+.4f}%|{c["mae"]["farms_improved"]}/55|{c["rmse"]["change_percent"]:+.4f}%|{c["rmse"]["farms_improved"]}/55|{c["high_power_rmse_change_percent"]:+.4f}%|{"是" if c["passed"] else "否"}|')
lines+=['', '**总体判定：FAIL_NO_LARGE_SCALE_PERSONALIZATION；没有可扩展候选。**', '',
'1. W0 对同输入 TF 的 MAE 改善 0.1848%，31/55 站改善，但 RMSE 变差 0.1687%，仅 19/55 站改善；高功率 RMSE 变差 0.4719%。共享训练没有恢复双指标优势。',
'2. M1 相对 W0 的 MAE/RMSE 分别变差 0.1107%/0.2035%，相对真实轨迹对应的异日对照也变差 0.1199%/0.1768%。不能声称参考轨迹条件提供有效增量。',
'3. M2 相对 M1 改善 0.1028%/0.1490%，但相对 W0 是 +0.0078%/+0.0542%，基本回到原模型。相对自身错配对照仅改善 0.0711%/0.0795%，未达 0.1% 门槛。不能把“修复 M1 的退化”写成优于基础模型的新机制。',
'4. M3 改参考轨迹锚点，相对 M1 的 MAE 变差 0.7677%，RMSE 仅改善 0.0171%，不合格。',
'5. M5 迁到 TF 后，相对 TF 的 MAE/RMSE 变差 0.2456%/0.2560%，5 个 seeds 的两个指标都更差；也差于同参数中性和异日条件对照。当前结果不支持可跨骨干迁移的参考轨迹机制。', '',
'## 时效与稳定性','', '|比较|0–1 h MAE / RMSE|1–3 h MAE / RMSE|3–6 h MAE / RMSE|','|---|---:|---:|---:|']
for a,b in [('W0_R3','TF_R3'),('M1_CB','W0_R3'),('M5_TF_CB','TF_R3')]:
 cols=[]
 for hz in ['0_1h','1_3h','3_6h']:
  dif=[100*(np.mean([d['metrics']['horizons'][hz][m] for d in groups[a]])/np.mean([d['metrics']['horizons'][hz][m] for d in groups[b]])-1) for m in ['mae','rmse']];cols.append(f'{dif[0]:+.4f}% / {dif[1]:+.4f}%')
 lines.append(f'|{a} / {b}|'+ '|'.join(cols)+'|')
lines+=['',
'W0 的短时优势仍在，3–6 h RMSE 相对 TF 变差 0.3102%。这是现象定位，不能据此认定输出锚点或天气融合方式是原因。',
'W0 对 TF 的站级 bootstrap 95% 区间：MAE [-0.4102%, +0.0115%]，RMSE [-0.0335%, +0.3598%]；两个区间均跨 0，种子方向也有变化。因此结论是“没有证明双指标优势”，不是证明两模型严格等效。站级 bootstrap 未建模空间相关，区间仅作描述。',
'M1 相对 W0 的站级 bootstrap 区间均为正：MAE [0.0252%, 0.2051%]、RMSE [0.1042%, 0.3116%]；4/5 seeds 的两个指标均变差。该实现的负结果比 W0/TF 的细小差异更一致。', '',
'## 能支持与不能支持的叙事','',
'- 共享 TF_R3 相对 TF_BASE：MAE 改善 2.3266%，RMSE 改善 2.1875%。共享 W0_R3 相对 TF_BASE：改善 2.5071%/2.0225%。说明本轮包含 SEL 与区域实况的整体方案有价值，但这不是单独的区域实况消融；两组还各自调参。不能把全部增益归于区域输入。',
'- 本轮没有历史 × 天气的 2×2，也没有共享模型的区域去除/错配，因此既不替代、也不推翻之前对历史—天气交互和区域补充的证据。它否定的是“当前参考轨迹条件实现在本业务协议共享设定下有增量”的预期。',
'- 旧协议 -5.1% 不能与本轮数值直接衔接：预报协议、B0 实现和输入、OOF 训练方式、风场集合都发生变化。当前 B0 是仅本站历史的共享 TCN + attention，不是旧全局历史模型。不能把失败原因唯一归于业务预报更旧。',
'- 参考轨迹由本站历史确定，可能与已有历史编码冗余；OOF 参考模型训练分布也可能与全年模型不同。这些是待检验解释，当前消融不能确定具体原因。', '',
'## 执行决策','',
'按冻结门槛，不启动约 2,800 次个性化，不扩到 10 seeds，不启动这一路线的 2024 复评。阶段 3–5 属于未触发的条件阶段，不应计为已完成或失败任务。',
'保留 W0_R3 与 TF_R3 作为已验证共享基线。若继续研究共享训练收益，优先在完全相同的 55 站、同区域输入、同选模协议下补共享/独立对照；若要研究通用个性化收益，应作为新的独立问题，公平地对两骨干采用相同微调方式，不能作为当前轨迹机制已成立后的扩展。以上为后续建议，本轮没有新增训练。', '',
'## 可复查文件','',
'- `FINAL_COMPLETENESS_AUDIT.json`：170 项逐任务核验与完成时间。',
'- `GATES.json`：原始门槛比较、逐 seed 变化与站级区间。',
'- `shared_summary.csv`：13 组完整指标。',
'- `per_farm_metrics.csv`：13 × 55 = 715 条逐场站指标。',
'- `per_seed_metrics.csv`：13 × 5 = 65 条逐 seed 指标。',
'- `per_seed_horizon_metrics.csv`：195 条分时效指标。','']
(R/'analysis/FINAL_REVIEW_CN.md').write_text('\n'.join(lines));atomic_json(R/'analysis/FINAL_DECISION.json',dict(timestamp=now(),gate=gate['stage'],eligible=[],training_complete=True,audit_passed=True,no_new_training=True,personalization_started=False,test2024_evaluated=False,report='analysis/FINAL_REVIEW_CN.md'))
print('Saved final review, seed metrics and horizon metrics')
