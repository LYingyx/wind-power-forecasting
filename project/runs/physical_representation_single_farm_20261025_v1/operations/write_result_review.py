"""Record the reviewed 2023 conclusions and freeze B1 comparisons only."""
from pathlib import Path
import json,hashlib,datetime
import pandas as pd
from review_completed_round import md,DEFAULT,TFSTAR
R=Path(__file__).resolve().parents[1];P=R.parents[1];A=R/'analysis'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    audit=json.loads((R/'COMPLETION_AUDIT.json').read_text());assert audit['passed']
    checks=json.loads((A/'POST_REVIEW_CHECKS.json').read_text());assert checks['source_freeze_unchanged']
    s=pd.read_csv(A/'key_comparison_effects.csv');pairs=pd.read_csv(A/'key_comparisons_per_farm_seed.csv')
    all_scores=pd.read_csv(A/'all_actual_per_farm_seed.csv')
    supplementary=pd.read_csv(A/'supplementary_effect_sizes.csv');scenarios=pd.read_csv(A/'scenario_effect_sizes.csv')
    def table(labels):
        rows=[]
        for label,title in labels:
            a=s[(s.comparison==label)&(s.metric=='mae')].iloc[0];b=s[(s.comparison==label)&(s.metric=='rmse')].iloc[0]
            rows.append({'比较（前者对后者）':title,'MAE变化':f'{a.effect_percent:+.4f}%','RMSE变化':f'{b.effect_percent:+.4f}%','MAE改善站seed42/43':f'{a.improved_seed42}/56；{a.improved_seed43}/56','RMSE改善站seed42/43':f'{b.improved_seed42}/56；{b.improved_seed43}/56'})
        return md(pd.DataFrame(rows))
    a1=table([('W0_'+v+'_vs_BASE',v+' / BASE') for v in ['SPD','SPD3','SPD_LV','SPD3_LV','RHO','DENS_NORHO','DENS_UNCOUPLED','DENS']])
    a2=table([('W0_SEL_vs_'+v,'SEL / '+v) for v in ['BASE','DENS','ALL','SEL_NO_DENS','SEL_NO_DYN','SEL_NO_STAB','SEL_NO_STATE','SEL_DENS_NORHO','SEL_DENS_UNCOUPLED','SEL_REF','SEL_REF_DELTA']])
    a3=table([(k,title) for k,title in [('TFstar_vs_TFdefault_BASE','TF* BASE / 默认TF BASE'),('TFstar_vs_TFdefault_DENS','TF* DENS / 默认TF DENS'),('TFstar_DENS_vs_BASE','TF* DENS / TF* BASE'),('W0_vs_TFdefault_BASE','W0 BASE / 默认TF BASE'),('W0_vs_TFstar_BASE','W0 BASE / TF* BASE'),('W0_vs_TFstar_DENS','W0 DENS / TF* DENS'),('W0_vs_TFdefault_SEL','W0 SEL / 默认TF SEL')]])
    horizons=supplementary[(supplementary.contrast=='input_vs_default_BASE')&(supplementary.candidate_configuration=='W0')&(supplementary.candidate_variant.isin(['DENS','SEL']))&(supplementary.selection=='mae')&(supplementary.metric=='mae')][['candidate_variant','horizon','effect_percent','improved_seed_pairs']].copy()
    horizons.effect_percent=horizons.effect_percent.map(lambda v:f'{v:+.4f}%')
    strata=scenarios[(scenarios.horizon=='all')&(scenarios.metric=='mae')&(scenarios.group.isin(['nwp_speed','forecast_wind_change_1h']))&(scenarios.comparison.isin(['W0_DENS_vs_BASE','W0_SEL_vs_BASE']))][['comparison','group','bin','candidate_equal_farm','reference_equal_farm','effect_percent','improved_seed_pairs']].copy()
    strata.effect_percent=strata.effect_percent.map(lambda v:f'{v:+.4f}%')
    for k in ['candidate_equal_farm','reference_equal_farm']:strata[k]=strata[k].map(lambda v:f'{v:.8f}')
    failures=pairs[(pairs.comparison=='W0_SEL_vs_BASE')&(pairs.mae_difference>=0)][['site','seed','mae_candidate','mae_reference','mae_difference','rmse_difference']].copy()
    for c in failures.columns[2:]:failures[c]=failures[c].map(lambda v:f'{v:.10f}')
    models=[dict(configuration=c,variant=v,existing_seeds=[42,43] if (c,v)!=(TFSTAR,'SEL') else [],source_run=('physics_observation_single_farm_20261024_v1' if v=='BASE' and c in ['W0',DEFAULT] else R.name) if (c,v)!=(TFSTAR,'SEL') else None) for c in ['W0',DEFAULT,TFSTAR] for v in ['BASE','SEL']]
    assert all(len(all_scores[(all_scores.configuration==m['configuration'])&(all_scores.variant==m['variant'])&(all_scores.selection=='mae')&(all_scores.horizon=='all')])==(112 if m['existing_seeds'] else 0) for m in models)
    evidence=[R/'SOURCE_FREEZE.json',R/'PROTOCOL.json',R/'COMPLETION_AUDIT.json',A/'SELECTION_RECOMMENDATION.json',A/'all_actual_per_farm_seed.csv',A/'key_comparison_effects.csv',A/'scenario_effect_sizes.csv']
    freeze=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),scope='B1 main-table comparisons only; B2 interventions need their own protocol',selected_on='2023 only; train 2022',main_variant='SEL',main_groups=['DENS','DYN','STAB','STATE'],main_extra_dimensions=22,selected_tf_configuration=TFSTAR,selection_rule='W0 equal-farm two-seed MAE minimum; TF configuration chosen on DENS validation MAE; no CI-as-equivalence or percentage cutoff',models=models,seeds=list(range(42,52)),farms=list(range(56)),loss='MAE',main_checkpoint='2023_min_MAE',secondary_checkpoint='2023_min_RMSE',primary_metric='all_horizon_MAE',secondary_metrics=['RMSE','bias','0_1h','1_3h','3_6h','predeclared_scenarios','per_farm_per_seed'],new_training_required=2800,model_farm_seed_evaluations=3360,missing_seed42_43_cell=dict(configuration=TFSTAR,variant='SEL',tasks=112,included_in_2800=True),training_completed_for_B1=False,ready_for_2024_evaluation=False,test_year_accessed_this_round=False,test_label='2024 retrospective evaluation after this freeze; previously analyzed in project',evidence_sha256={str(p):sha(p) for p in evidence},guard='Do not interpret A completion as B completion. Keep default TF and TF* comparisons regardless of transfer-cell result. No 2024-driven changes.')
    target=R/'FROZEN_SELECTION.json'
    if target.exists():
        existing=json.loads(target.read_text());assert existing['main_variant']==freeze['main_variant'] and existing['selected_tf_configuration']==freeze['selected_tf_configuration']
    else:target.write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    body=f'''# 本轮完成审计、归因结论与 B1 选型

日期：2026-10-25。A阶段现已全部完成：6272项新增训练、672项旧权重诊断，共6944项，未解决失败0。新训练约06:09完成；CPU重放一项RMSE差异略超容差，16:13用CUDA补齐；16:16全部审计通过。9461份不同选定权重及冻结源文件核验一致。两台V100均无遗留训练、调度进程或权重副本，远端分别归档2994/3247项，本机完成31项训练。

CPU失败不是训练失败：站29/seed42/TF-DENS同一权重在CPU的RMSE差异−2.3119e−6，CUDA仅+6.7492e−9；保持2e−6容差，未改模型、未重训。证据见 [恢复记录](../operations/REFERENCE_REPLAY_RECOVERY.json)。

## 原值和统计口径

- [关键模型逐风场逐seed可读表](KEY_MODELS_PER_FARM_SEED_CN.md)：8个关键模型，每个56站×2seed，含实际MAE/RMSE/bias及选中轮次。
- [全部原值55552行](all_actual_per_farm_seed.csv)：62种配置、两种选轮口径、四个时效段；不以均值替代原值。
- [全部配对差181888行](paired_per_farm_seed.csv)、[关键配对差3472行](key_comparisons_per_farm_seed.csv)。
- [分场景原值611072行](diagnostics_per_farm_seed.csv)、[分组补充效应量](scenario_effect_sizes.csv)。
- [模型目录](MODEL_CATALOG.csv)、[真实结构与各输入定义](../../../NEXT_ROUND_REVIEW_AND_EXECUTION_20261025_CN.md)、[自动完整排序](REPORT_CN.md)。

以下百分比是补充汇总：每站内seed等权、56站再等权，(候选/参考−1)×100，负值改善。每seed改善站数单独列出，不设最小改善门槛。默认MAE训练/MAE选轮/2023全时效。站点bootstrap仅探索性描述，未修正空间相关和多候选选择；区间跨零不等于等效。

## A1：主体收益来自显式风速表示，密度增量较小

以下均为 W0：

{a1}

SPD是10m风速模长；SPD3是其立方；LV表示10m/1000/925hPa三层。DENS含相对密度、密度修正风速、rho·v³；DENS_NORHO固定密度，仍保留风速和立方；DENS_UNCOUPLED保留密度及风速/立方，但去掉密度乘法耦合；RHO仅提供密度。A1均3个名义投影槽，单变量用零槽补齐，有效输入维数并不相同。

**支持的解释**：显式模长以及组合表示很有价值。仅SPD就使MAE改善1.1120%，固定密度的DENS_NORHO改善1.2127%。不能把DENS全部1.3356%的收益都写为空气密度修正。

**密度不是完全没有收益**：DENS相对DENS_NORHO的MAE/RMSE进一步改善0.1245%/0.1441%；相对DENS_UNCOUPLED则是0.0456%/0.0035%。后者RMSE区间跨零，耦合增量较弱。完整SEL相对SEL_DENS_NORHO的MAE优势只有0.0229%，仍如实计为改善，不把小差异说成等效或强物理证明。

**不支持“v³是核心”或“网络学不会”**：SPD3比SPD差，多层立方也不优于多层模长。这证明了本训练设置下表示选择影响优化与泛化，不证明网络表达能力不够，更不能单凭此确认数据量是唯一原因。

## A2：按事前MAE规则，主变体选择SEL

SEL=DENS(3)+DYN(7)+STAB(5)+STATE(7)，共22维额外输入。前15维为物理天气代理，后7维是本站历史运行状态描述；不是22维纯天气量。W0保留NEUTRAL_LAST的历史记忆和最后观测锚点，物理/状态向量经零初始化Linear(22,64)加入天气查询。

{a2}

SEL相对W0 BASE的MAE在seed42/43分别改善53/55站，RMSE改善49/51站。站内seed平均后是54/56站MAE、53/56站RMSE改善。SEL也同时优于DENS和ALL。去掉DENS、DYN、STAB、STATE均损失部分收益，因此保留四组。该结论是组合消融，不把各组增益简单相加或当成因果机制。

SEL与解除耦合/固定密度版本的差距很小，但预先规定按MAE最低选择，故仍保留SEL，不能事后用区间重叠改选更简单者。默认TF的最佳输入是SEL_REF_DELTA，而W0最佳是SEL，说明最优组合依赖骨干；主表仍使用共同SEL输入以保证比较条件一致。

SEL相对BASE的4个MAE退化配对如下（站点编号沿用0–55）：

{md(failures)}

## A3：TF训练设置解释了大部分原始骨干差距

TF*=`{TFSTAR}`：输出末层零初始化、骨干学习率2e−4、dropout0.05，新增物理投影学习率仍2e−4。配置严格按DENS上的2023MAE最低选出，默认TF保留为对照。

{a3}

在BASE同输入下，W0相对默认TF的MAE优势从2.4941%缩为相对TF*的0.6762%；RMSE优势从1.0381%缩为0.2132%，后者仅26/56个风场的站内seed均值改善，不能称普遍占优。DENS同输入下W0仍有MAE1.5798%、RMSE1.0231%的汇总优势。

TF*内部DENS带来的增益为MAE0.4297%、RMSE0.3495%，不能把TF*对默认TF的全部收益归给物理输入。按DENS的RMSE最小选择会得到另一配置（随机头、lr2e−4、dropout0.05），这是指标权衡，主比较仍按事前MAE规则。

**还没有TF*+SEL的56站×2seed结果**。因此W0+SEL对默认TF+SEL的−3.8181%/−2.0107%不能替代对优化后TF*+SEL的比较。该112项已计入下一阶段所需训练，不把缺失格子当成模型优势证据。

## A4：收益主要在1–6h；预设物理解释需收紧

下表为相对W0 BASE的分时效MAE补充变化，改善配对数分母112：

{md(horizons)}

风速四分位q1–q4及预报风速变化分组均用2022边界。以下为分组MAE，candidate/reference为补充等权均值；改善配对分母112：

{md(strata)}

DENS在q1的相对改善最大，但q4的绝对MAE降低更大，不能只据百分比断言低风速贡献最大。SEL在q3相对改善更突出。两者在预报变化平缓组都有明显收益，不能写成只在爬坡起作用；这些分组也不是风机的切入/额定区间。当前结果支持描述适用区间，不足以证明“功率曲线斜率导致增益”。

## 已冻结的B1清单与尚未执行部分

人工审查已按事前规则写入 [FROZEN_SELECTION.json](../FROZEN_SELECTION.json)，**范围仅B1主表**：W0/默认TF/TF*分别做BASE与SEL，56站、seed42–51。主指标全时效MAE，RMSE及三个时效段、既定场景为次要指标；主权重按2023MAE选取，RMSE选轮结果单列。

6模型×56站×10seed共3360个模型-站点-seed评价格子。当前可复用560个训练格子，尚需2800次训练；其中112项是TF*+SEL的seed42/43，其余2688项是六模型的8个新seed。该清单不会因TF*+SEL的验证结果而删除默认TF或改换主输入。**B1训练及2024复评尚未启动，A完成不等于B完成。**

B2历史×天气2×2须单独冻结干预实现：历史记忆、输出锚点及STATE同步替换；天气及其DENS/DYN/STAB等派生路径同步替换，并明确重新训练或固定权重干预。当前结果不证明新输入下交互仍成立，不能提前写入贡献。2024在项目早期已分析，后续仍称“冻结后的回顾性复评”。

## 贡献表述建议

1. 方法候选：面向单风场预测的显式风速/物理天气表示与历史运行状态补充，低成本加入既有预测骨干。强调受控结果，不主张密度主导或新物理定律。
2. 归因证据：风速模长、立方、密度及耦合的拆分；同槽消融；TF初始化/学习率/dropout因素控制。明确参数量相同不等于有效容量完全相同。
3. 适用边界：逐站逐seed、1–6h和分场景收益及失败风场；跨年与历史×天气交互仍待B阶段。尚不能把2023选型成绩写成跨年方法优势，也不预设达到顶会的方法新颖性标准。
'''
    (A/'ATTRIBUTION_REVIEW_20261025_CN.md').write_text(body)
    root=f'''# 物理表示筛选完成：结果入口

2026-10-25：本轮A阶段6944/6944项全部完成（6272训练+672旧权重诊断），未解决失败0，9461份选定权重核验通过。

阅读 [归因审查与已冻结B1清单](runs/{R.name}/analysis/ATTRIBUTION_REVIEW_20261025_CN.md)，然后查看 [逐风场逐seed实测表](runs/{R.name}/analysis/KEY_MODELS_PER_FARM_SEED_CN.md)、[全部原值CSV](runs/{R.name}/analysis/all_actual_per_farm_seed.csv) 和 [完成审计](runs/{R.name}/COMPLETION_AUDIT.json)。

主输入选SEL=DENS+DYN+STAB+STATE；2023相对W0 BASE的补充等权MAE/RMSE为−2.3525%/−2.0335%，逐seed改善站数分别53/55和49/51。主体收益不应归给密度或v³；优化TF后骨干差距缩小。B1选型已冻结，但还缺TF*+SEL的112个既有seed格子及2688个新seed训练；2024尚未运行，不能把A完成写成全项目完成。
'''
    (P/'PHYSICAL_REPRESENTATION_RESULTS_20261025_CN.md').write_text(root)
    print('Wrote complete attribution review and B1-only selection freeze')

if __name__=='__main__':main()
