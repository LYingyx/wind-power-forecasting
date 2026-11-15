"""Describe paired validation outcomes without averaging scores across farms/seeds."""
from pathlib import Path
import collections,csv,json,sys
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now
OUT=RUN/'analysis'


def table(path,rows):
    if not rows:return
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def main():
    manifest=json.loads((RUN/'manifest.json').read_text());specs={s['id']:s for s in manifest}
    protocol=json.loads((RUN/'PROTOCOL.json').read_text())
    actual=list(csv.DictReader((OUT/'new_actual_per_farm_seed.csv').open()))
    pairs=list(csv.DictReader((OUT/'new_paired_per_farm_seed.csv').open()))
    idx={(int(r['site']),int(r['seed']),r['backbone'],r['loss'],r['variant'],r['capacity_control']=='True',int(r['delay_minutes']),r['selection'],r['horizon']):r for r in actual}
    additional=[]
    for a in actual:
        refs={'ALL':'DENS','NEIGH_GENERIC':'CAL','NEIGH_ADVECT':'CAL'}
        if a['capacity_control']=='True' or a['delay_minutes']!='15' or a['variant'] not in refs:continue
        k=(int(a['site']),int(a['seed']),a['backbone'],a['loss'],refs[a['variant']],False,15,a['selection'],a['horizon'])
        b=idx.get(k)
        if b is None:continue
        additional.append(dict(site=int(a['site']),seed=int(a['seed']),backbone=a['backbone'],loss=a['loss'],selection=a['selection'],horizon=a['horizon'],candidate=a['id'],reference=b['id'],candidate_mae=a['mae'],reference_mae=b['mae'],candidate_rmse=a['rmse'],reference_rmse=b['rmse'],mae_difference=float(a['mae'])-float(b['mae']),rmse_difference=float(a['rmse'])-float(b['rmse'])))
    table(OUT/'screening_additional_pairs_per_farm_seed.csv',additional)
    source_pair_rows=len(pairs);pairs+=additional
    grouped=collections.defaultdict(list)
    fields=['candidate_backbone','candidate_variant','candidate_loss','candidate_capacity_control','candidate_delay_minutes','reference_backbone','reference_variant','reference_loss','reference_capacity_control','reference_delay_minutes','seed','selection','horizon']
    def key(row):
        a,b=specs[row['candidate']],specs[row['reference']]
        return tuple(a[k] for k in ['backbone','variant','loss','capacity_control','delay_minutes'])+tuple(b[k] for k in ['backbone','variant','loss','capacity_control','delay_minutes'])+(int(row['seed']),row['selection'],row['horizon'])
    for row in pairs:grouped[key(row)].append(row)
    counts=[]
    for k,rows in sorted(grouped.items()):
        record=dict(zip(fields,k));record['farms']=len(rows)
        for metric in ['mae','rmse']:
            diffs=[float(r[metric+'_difference']) for r in rows]
            for name,fn in [('better',lambda x:x<0),('equal',lambda x:x==0),('worse',lambda x:x>0)]:record[metric+'_'+name]=sum(fn(x) for x in diffs)
            record[metric+'_better_sites']=';'.join(str(r['site']) for r in rows if float(r[metric+'_difference'])<0)
        record['both_better']=sum(float(r['mae_difference'])<0 and float(r['rmse_difference'])<0 for r in rows)
        counts.append(record)
    table(OUT/'screening_comparison_counts_by_seed.csv',counts)
    consistent=[]
    for k,rows in sorted(grouped.items()):
        if k[-3]!=42:continue
        other=grouped.get(k[:-3]+(43,)+k[-2:])
        if other is None:continue
        a={str(r['site']):r for r in rows};b={str(r['site']):r for r in other};sites=sorted(a.keys()&b.keys(),key=int)
        rec=dict(zip(fields,k));rec.pop('seed');rec['paired_farms']=len(sites)
        for metric in ['mae','rmse']:
            win=[s for s in sites if float(a[s][metric+'_difference'])<0 and float(b[s][metric+'_difference'])<0]
            lose=[s for s in sites if float(a[s][metric+'_difference'])>0 and float(b[s][metric+'_difference'])>0]
            rec[metric+'_both_seeds_better']=len(win);rec[metric+'_both_seeds_worse']=len(lose)
            rec[metric+'_both_seeds_better_sites']=';'.join(win)
        consistent.append(rec)
    table(OUT/'screening_both_seed_consistency.csv',consistent)
    # Selected-MAE and selected-RMSE checkpoints are distinct estimators, never mixed.
    selection=[];interventions=[]
    for spec in manifest:
        m=json.loads((JOBS/spec['id']/'metrics.json').read_text())
        for horizon in ['all','0_1h','1_3h','3_6h']:
            vals=[]
            for criterion in ['mae','rmse']:
                v=m['selections'][criterion]['metrics'];vals.append(v if horizon=='all' else v['horizons'][horizon])
            selection.append(dict(id=spec['id'],site=spec['site'],seed=spec['seed'],backbone=spec['backbone'],variant=spec['variant'],loss=spec['loss'],capacity_control=spec['capacity_control'],delay_minutes=spec['delay_minutes'],horizon=horizon,
                mae_selected_epoch=m['selections']['mae']['epoch'],rmse_selected_epoch=m['selections']['rmse']['epoch'],mae_selected_mae=vals[0]['mae'],mae_selected_rmse=vals[0]['rmse'],rmse_selected_mae=vals[1]['mae'],rmse_selected_rmse=vals[1]['rmse'],mae_difference=vals[1]['mae']-vals[0]['mae'],rmse_difference=vals[1]['rmse']-vals[0]['rmse']))
            for name,v in m['interventions'].items():
                v=v if horizon=='all' else v['horizons'][horizon];base=vals[0]
                interventions.append(dict(id=spec['id'],site=spec['site'],seed=spec['seed'],backbone=spec['backbone'],variant=spec['variant'],loss=spec['loss'],delay_minutes=spec['delay_minutes'],selection='mae',horizon=horizon,intervention=name,normal_mae=base['mae'],normal_rmse=base['rmse'],intervened_mae=v['mae'],intervened_rmse=v['rmse'],mae_difference=v['mae']-base['mae'],rmse_difference=v['rmse']-base['rmse']))
    table(OUT/'selection_effect_per_farm_seed.csv',selection)
    table(OUT/'neighbor_interventions_per_farm_seed.csv',interventions)
    # Same neighbor model and seed at 0/30 versus 15 minute observation latency.
    latency=[]
    for spec in manifest:
        if spec['stage']!='latency':continue
        for criterion in ['mae','rmse']:
            for horizon in ['all','0_1h','1_3h','3_6h']:
                k=(spec['site'],spec['seed'],spec['backbone'],spec['loss'],spec['variant'],False)
                a=idx[k+(spec['delay_minutes'],criterion,horizon)];b=idx[k+(15,criterion,horizon)]
                latency.append(dict(site=spec['site'],seed=spec['seed'],backbone=spec['backbone'],variant=spec['variant'],loss=spec['loss'],delay_minutes=spec['delay_minutes'],reference_delay_minutes=15,selection=criterion,horizon=horizon,candidate=a['id'],reference=b['id'],candidate_mae=a['mae'],reference_mae=b['mae'],candidate_rmse=a['rmse'],reference_rmse=b['rmse'],mae_difference=float(a['mae'])-float(b['mae']),rmse_difference=float(a['rmse'])-float(b['rmse'])))
    table(OUT/'neighbor_latency_paired_per_farm_seed.csv',latency)
    assert len(selection)==3432*4 and len(latency)==96*8
    # Count helper: each cell counts strict paired wins; values themselves are not averaged.
    def wins(a,b,bb='W0',loss='MAE',seed=42,cap=False,refcap=False,refbb=None,refloss=None,criterion='mae',horizon='all'):
        out=[]
        for site in range(56):
            ra=idx.get((site,seed,bb,loss,a,cap,15,criterion,horizon))
            rb=idx.get((site,seed,refbb or bb,refloss or loss,b,refcap,15,criterion,horizon))
            if ra and rb:out.append((site,ra,rb))
        return dict(n=len(out),mae=sum(float(x['mae'])<float(y['mae']) for _,x,y in out),rmse=sum(float(x['rmse'])<float(y['rmse']) for _,x,y in out),both=sum(float(x['mae'])<float(y['mae']) and float(x['rmse'])<float(y['rmse']) for _,x,y in out),rows=out)
    def cell(w):return f"{w['mae']}/{w['n']}；{w['rmse']}/{w['n']}"
    lines=['# 物理与观测增强实验：2023筛选完成与结果入口','',f'生成时间：{now()}。本轮3432项训练已完成；完成审计见 [COMPLETION_AUDIT](../COMPLETION_AUDIT.json)。本报告不跨风场或seed平均误差；任何严格降低均计改善，同时保留退化与持平。下列表格中的分子是获胜风场数，不是误差均值，也不是显著性检验。','',
        '## 完成范围与数据入口','',
        '| 实验阶段 | 完成/计划 |','| --- | ---: |']
    for label,stage in [('输入特征筛选','input'),('损失函数对照','loss'),('同维度参数容量对照','capacity'),('结构机制对照','mechanism'),('邻站观测延迟','latency')]:
        n=sum(s['stage']==stage for s in manifest);lines.append(f'| {label} | {n}/{n} |')
    lines += ['', '训练2022、验证选择2023；本轮未运行2024复评。输入与原骨干损失对照覆盖56站；结构机制及容量/延迟对照只覆盖既定6代表站（索引6、8、14、16、43、46）。两种选权重规则分别保留。', '',
        '- [逐风场逐seed实际误差](new_actual_per_farm_seed.csv)：27456行，含MAE/RMSE选权重、全时效及三个时效段。',
        '- [逐风场逐seed原有配对差](new_paired_per_farm_seed.csv)：47520行，负差为改善。',
        '- [补充直接配对](screening_additional_pairs_per_farm_seed.csv)：ALL相对DENS、邻站模型相对CAL的原值与差值，无需重新训练。',
        '- [逐seed胜/平/负计数](screening_comparison_counts_by_seed.csv)及[两个seed均改善的站点](screening_both_seed_consistency.csv)：不混合seed成绩。',
        '- [同一训练轨迹的选权重效应](selection_effect_per_farm_seed.csv)：比较MAE最优与RMSE最优轮次，不能把验证集择优当新模型证据。',
        '- [邻站干预原值](neighbor_interventions_per_farm_seed.csv)及[0/30分钟相对15分钟延迟原值](neighbor_latency_paired_per_farm_seed.csv)。', '',
        '## 实际结构与缩写','',
        'W0：NEUTRAL_LAST活跃骨干，包含历史编码器、E2C固定点天气编码、条件适配器、历史交叉注意力解码器和末次观测残差输出。TF：64维Transformer未来协变量基线，历史编码器、天气查询解码器及末次观测残差头。两者按站独立训练。新增普通输入均经零初始化线性投影加到64维查询。', '',
        '| 变体 | 实际输入/结构 |','| --- | --- |']
    for name,description in protocol['descriptions'].items():lines.append(f'| {name} | {description} |')
    lines += ['', 'DENS的三个输入准确为：ρ/1.225、(ρ/1.225)^(1/3)×10m风速/10、ρ×10m风速³/1000。它同时引入密度、风速重参数化和风能代理；当前收益不能单独归因于空气密度。容量对照为同形状投影输入既有特征的确定性展开，未单独控制v、v³或恒定密度。', '',
        'MAE/MSE/MIX是训练损失；MIX=0.5 MAE+0.5 MSE/0.1。所有损失均按验证MAE耐心10早停。表内默认MAE选权重，RMSE选权重的完整对照在CSV中，不能混为同一协议。', '',
        '## 输入增强：56站、同损失、同选权重','',
        '单元格为“MAE改善站数；RMSE改善站数”。所有行均为训练MAE、选权重MAE、全0–6h。','',
        '| 候选 / 参考 | W0 seed42 | W0 seed43 | TF seed42 | TF seed43 |','| --- | --- | --- | --- | --- |']
    for variant in ['CAL','REF','DELTA','DENS','STAB','DYN','STATE','CYCLE','ALL']:
        lines.append('| '+variant+' / BASE | '+' | '.join(cell(wins(variant,'BASE',bb=bb,seed=seed)) for bb in ['W0','TF'] for seed in [42,43])+' |')
    lines.append('| DELTA / REF | '+' | '.join(cell(wins('DELTA','REF',bb=bb,seed=seed)) for bb in ['W0','TF'] for seed in [42,43])+' |')
    lines.append('| ALL / DENS | '+' | '.join(cell(wins('ALL','DENS',bb=bb,seed=seed)) for bb in ['W0','TF'] for seed in [42,43])+' |')
    lines += ['', 'DENS与ALL的改善覆盖面较广，优先保留为后续候选；CAL、CYCLE和时间位移不能因物理动机而预设有效。ALL胜过BASE只说明组合有效，不能由此断言每个组成部分都必要。', '',
        'DENS、ALL相对BASE在不同预测时段的表现如下，W0骨干、MAE训练/选权重。', '',
        '| 候选 | 时效 | seed42：MAE；RMSE | seed43：MAE；RMSE |','| --- | --- | --- | --- |']
    for variant in ['DENS','ALL']:
        for horizon in ['0_1h','1_3h','3_6h']:
            lines.append(f'| {variant} | {horizon} | '+' | '.join(cell(wins(variant,'BASE',seed=seed,horizon=horizon)) for seed in [42,43])+' |')
    lines += ['',
        '## 容量与强基线公平性','',
        '以下仍使用MAE训练/MAE选权重。容量对照只有6代表站；同输入W0/TF比较覆盖56站。','',
        '| 对照 | seed42：MAE；RMSE | seed43：MAE；RMSE |','| --- | --- | --- |']
    for variant in ['DENS','ALL']:
        for bb in ['W0','TF']:
            lines.append(f'| {bb} {variant} / 同维度容量对照 | '+' | '.join(cell(wins(variant,variant,bb=bb,seed=seed,refcap=True)) for seed in [42,43])+' |')
    for variant in ['BASE','DENS','ALL']:
        lines.append(f'| W0 {variant} / TF {variant}，同输入 | '+' | '.join(cell(wins(variant,variant,seed=seed,refbb='TF')) for seed in [42,43])+' |')
    lines += ['', '即使增强W0超过原TF，也应同时比较增强TF；不能把输入优势归给骨干。每站/seed的误差大小与退化范围应查原值表。', '',
        '## 损失函数与RMSE','',
        '原骨干BASE同输入，选权重仍为MAE；单元格分别为MAE、RMSE改善站数。','',
        '| 训练损失候选 / MAE训练 | W0 seed42 | W0 seed43 | TF seed42 | TF seed43 |','| --- | --- | --- | --- | --- |']
    for loss in ['MSE','MIX']:
        lines.append('| '+loss+' | '+' | '.join(cell(wins('BASE','BASE',bb=bb,loss=loss,refloss='MAE',seed=seed)) for bb in ['W0','TF'] for seed in [42,43])+' |')
    lines += ['', '换损失、换选权重、换输入是三个独立因素。不能把MSE训练或RMSE选轮带来的验证RMSE改善记作物理机制贡献；同一验证集选取的效果也需后续冻结方案后复评。', '',
        '## 结构性假说：6代表站','',
        '默认15分钟邻站观测延迟。单元格为MAE、RMSE改善站数；每个seed单列。NEIGH改变了输入协议，不能当作严格仅本站输入方法。','',
        '| 候选 / 参考 | 训练损失 | W0 seed42 | W0 seed43 | TF seed42 | TF seed43 |','| --- | --- | --- | --- | --- | --- |']
    for a,b in [('HIST','BASE'),('SHIFT_FIXED','SHIFT_ZERO'),('SHIFT_SAMPLE','SHIFT_FIXED'),('SHIFT_SAMPLE','HIST'),('CURVE_FREE','PC_INPUT'),('CURVE_BIAS','CURVE_FREE'),('CURVE_BIAS','PC_INPUT'),('NEIGH_GENERIC','BASE'),('NEIGH_GENERIC','CAL'),('NEIGH_ADVECT','CAL'),('NEIGH_ADVECT','NEIGH_GENERIC')]:
        for loss in ['MAE','MSE']:
            if b=='CAL' and loss=='MSE':continue
            lines.append(f'| {a} / {b} | {loss} | '+' | '.join(cell(wins(a,b,bb=bb,loss=loss,seed=seed)) for bb in ['W0','TF'] for seed in [42,43])+' |')
    lines += ['', 'SHIFT_SAMPLE相对SHIFT_FIXED缺少跨seed、跨骨干一致优势，暂不能主张样本条件时间校准已成立。曲线与传播先验存在部分改善，也存在退化；从这6站不能推出普遍有效。NEIGH_GENERIC相对BASE还含日历输入，因此另列相对CAL的邻站增量；MSE分支没有同CAL训练参考。', '',
        '邻站身份置换为固定roll，历史干预为逆序，方向干预为180°旋转；它们不是随机样本置换，也不是自然实验。干预后误差上升说明模型依赖这部分输入，不能单独证明真实大气传播机制。', '',
        '## 预选6代表站的实际误差示例','',
        'W0，MAE训练与MAE选权重，全0–6h；每行一个风场和seed，无跨seed平均。显示6位小数，胜负判断使用CSV完整精度。','',
        '| 站索引 | seed | BASE MAE | DENS MAE | ALL MAE | BASE RMSE | DENS RMSE | ALL RMSE |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for site in protocol['representatives']:
        for seed in [42,43]:
            vals=[idx[(site,seed,'W0','MAE',v,False,15,'mae','all')] for v in ['BASE','DENS','ALL']]
            lines.append(f'| {site} | {seed} | '+' | '.join(f'{float(v[k]):.6f}' for k in ['mae','rmse'] for v in vals)+' |')
    lines += ['', '## 当前可支持的表述与下一步','',
        '可保留的研究主线是“单风场历史状态与天气信息表示的受控增强”。当前新增证据优先支持DENS特征组合与ALL组合的验证收益，不支持预先把自适应空间提取、样本时间位移或上游传播定为核心贡献。DENS并不等于单独密度修正，ALL也不等于每项物理特征均有效。', '',
        '后续应先冻结具体选择与比较清单，再做跨年回顾性复评。若继续归因，最直接的缺口是：DENS中密度、原风速/风速³、密度校正风速的分拆，以及ALL逐项去除；ALL相对DENS的直接对照已经列出。复评仍应保留同输入、同损失、同选权重的TF比较。容量结果只覆盖6站。邻站扩展需同时审查原值干预与延迟结果，不把胜场多或某个seed改善当作显著机制。', '',
        '本报告完成2023筛选结果的汇总与初步解释，后续归因、跨年复评和论文更新仍需据此确定具体方案。']
    (OUT/'SCREENING_RESULTS_20261025_CN.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'SCREENING_SUMMARY.json',dict(timestamp=now(),source_jobs=len(manifest),actual_rows=len(actual),original_paired_rows=source_pair_rows,additional_paired_rows=len(additional),comparison_count_rows=len(counts),seed_consistency_rows=len(consistent),selection_effect_rows=len(selection),neighbor_intervention_rows=len(interventions),neighbor_latency_rows=len(latency),score_averaging_across_sites_or_seeds=False,improvement_threshold='strictly lower; no percentage cutoff',validation_year=2023))
    print(json.dumps(dict(summary_rows=len(counts),report=str(OUT/'SCREENING_RESULTS_20261025_CN.md'),interventions=len(interventions))))


if __name__=='__main__':main()
