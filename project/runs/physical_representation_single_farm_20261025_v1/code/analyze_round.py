"""A-only attribution report, with actual farm/seed values preceding summaries."""
from settings import *
from physical_data import A1,FEATURES
import csv,itertools,collections,numpy as np

DEFAULT='TF_z0_lr0.001_dp0'
def write_csv(path,rows):
    rows=iter(rows);first=next(rows,None)
    if first is None:return
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(rows)
    os.replace(tmp,path)

def main():
    manifest=json.loads((RUN/'manifest.json').read_text());actual=[];index={};diagnostic_jobs=[]
    complete=[s for s in manifest if (JOBS/s['id']/'SUCCESS.json').exists()]
    # Baseline results exist even while their zero-training diagnosis is queued.
    sources=[(s,PARENT_RUN/'jobs'/s['source_job']) for s in manifest if s['task']=='diagnose_reference']
    sources += [(s,JOBS/s['id']) for s in complete if s['task']=='train']
    for spec,folder in sources:
        m=json.loads((folder/'metrics.json').read_text())
        for criterion,selection in m['selections'].items():
            for horizon,score in [('all',selection['metrics']),*selection['metrics']['horizons'].items()]:
                row=dict(configuration=spec['configuration'],backbone=spec['backbone'],variant=spec['variant'],site=spec['site'],seed=spec['seed'],selection=criterion,horizon=horizon,epoch=selection['epoch'],source=str(folder/'metrics.json'),trainable_parameters=m.get('trainable_parameters',''),**{k:score[k] for k in ['n','mae','rmse','bias']})
                key=tuple(row[k] for k in ['configuration','variant','site','seed','selection','horizon'])
                assert key not in index;index[key]=row;actual.append(row)
    write_csv(RUN/'analysis/all_actual_per_farm_seed.csv',actual)
    edges=set()
    def edge(kind,cand,ref):
        if cand!=ref:edges.add((kind,*cand,*ref))
    labels={(r['configuration'],r['variant']) for r in actual}
    for c,v in labels:
        default='W0' if c=='W0' else DEFAULT
        edge('input_vs_default_BASE',(c,v),(default,'BASE'))
        if c==default:edge('input_vs_DENS',(c,v),(default,'DENS'))
        if v.startswith('SEL'):edge('combination_vs_SEL',(c,v),(c,'SEL'))
        if v=='DENS':edge('within_configuration_DENS',(c,v),(c,'BASE'))
        if c=='W0':edge('same_input_backbone',(c,v),(DEFAULT,v))
    for c in ['W0',DEFAULT]:
        for v,ref in [('SPD3','SPD'),('SPD_LV','SPD'),('SPD3_LV','SPD3'),('DENS_UNCOUPLED','DENS_NORHO'),('DENS','DENS_UNCOUPLED'),('DENS','DENS_NORHO'),('SEL_REF_DELTA','SEL_REF')]:edge('representation_attribution',(c,v),(c,ref))
    factors=list(itertools.product([False,True],[.001,.0002],[0.,.05]))
    config=lambda t:f'TF_z{int(t[0])}_lr{t[1]:g}_dp{t[2]:g}'
    for a,b in itertools.combinations(factors,2):
        changed=[i for i in range(3) if a[i]!=b[i]]
        if len(changed)==1:
            for v in ['BASE','DENS']:edge('factor_'+['zero_init','learning_rate','dropout'][changed[0]],(config(b),v),(config(a),v))
    paired=[];grouped=collections.defaultdict(list)
    for kind,cc,cv,rc,rv in sorted(edges):
        for site,seed,selection,horizon in itertools.product(range(56),SEEDS,['mae','rmse'],['all','0_1h','1_3h','3_6h']):
            a=index.get((cc,cv,site,seed,selection,horizon));b=index.get((rc,rv,site,seed,selection,horizon))
            if a is None or b is None:continue
            assert a['n']==b['n']
            row=dict(contrast=kind,candidate_configuration=cc,candidate_variant=cv,reference_configuration=rc,reference_variant=rv,site=site,seed=seed,selection=selection,horizon=horizon)
            for metric in ['mae','rmse','bias']:
                row.update({metric+'_candidate':a[metric],metric+'_reference':b[metric],metric+'_difference':a[metric]-b[metric]})
            paired.append(row);grouped[(kind,cc,cv,rc,rv,selection,horizon)].append(row)
    write_csv(RUN/'analysis/paired_per_farm_seed.csv',paired)
    summaries=[]
    for key,rows in sorted(grouped.items()):
        farms=collections.defaultdict(list)
        for r in rows:farms[r['site']].append(r)
        # Do not silently give partially completed farms different seed weights.
        sites=[rs for rs in farms.values() if {r['seed'] for r in rs}==set(SEEDS)]
        if not sites:continue
        for metric in ['mae','rmse']:
            a=np.array([np.mean([r[metric+'_candidate'] for r in rs]) for rs in sites]);b=np.array([np.mean([r[metric+'_reference'] for r in rs]) for rs in sites])
            rng=np.random.default_rng(20261025);ix=rng.integers(0,len(a),(2000,len(a)))
            effects=100*(a[ix].mean(1)-b[ix].mean(1))/b[ix].mean(1)
            summaries.append(dict(zip(['contrast','candidate_configuration','candidate_variant','reference_configuration','reference_variant','selection','horizon'],key))|dict(metric=metric,farms=len(sites),pairs=2*len(sites),candidate_equal_farm=float(a.mean()),reference_equal_farm=float(b.mean()),difference=float((a-b).mean()),effect_percent=float(100*(a.mean()-b.mean())/b.mean()),ci_low=float(np.quantile(effects,.025)),ci_high=float(np.quantile(effects,.975)),improved_farms=int((a<b).sum()),improved_seed_pairs=sum(r[metric+'_difference']<0 for rs in sites for r in rs),complete_56farms=len(sites)==56))
    write_csv(RUN/'analysis/supplementary_effect_sizes.csv',summaries)
    def diagnostics():
        for spec in complete:
            path=JOBS/spec['id']/'diagnostics.json'
            if not path.exists():continue
            for row in json.loads(path.read_text())['rows']:
                yield dict(configuration=spec['configuration'],backbone=spec['backbone'],variant=spec['variant'],site=spec['site'],seed=spec['seed'],selection='mae',**row)
    write_csv(RUN/'analysis/diagnostics_per_farm_seed.csv',diagnostics())
    # Only complete cells enter selection. A recommendation is not a B freeze.
    ranking=[]
    for c,v in labels:
        rows=[index.get((c,v,s,seed,'mae','all')) for s,seed in itertools.product(range(56),SEEDS)]
        if not all(rows):continue
        groups=FEATURES[v];active=sum({'density':3,'dynamics':7,'stability':5,'state':7,'reference':16,'delta':14,'calendar':4,'disagreement':15,'spd':1,'spd3':1,'rho':1,'dens_norho':2}.get(g,0 if g.startswith('zero_') else 3) for g in groups)
        ranking.append(dict(configuration=c,variant=v,mae=float(np.mean([r['mae'] for r in rows])),rmse=float(np.mean([r['rmse'] for r in rows])),active_inputs=active))
    ranking.sort(key=lambda r:(r['mae'],r['active_inputs'],r['variant']))
    all_done=len(complete)==len(manifest)
    eligible=[r for r in ranking if r['configuration']=='W0']
    tf=[r for r in ranking if r['configuration'].startswith('TF_') and r['variant']=='DENS']
    recommendation=dict(timestamp=now(),a_complete=all_done,test_year_accessed=False,selection_frozen=False,eligible_complete_cells=ranking,recommended_main=eligible[0] if eligible and all_done else None,recommended_tf_configuration=min(tf,key=lambda r:r['mae']) if tf and all_done else None,next_action='Review paired attribution and diagnostics, explicitly write FROZEN_SELECTION before B. Do not run B from this file.')
    atomic_json(RUN/'analysis/SELECTION_RECOMMENDATION.json',recommendation)
    lines=['# 2023 归因筛选结果','',f'更新：{now()}；完成 {len(complete)}/{len(manifest)} 项（含零训练诊断）。2024 未访问。','',
        '原值见 [逐风场逐seed全表](all_actual_per_farm_seed.csv)、[配对差](paired_per_farm_seed.csv)、[分场景原值](diagnostics_per_farm_seed.csv)。效应量见 [补充汇总](supplementary_effect_sizes.csv)。模型结构及输入定义见 [执行说明](../../../NEXT_ROUND_REVIEW_AND_EXECUTION_20261025_CN.md)。','',
        '汇总仅含两seed均齐的风场。百分比为站内seed平均后、风场等权误差的相对差，负值改善；站点bootstrap区间仅为探索性描述，未校正空间相关或多重选择。区间重叠不是等效证据。小幅改善保留，不设百分比门槛。','',
        '|配置|输入|2023 MAE（补充均值）|RMSE（补充均值）|','|---|---|---:|---:|']
    for r in ranking:lines.append(f"|{r['configuration']}|{r['variant']}|{r['mae']:.8f}|{r['rmse']:.8f}|")
    lines+=['','自动建议见 SELECTION_RECOMMENDATION.json；此文件不代表已冻结最终方案，也不启动2024。']
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return dict(complete=len(complete),actual_rows=len(actual),paired_rows=len(paired),summary_rows=len(summaries))

if __name__=='__main__':print(json.dumps(main()))
