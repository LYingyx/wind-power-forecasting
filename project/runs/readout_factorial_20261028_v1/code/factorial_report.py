"""Paired station/seed comparisons; final selection only after all2800 tasks."""
from settings import *
from collect_physics import collect,csv_write
import numpy as np

def ledger():
    data={}
    for s in json.loads((RUN/'manifest.json').read_text()):
        p=JOBS/s['id']
        if not (p/'SUCCESS.json').exists():continue
        m=json.loads((p/'metrics.json').read_text())
        assert m['validation_mask_sha256']==sha(ASSETS/'val_keep.npy') and m['evaluation_year']==2023
        row=m['selections']['mae']['metrics'].copy()
        d=json.loads((p/'diagnostics.json').read_text())
        high=next((x['rmse'] for x in d['rows'] if x['group']=='current_power' and x['bin']=='high' and x['horizon']=='all'),None)
        row['high_rmse']=high
        data[s['configuration'],s['site'],s['seed']]=row
    return data

def comparison(data,base,arm):
    sites=[i for i in range(56) if all((c,i,s) in data for c in [base,arm] for s in SEEDS)]
    result=dict(base=base,arm=arm,paired_farms=len(sites),seeds_per_farm=10,passed=False)
    if len(sites)!=56:return result
    rng=np.random.default_rng(281028);boot=rng.integers(0,56,(4000,56))
    for metric in ['mae','rmse']:
        a=np.array([[data[base,i,s][metric] for s in SEEDS] for i in sites])
        b=np.array([[data[arm,i,s][metric] for s in SEEDS] for i in sites])
        am=a.mean(1);bm=b.mean(1)
        rel=float(bm.mean()/am.mean()-1)
        ci=np.quantile(bm[boot].mean(1)/am[boot].mean(1)-1,[.025,.975]).tolist()
        result[metric]=dict(base=float(am.mean()),arm=float(bm.mean()),relative_change=rel,farm_wins=int((bm<am).sum()),paired_farm_bootstrap95=ci,per_seed_macro_relative_change=(b.mean(0)/a.mean(0)-1).tolist())
    def macro_high(c):
        values=[data[c,i,s]['high_rmse'] for i in sites for s in SEEDS]
        if any(x is None for x in values):return None
        return float(np.mean(values))
    ha,hb=macro_high(base),macro_high(arm)
    result['high_power_rmse_relative_change']=hb/ha-1 if ha is not None and hb is not None and ha>0 else None
    result['short_0_1h_relative_change']={k:float(np.mean([data[arm,i,s]['horizons']['0_1h'][k] for i in sites for s in SEEDS])/np.mean([data[base,i,s]['horizons']['0_1h'][k] for i in sites for s in SEEDS])-1) for k in ['mae','rmse']}
    result['passed']=bool(all(result[k]['relative_change']<=-.001 and result[k]['farm_wins']>=29 for k in ['mae','rmse']) and result['high_power_rmse_relative_change'] is not None and result['high_power_rmse_relative_change']<=.005 and (base!='W0_H32' or max(result['short_0_1h_relative_change'].values())<=.001))
    return result

def run():
    state=collect();data=ledger();protocol=require_protocol();rules=protocol['selection']
    comparisons=[comparison(data,b,c) for c in rules['candidates'] for b in rules['references']]
    comparisons.append(comparison(data,'TF_R3_RAW','W0_H32'))
    full=state['completed']==state['planned']==2800 and not state['failed']
    eligible=[c for c in rules['candidates'] if all(next(x for x in comparisons if x['arm']==c and x['base']==b)['passed'] for b in rules['references'])] if full else []
    selected=min(eligible,key=lambda c:np.mean([data[c,i,s]['mae'] for i in range(56) for s in SEEDS])) if eligible else None
    effects=[]
    if full:
        terms={'current_power_at32':[(1,'W0_O_H32'),(-1,'W0_H32')],'current_power_at64':[(1,'W0_O_H64'),(-1,'W0_H64')],'width_without_current':[(1,'W0_H64'),(-1,'W0_H32')],'width_with_current':[(1,'W0_O_H64'),(-1,'W0_O_H32')],'interaction':[(1,'W0_O_H64'),(-1,'W0_H64'),(-1,'W0_O_H32'),(1,'W0_H32')]}
        rng=np.random.default_rng(281028);boot=rng.integers(0,56,(4000,56))
        for name,ts in terms.items():
            for metric in ['mae','rmse']:
                v=np.array([np.mean([sum(coef*data[c,i,s][metric] for coef,c in ts) for s in SEEDS]) for i in range(56)])
                effects.append(dict(contrast=name,metric=metric,mean_absolute_change=float(v.mean()),paired_farm_bootstrap95=np.quantile(v[boot].mean(1),[.025,.975]).tolist(),farms_negative=int((v<0).sum()),per_farm_change=v.tolist()))
    result=dict(timestamp=now(),complete=full,completed=state['completed'],planned=2800,comparisons=comparisons,factorial_effects=effects,selected=selected,eligible=eligible,uncertainty_note='Farm bootstrap is descriptive; spatial/weather dependence and interannual uncertainty are not resolved by10 training seeds.',test_year_evaluated=False)
    atomic_json(RUN/'analysis/FACTORIAL_RESULTS.json',result)
    lines=['# 输出头2×2：2023去11月验证',f'更新：{now()}。完成{state["completed"]}/2800。所有组2022训练；先站内平均10种子，再56站等权平均。','', '|参照|变体|完整配对风场|MAE变化|RMSE变化|通过预设门槛|','|---|---|---:|---:|---:|---|']
    for x in comparisons:
        lines.append(f'|{x["base"]}|{x["arm"]}|{x["paired_farms"]}|'+(f'{x["mae"]["relative_change"]*100:+.3f}%|{x["rmse"]["relative_change"]*100:+.3f}%|{x["passed"]}|' if 'mae' in x else '待齐|待齐|待齐|'))
    lines+=['',f'选择结果：{selected or ("无新变体同时通过两个参照的门槛" if full else "尚未齐全，不做选择")}。','未通过优势检验不等于模型等效，也不自动改用TF。当前轮不读取2024进行评价；后续复评须先冻结，且应注明其探索性质。','逐风场、种子、时效及场景数据见per_farm_seed_year.csv、stations/、leads_2023.csv及scenarios_2023.csv。']
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':run()
