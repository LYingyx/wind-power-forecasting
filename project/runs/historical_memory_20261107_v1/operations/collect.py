from pathlib import Path
import sys,json,csv
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np

def compare(a,b):
    a=np.asarray(a).mean(0);b=np.asarray(b).mean(0)
    return dict(baseline=float(b.mean()),corrected=float(a.mean()),change_pct=float(100*(a.mean()/b.mean()-1)),wins=int((a<b).sum()),stations=len(a))

def collect():
    specs=json.loads((RUN/'manifest.json').read_text());ready=[s for s in specs if (JOBS/s['id']/'SUCCESS.json').exists()]
    result=dict(timestamp=now(),planned=len(specs),completed=len(ready),stage='running',Z1={},RM={},next_LM_training='held_for_user_review')
    z={s['id']:json.loads((JOBS/s['id']/'Z1.json').read_text()) for s in ready if s['stage']=='Z1'}
    gate=[];rows=[]
    for family in ['W0','TF']:
        rs=[s for s in ready if s['stage']=='Z1' and s['family']==family]
        if len(rs)!=275:
            result['Z1'][family]=dict(completed=len(rs),required=275);continue
        index={(s['seed'],s['station']):z[s['id']] for s in rs}
        for mode in ['P0','E1']:
            key=f'{family}_{mode}';result['Z1'][key]={}
            for variant in ['bin30','bin7','bin14','bin30_exclude24h','uniform30']:
                stat={}
                for horizon in ['all','0_1h','1_3h','3_6h']:
                    stat[horizon]={}
                    for metric in ['mae','rmse']:
                        def arr(v):
                            return [[(index[seed,s][mode][v] if horizon=='all' else index[seed,s][mode][v]['horizons'][horizon])[metric] for s in range(55)] for seed in range(42,47)]
                        c=compare(arr(variant),arr('raw'));stat[horizon][metric]=c
                        rows.append([key,variant,horizon,metric,c['baseline'],c['corrected'],c['change_pct'],c['wins'],55])
                def high(v):return [[index[seed,s][mode][v]['high_power_rmse'] for s in range(55)] for seed in range(42,47)]
                h0,h1=np.array(high('raw'),float),np.array(high(variant),float);mask=np.isfinite(h0).all(0)&np.isfinite(h1).all(0)
                stat['high_power_rmse']=compare(h1[:,mask],h0[:,mask]) if mask.any() else None
                stat['per_seed']={str(seed):{m:compare([[index[seed,s][mode][variant][m] for s in range(55)]],[[index[seed,s][mode]['raw'][m] for s in range(55)]]) for m in ['mae','rmse']} for seed in range(42,47)}
                stat['passes_Z1_metrics']=all(stat['all'][m]['change_pct']<=-.1 and stat['all'][m]['wins']>27 for m in ['mae','rmse'])
                stat['high_power_guard']=stat['high_power_rmse'] is not None and stat['high_power_rmse']['change_pct']<=.1
                result['Z1'][key][variant]=stat
            if mode=='E1':gate.append(result['Z1'][key]['bin30']['passes_Z1_metrics'])
    result['Z1_gate']=bool(all(gate)) if len(gate)==2 else None
    for family in ['W0','TF']:
        rs=sorted([s for s in ready if s['stage']=='RM' and s['family']==family],key=lambda s:s['seed'])
        if len(rs)!=5:result['RM'][family]=dict(completed=len(rs),required=5);continue
        a=[json.loads((JOBS/s['id']/'metrics.json').read_text())['metrics'] for s in rs]
        b=[json.loads((RUN/f"source_metrics/{family}_r{s['seed']}.json").read_text())['metrics'] for s in rs]
        d={m:compare([x['per_farm_'+m] for x in a],[x['per_farm_'+m] for x in b]) for m in ['mae','rmse']}
        d['high_power_rmse']=compare([x['high_power_per_farm_rmse'] for x in a],[x['high_power_per_farm_rmse'] for x in b])
        d['horizons']={}
        for h,lo,hi in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
            d['horizons'][h]={}
            for m in ['mae','rmse']:
                def arr(allmetrics):
                    v=np.array([x['lead_'+m] for x in allmetrics])[:,:,lo:hi]
                    return v.mean(-1) if m=='mae' else np.sqrt((v**2).mean(-1))
                d['horizons'][h][m]=compare(arr(a),arr(b))
        d['per_seed']={str(s['seed']):{m:compare([aa['per_farm_'+m]],[bb['per_farm_'+m]]) for m in ['mae','rmse']} for s,aa,bb in zip(rs,a,b)}
        d['pass']=bool(all(d[m]['change_pct']<=-.1 and d[m]['wins']>27 for m in ['mae','rmse']) and d['high_power_rmse']['change_pct']<=.1)
        result['RM'][family]=d
    if len(ready)==len(specs):result['stage']='GPU_experiments_complete'
    atomic_json(RUN/'analysis/RESULTS.json',result)
    with (RUN/'analysis/Z1_horizons.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['model','variant','horizon','metric','raw','corrected','change_pct','wins','stations']);w.writerows(rows)
    lines=['# Z1 与 RM 结果','',f"更新：{result['timestamp']}。GPU 任务 {len(ready)}/{len(specs)} 完成。",'',
        '固定 2022 训练、2023 选择；2023 不含 11 月。2024 未评价。负值为改善。先平均各 seed 指标，再对站点宏平均。', '',
        '| Z1 模型 | 时效 | MAE 变化 | RMSE 变化 | MAE / RMSE 胜场 |','|---|---|---:|---:|---:|']
    for key,v in result['Z1'].items():
        if 'bin30' not in v:continue
        for h in ['all','0_1h','1_3h','3_6h']:
            a=v['bin30'][h];lines.append(f"| {key} | {h} | {a['mae']['change_pct']:+.3f}% | {a['rmse']['change_pct']:+.3f}% | {a['mae']['wins']} / {a['rmse']['wins']} |")
    lines+=['',f"Z1 双骨干 E1 门槛：{result['Z1_gate']}（None 表示未齐）。",'', '| RM 相对各自 R3 | MAE 变化 | RMSE 变化 | 胜场 | 高功率 RMSE |','|---|---:|---:|---:|---:|']
    for family,v in result['RM'].items():
        if 'mae' not in v:continue
        lines.append(f"| {family} | {v['mae']['change_pct']:+.3f}% | {v['rmse']['change_pct']:+.3f}% | {v['mae']['wins']} / {v['rmse']['wins']} | {v['high_power_rmse']['change_pct']:+.3f}% |")
    lines+=['','RM 同时改变区域输入位置与 W0 历史站点向量；若改善，只能归因于该组合，不能单独归因于其中一项。','',
        'LM 仅准备数据，后续训练等待本报告审阅。7/14 天、排除最近 24 h、高功率与逐 seed 结果见 RESULTS.json。']
    (RUN/'analysis/RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
