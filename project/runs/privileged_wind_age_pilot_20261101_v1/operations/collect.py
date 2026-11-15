from pathlib import Path
import sys,json,csv,collections
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now,sha
import numpy as np
def collect():
    manifest=json.loads((RUN/'manifest.json').read_text());results=[]
    for spec in manifest:
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        d=json.loads((out/'metrics.json').read_text());assert d['spec']==spec and not d['smoke_only']
        results.append(d)
    groups=collections.defaultdict(list);rows=[]
    for d in results:
        s=d['spec'];groups[(s['age_shift'],s['variant'])].append(d)
        r=dict(**s,mae=d['metrics']['mae'],rmse=d['metrics']['rmse'],epochs=d['epochs'],seconds=d['seconds'],**d['diagnostics']['weather'])
        r.pop('lead_raw_wind_mse');r.pop('lead_corrected_wind_mse');rows.append(r)
    folder=RUN/'analysis';folder.mkdir(exist_ok=True)
    if rows:
        with (folder/'per_farm_seed.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        leads=[];scenes=[]
        for d in results:
            s=d['spec'];base={k:s[k] for k in ['id','site','seed','age_shift','variant']}
            for k,(a,b) in enumerate(zip(d['metrics']['lead_mae'],d['metrics']['lead_rmse'])):
                leads.append(dict(**base,lead_minutes=(k+1)*15,mae=a,rmse=b))
            for group,m in d['diagnostics']['current_power'].items():
                scenes.append(dict(**base,group=group,horizon='all',mae=m['mae'],rmse=m['rmse'],n=m['n']))
                for hz,v in m['horizons'].items():scenes.append(dict(**base,group=group,horizon=hz,**v))
        for name,table in [('per_lead.csv',leads),('current_power_groups.csv',scenes)]:
            with (folder/name).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    summary=[]
    for (age,arm),ds in sorted(groups.items()):
        summary.append(dict(age_shift=age,variant=arm,n=len(ds),complete=len(ds)==112,
            mae=float(np.mean([d['metrics']['mae'] for d in ds])),rmse=float(np.mean([d['metrics']['rmse'] for d in ds])),
            raw_wind_mse=float(np.mean([d['diagnostics']['weather']['raw_wind_mse'] for d in ds])),
            corrected_wind_mse=float(np.mean([d['diagnostics']['weather']['corrected_wind_mse'] for d in ds]))))
    comparisons=[];byid={(d['spec']['age_shift'],d['spec']['variant'],d['spec']['site'],d['spec']['seed']):d for d in results}
    for age in [0,12]:
        if any(len(groups[(age,a)])!=112 for a in ['E0','E1','E2','E3','E4','E5']):continue
        for arm in ['E0','E2','E3','E4','E5']:
            comp=dict(age_shift=age,candidate='E1',reference=arm)
            for metric in ['mae','rmse']:
                new=np.array([[byid[age,'E1',s,r]['metrics'][metric] for r in [42,43]] for s in range(56)]).mean(1)
                ref=np.array([[byid[age,arm,s,r]['metrics'][metric] for r in [42,43]] for s in range(56)]).mean(1)
                comp[metric+'_improvement_percent']=float((1-new.mean()/ref.mean())*100);comp[metric+'_improved_farms']=int((new<ref).sum())
            comparisons.append(comp)
    complete=len(results)==len(manifest)
    atomic_json(folder/'SUMMARY.json',dict(timestamp=now(),completed=len(results),planned=len(manifest),complete=complete,groups=summary,comparisons=comparisons))
    lines=['# 预报年龄与 ERA5 特权监督校正：2023 筛选',f'更新时间：{now()}。完成 {len(results)}/{len(manifest)}。',
        '2022训练，2023剔除11月及两年龄相关边界选点；56站×2种子。功率指标来自同一个按验证MAE选择的checkpoint。未读取2024。',
        'E0原模型；E1实时本站+区域功率驱动的ERA5监督风矢量校正；E2无ERA5辅助损失；E3校正模块仅天气；E4校正模块仅本站功率；E5校正模块异日功率。E3/E4/E5的主骨干仍保留原有真实功率输入。',
        '未完成组的均值仅供运行检查，不能与另一组直接判优。',
        '|年龄|组|已完成|功率MAE|功率RMSE|原始风速MSE|校正风速MSE|','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f'|{12+r["age_shift"]}–{24+r["age_shift"]}h|{r["variant"]}|{r["n"]}/112|{r["mae"]:.6f}|{r["rmse"]:.6f}|{r["raw_wind_mse"]:.6f}|{r["corrected_wind_mse"]:.6f}|')
    if comparisons:
        lines+=['','完整配对比较（正值为E1改善）：','|年龄增量|对照|MAE改善|RMSE改善|MAE/RMSE改善站数|','|---|---|---:|---:|---:|']
        for r in comparisons:lines.append(f'|+{r["age_shift"]}h|{r["reference"]}|{r["mae_improvement_percent"]:.3f}%|{r["rmse_improvement_percent"]:.3f}%|{r["mae_improved_farms"]}/{r["rmse_improved_farms"]}|')
    if complete:
        assert len({d['spec']['id'] for d in results})==len(manifest)
        assert len({d['n_train'] for d in results})==1 and len({d['n_val'] for d in results})==1
        for d in results:
            out=JOBS/d['spec']['id'];assert sha(out/'best.pt')==d['checkpoint_sha256']
            z=np.load(out/'daily.npz');n,ab,sq=z['power'].sum((0,1))
            assert abs(ab/n-d['metrics']['mae'])<1e-10 and abs(np.sqrt(sq/n)-d['metrics']['rmse'])<1e-10
        inference=bootstrap(byid);atomic_json(folder/'PAIRED_WEEK_BOOTSTRAP.json',inference)
        verdict=[]
        for age in [0,12]:
            base=next(x for x in comparisons if x['age_shift']==age and x['reference']=='E0')
            controls=[x for x in comparisons if x['age_shift']==age and x['reference'] in ['E2','E3','E5']]
            q=groups[(age,'E1')];b=groups[(age,'E0')]
            high=lambda ds:np.mean([d['diagnostics']['current_power']['high']['rmse'] for d in ds if 'high' in d['diagnostics']['current_power']])
            high_regression=float(high(q)/high(b)-1)
            weather_good=all(np.mean([d['diagnostics']['weather'][k] for d in q])<np.mean([d['diagnostics']['weather'][k] for d in b]) for k in ['corrected_wind_mse','corrected_wind_mse_4_6'])
            power_good=all(base[m+'_improvement_percent']>=.1 and base[m+'_improved_farms']>=29 for m in ['mae','rmse'])
            control_good=all(x[m+'_improvement_percent']>0 for x in controls for m in ['mae','rmse'])
            verdict.append(dict(age_shift=age,power_gate=power_good,control_gate=control_good,weather_gate=bool(weather_good),high_power_rmse_regression_percent=100*high_regression,
                passed=bool(power_good and control_good and weather_good and high_regression<=.001)))
        atomic_json(folder/'VERDICT.json',dict(timestamp=now(),results=verdict,scope='2023 screening, not untouched confirmatory test',next_stage_automatic=False))
        atomic_json(folder/'COMPLETENESS_AUDIT.json',dict(timestamp=now(),passed=True,completed=len(results),planned=len(manifest),
            duplicate_specs=len({d['spec']['id'] for d in results})!=len(results),test_read=False,all_metrics_same_checkpoint=True,all_n_train_equal=len({d['n_train'] for d in results})==1,all_n_val_equal=len({d['n_val'] for d in results})==1))
        lines+=['','验收与同步日历周配对区间已写入 VERDICT.json 和 PAIRED_WEEK_BOOTSTRAP.json。负结果也完整保留；不会自动启动2024或GFS实验。']
    (folder/'REPORT_CN.md').write_text('\n\n'.join(lines[:5])+'\n\n'+'\n'.join(lines[5:])+'\n')

def bootstrap(byid):
    # Same week multiplicities for every farm, seed, age and correction arm.
    entries={};week_ids=None
    for key,d in byid.items():
        z=np.load(JOBS/d['spec']['id']/'daily.npz');week=(z['day']-4)//7
        if week_ids is None:week_ids=np.unique(week)
        assert np.array_equal(np.unique(week),week_ids)
        p=z['power'];entries[key]=np.array([p[week==w].sum((0,1)) for w in week_ids])
    rng=np.random.default_rng(61101);counts=rng.multinomial(len(week_ids),np.full(len(week_ids),1/len(week_ids)),size=1000)
    values={}
    for age in [0,12]:
        for arm in ['E0','E1','E2','E3','E4','E5']:
            x=np.stack([entries[age,arm,s,r] for s in range(56) for r in [42,43]])
            n=np.einsum('bw,jw->bj',counts,x[:,:,0]);ab=np.einsum('bw,jw->bj',counts,x[:,:,1]);sq=np.einsum('bw,jw->bj',counts,x[:,:,2])
            values[age,arm,'mae']=(ab/n).mean(1);values[age,arm,'rmse']=np.sqrt(sq/n).mean(1)
    rows=[]
    for age in [0,12]:
        for arm in ['E0','E2','E3','E4','E5']:
            for m in ['mae','rmse']:
                gain=100*(1-values[age,'E1',m]/values[age,arm,m]);rows.append(dict(age_shift=age,reference=arm,metric=m,ci95=np.quantile(gain,[.025,.975]).tolist()))
    paired=[]
    for m in ['mae','rmse']:
        g=lambda age:100*(1-values[age,'E1',m]/values[age,'E0',m])
        paired.append(dict(metric=m,older_minus_current_gain_percentage_points_ci95=np.quantile(g(12)-g(0),[.025,.975]).tolist()))
    return dict(bootstrap_unit='calendar week synchronized across all farms/seeds/ages',replicates=1000,conditional_on_selected_checkpoints=True,comparisons=rows,paired_age_effect=paired)
if __name__=='__main__':collect()
