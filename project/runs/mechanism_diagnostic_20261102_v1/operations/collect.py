from pathlib import Path
import sys,json,csv
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np

def csvfile(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bootstrap(groups):
    days=None;entries={}
    for variant,ds in groups.items():
        entries[variant]=[]
        for d in sorted(ds,key=lambda x:(x['spec']['site'],x['spec']['seed'])):
            sp=d['spec'];path=RUN/'assets/baseline_metrics'/f's{sp["site"]:02d}_r{sp["seed"]}_daily.npz' if variant=='E0' else JOBS/sp['id']/'daily.npz'
            z=np.load(path);day=z['day'];week=(day-4)//7;uw=np.unique(week)
            if days is None:days=uw
            assert np.array_equal(days,uw)
            entries[variant].append(np.array([z['power'][week==w].sum((0,1)) for w in uw]))
    rng=np.random.default_rng(61102);counts=rng.multinomial(len(days),np.ones(len(days))/len(days),size=1000);values={}
    for v,entries_v in entries.items():
        x=np.array(entries_v);n=np.einsum('bw,jw->bj',counts,x[:,:,0]);ab=np.einsum('bw,jw->bj',counts,x[:,:,1]);sq=np.einsum('bw,jw->bj',counts,x[:,:,2])
        values[v]={'mae':(ab/n).mean(1),'rmse':np.sqrt(sq/n).mean(1)}
    return [dict(candidate='I_MUL',reference=v,metric=m,regression_percent_ci95=np.quantile(100*(values['I_MUL'][m]/values[v][m]-1),[.025,.975]).tolist()) for v in groups if v!='I_MUL' for m in ['mae','rmse']]

def collect():
    manifest=json.loads((RUN/'manifest.json').read_text());done=[]
    for spec in manifest:
        p=JOBS/spec['id']
        if not (p/'SUCCESS.json').exists():continue
        d=json.loads((p/'metrics.json').read_text());assert d['spec']==spec and not d['smoke_only'];done.append(d)
    out=RUN/'analysis';out.mkdir(exist_ok=True)
    correction=[d for d in done if d['spec']['role']=='correction'];interaction=[d for d in done if d['spec']['role']=='interaction']
    corr_rows=[];corr_summary=[]
    for d in correction:
        dg=d['diagnostics'];models={'RAW':dg['raw_val'],'PURE':d['metrics'],'PURE_SPEED_SELECTED':dg['speed_selected_val'],'E1':dg.get('E1_val',{}),'E3':dg.get('E3_val',{}),**dg['linear']}
        for model,m in models.items():
            corr_rows.append(dict(site=d['spec']['site'],seed=d['spec']['seed'],model=model,**{k:m.get(k) for k in ['node_normalized_uv_mse','u_mse','v_mse','uv_mse','speed_mse','speed_rmse','uv_mse_4_6','speed_mse_4_6']}))
    if correction:
        for model in dict.fromkeys(x['model'] for x in corr_rows):
            rows=[x for x in corr_rows if x['model']==model]
            corr_summary.append(dict(model=model,n=len(rows),**{k:float(np.mean([x[k] for x in rows if x[k] is not None])) if any(x[k] is not None for x in rows) else None for k in ['node_normalized_uv_mse','u_mse','v_mse','uv_mse','speed_mse','speed_rmse','uv_mse_4_6','speed_mse_4_6']}))
    csvfile(out/'weather_per_farm_seed.csv',corr_rows)
    generalization=[];weather_comparisons=[]
    for split in ['train','val']:
        for model in ['RAW','PURE','E1','E3']:
            field={'RAW':'raw_'+split,'PURE':'pure_train' if split=='train' else None,'E1':'E1_'+split,'E3':'E3_'+split}[model]
            ms=[d['metrics'] if field is None else d['diagnostics'][field] for d in correction if field is None or field in d['diagnostics']]
            if ms:generalization.append(dict(split=split,model=model,n=len(ms),**{k:float(np.mean([x[k] for x in ms])) for k in ['node_normalized_uv_mse','uv_mse','speed_mse']}))
    if len(correction)==112:
        for ref in ['RAW','E1','E3','MOS_UV','RIDGE_UV_CONTEXT','RIDGE_UV_OBS','MOS_SPEED']:
            row=dict(candidate='PURE',reference=ref)
            for metric in ['uv_mse','speed_mse','speed_mse_4_6']:
                def values(model):
                    return np.array([[next(x[metric] for x in corr_rows if x['site']==site and x['seed']==seed and x['model']==model) for seed in [42,43]] for site in range(56)],dtype=float).mean(1)
                x,y=values('PURE'),values(ref)
                if np.isfinite(y).all():
                    row[metric+'_change_percent']=float(100*(x.mean()/y.mean()-1));row[metric+'_improved_farms']=int((x<y).sum())
            weather_comparisons.append(row)
    base=[json.loads(p.read_text()) for p in sorted((ASSETS/'baseline_metrics').glob('*.json'))]
    groups={'E0':base,**{v:[d for d in interaction if d['spec']['variant']==v] for v in ['I_MUL','I_ADD','I_XATTN','I_LOCAL','I_WRONG']}}
    power_summary=[];power_rows=[];leads=[];scenes=[];comparisons=[]
    for variant,ds in groups.items():
        if not ds:continue
        power_summary.append(dict(variant=variant,n=len(ds),mae=float(np.mean([d['metrics']['mae'] for d in ds])),rmse=float(np.mean([d['metrics']['rmse'] for d in ds]))))
        for d in ds:
            sp=d['spec'];key=dict(site=sp['site'],seed=sp['seed'],variant=variant)
            power_rows.append(dict(**key,mae=d['metrics']['mae'],rmse=d['metrics']['rmse']))
            for i,(mae,rmse) in enumerate(zip(d['metrics']['lead_mae'],d['metrics']['lead_rmse'])):leads.append(dict(**key,lead_minutes=15*(i+1),mae=mae,rmse=rmse))
            for g,v in d['diagnostics']['current_power'].items():scenes.append(dict(**key,group=g,mae=v['mae'],rmse=v['rmse'],n=v['n']))
    for v in ['E0','I_ADD','I_XATTN','I_LOCAL','I_WRONG']:
        if len(groups['I_MUL'])!=112 or len(groups[v])!=112:continue
        row=dict(candidate='I_MUL',reference=v)
        for m in ['mae','rmse']:
            avg=lambda arm:np.array([[next(d['metrics'][m] for d in groups[arm] if d['spec']['site']==s and d['spec']['seed']==seed) for seed in [42,43]] for s in range(56)]).mean(1)
            cand,ref=avg('I_MUL'),avg(v);row[m+'_change_percent']=float(100*(cand.mean()/ref.mean()-1));row[m+'_improved_farms']=int((cand<ref).sum())
        comparisons.append(row)
    for name,rows in [('power_per_farm_seed.csv',power_rows),('power_per_lead.csv',leads),('power_current_groups.csv',scenes)]:csvfile(out/name,rows)
    verdict=None
    if len(interaction)==560:
        c=next(x for x in comparisons if x['reference']=='E0')
        high=lambda variant:np.mean([d['diagnostics']['current_power']['high']['rmse'] for d in groups[variant] if 'high' in d['diagnostics']['current_power']])
        high_change=float(100*(high('I_MUL')/high('E0')-1))
        pg=all(c[m+'_change_percent']<=-.1 and c[m+'_improved_farms']>=29 for m in ['mae','rmse'])
        cg=all(x[m+'_change_percent']<0 for x in comparisons if x['reference']!='E0' for m in ['mae','rmse'])
        verdict=dict(power_gate=pg,control_gate=cg,high_power_rmse_change_percent=high_change,passed=bool(pg and cg and high_change<=.1))
        atomic_json(out/'PAIRED_WEEK_BOOTSTRAP.json',dict(conditional_on_2023_selected_checkpoints=True,replicates=1000,comparisons=bootstrap(groups)))
    summary=dict(timestamp=now(),completed=len(done),planned=len(manifest),correction_complete=len(correction)==112,interaction_complete=len(interaction)==560,weather=corr_summary,weather_comparisons=weather_comparisons,weather_train_val=generalization,power=power_summary,comparisons=comparisons,interaction_verdict=verdict)
    atomic_json(out/'SUMMARY.json',summary)
    lines=['# 校正排查与区域条件响应探索',f'更新时间：{now()}；已完成 {len(done)}/{len(manifest)}。2022 训练，2023 去除 11 月及共同边界筛选；不使用 2024。',
       '未满 112 项的组只供进度检查，不能用不同站点覆盖的均值判优。百分比正值表示变差。','', '## 纯监督天气校正','',
       'PURE 与旧 E1 使用相同校正器、初始化和输入，仅移除功率骨干与功率损失；按2023标准化u/v MSE选择checkpoint。另列按风速选点的诊断结果，不能混作同一checkpoint。MOS重新在本轮共同15min样本上拟合，不能直接套用旧整点诊断的0.648/0.619。','',
       '|模型|数量|u/v MSE（物理量）|风速MSE|逐站风速RMSE均值|4–6h风速MSE|','|---|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    for x in corr_summary:lines.append(f'|{x["model"]}|{x["n"]}|{fmt(x["uv_mse"])}|{fmt(x["speed_mse"])}|{fmt(x["speed_rmse"])}|{fmt(x["speed_mse_4_6"])}|')
    lines+=['','|年份用途|模型|标准化节点u/v MSE|物理u/v MSE|风速MSE|','|---|---|---:|---:|---:|']
    for x in generalization:lines.append(f'|{x["split"]}|{x["model"]}|{x["node_normalized_uv_mse"]:.6f}|{x["uv_mse"]:.6f}|{x["speed_mse"]:.6f}|')
    lines+=['','|PURE相对|u/v MSE变化|风速MSE变化|4–6h风速MSE变化|','|---|---:|---:|---:|']
    pct=lambda x:'—' if x is None else f'{x:+.3f}%'
    for x in weather_comparisons:lines.append(f'|{x["reference"]}|{pct(x.get("uv_mse_change_percent"))}|{pct(x.get("speed_mse_change_percent"))}|{pct(x.get("speed_mse_4_6_change_percent"))}|')
    lines+=['','## 显式区域条件响应','',
       'I_MUL：低秩的历史×区域×天气乘法通路，修改进入功率解码器的天气响应表示；不校正气象量。I_ADD：同参数加法；I_XATTN：同参数、天气query对本站/区域状态token做softmax交叉注意力；I_LOCAL：新增通路区域槽置零；I_WRONG：新增通路区域状态换成2022同月同钟点异日。五组均保留原E0的全部真实输入与交叉注意力骨干，新增参数均为4512；不宣称优于所有交叉注意力架构。','',
       '|模型|数量|功率MAE|功率RMSE|','|---|---:|---:|---:|']
    for x in power_summary:lines.append(f'|{x["variant"]}|{x["n"]}|{x["mae"]:.6f}|{x["rmse"]:.6f}|')
    lines+=['','|I_MUL相对|MAE变化|RMSE变化|MAE/RMSE改善站数|','|---|---:|---:|---:|']
    for x in comparisons:lines.append(f'|{x["reference"]}|{x["mae_change_percent"]:+.3f}%|{x["rmse_change_percent"]:+.3f}%|{x["mae_improved_farms"]}/56；{x["rmse_improved_farms"]}/56|')
    if verdict:lines+=['',f'预设筛选判定：{json.dumps(verdict,ensure_ascii=False)}。筛选通过也需要跨骨干和真正冻结后的验证，才能上升为机制贡献。']
    (out/'REPORT_CN.md').write_text('\n\n'.join(lines[:3])+'\n'+'\n'.join(lines[3:])+'\n')
    if len(done)==len(manifest):
        for d in done:
            assert d['n_train']==31190 and d['n_val']==29647 and d['evaluation_year']==2023
            assert sha(JOBS/d['spec']['id']/'best.pt')==d['checkpoint_sha256']
        atomic_json(out/'COMPLETENESS_AUDIT.json',dict(timestamp=now(),passed=True,completed=len(done),planned=len(manifest),test_read=False,all_checkpoint_hashes_verified=True))
    print(json.dumps(dict(completed=len(done),correction=len(correction),interaction=len(interaction))),flush=True)
if __name__=='__main__':collect()
