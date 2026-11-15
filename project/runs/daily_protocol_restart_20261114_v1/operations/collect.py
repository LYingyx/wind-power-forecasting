from pathlib import Path
import sys,json,csv,os
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np

def complete(spec):
    out=JOBS/spec['id']
    if not (out/'SUCCESS.json').exists():return None
    d=json.loads((out/'metrics.json').read_text());assert d['requested_spec']==spec and not d['smoke_only'];return d
def collect():
    specs=json.loads((RUN/'manifest.json').read_text());done={s['id']:d for s in specs if (d:=complete(s)) is not None}
    # Tuning: one configuration per family, equal-site MAE averaged across both seeds.
    for group in ['TF_BASE','TF_R3','W0_R3','TIDE_R3','TFT_R3']:
        jobs=[s for s in specs if s['stage']=='tune' and s['group']==group];out=ASSETS/'selected'/f'{group}.json'
        if not out.exists() and all(s['id'] in done for s in jobs):
            table=[]
            for ci in range(8):
                pair=[s for s in jobs if f'_c{ci}_' in s['id']];ms=[done[s['id']]['metrics'] for s in pair]
                table.append(dict(index=ci,config=pair[0]['config'],mae=float(np.mean([m['mae'] for m in ms])),rmse=float(np.mean([m['rmse'] for m in ms])),jobs=[s['id'] for s in pair]))
            best=min(table,key=lambda x:(x['mae'],x['index']));atomic_json(out,dict(timestamp=now(),group=group,config=best['config'],selected_by='2023 macroMAE averaged tune seeds40,41',selected_index=best['index'],table=table))
    # Assemble genuinely out-of-fold2022 references; final full model predicts2023.
    for seed in SEEDS:
        ids=[f'B0_r{seed}_f{f}' for f in [-1,0,1,2,3]];folder=DATA/f'reference_r{seed}'
        if not (folder/'READY.json').exists() and all(j in done for j in ids):
            folder.mkdir(exist_ok=True);shape=np.load(DATA/'train/h.npy',mmap_mode='r').shape
            path=folder/'train.partial.npy';a=np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=shape);covered=np.zeros(shape[0],bool)
            sources={}
            for f in range(4):
                jid=f'B0_r{seed}_f{f}';p=JOBS/jid/'reference_prediction.npy';assert sha(p)==done[jid]['reference_sha256'];held=np.load(ASSETS/f'fold{f}.npz')['heldout'];v=np.load(p,mmap_mode='r');assert v.shape==(len(held),*shape[1:]);a[held]=v;covered[held]=True;sources[jid]=done[jid]['checkpoint_sha256']
            assert covered.all() and np.isfinite(a).all();a.flush();del a;os.replace(path,folder/'train.npy')
            jid=ids[0];p=JOBS/jid/'reference_prediction.npy';assert sha(p)==done[jid]['reference_sha256'];v=np.load(p,mmap_mode='r');assert v.shape==np.load(DATA/'val/h.npy',mmap_mode='r').shape
            np.save(folder/'val.npy',v);sources[jid]=done[jid]['checkpoint_sha256']
            atomic_json(folder/'READY.json',dict(timestamp=now(),passed=True,seed=seed,sources=sources,train_mode='purged-quarter-OOF',val_mode='full2022_model',files={x:sha(folder/x) for x in ['train.npy','val.npy']}))
    grouped={}
    for s in specs:
        if s['id'] in done and s['stage'] in ['baseline','mechanism','regional_memory']:
            grouped.setdefault(s['group'],[]).append(done[s['id']])
    rows=[];byfarm=[]
    cohort=json.loads((ASSETS/'COHORT.json').read_text());nf=cohort['count']
    for group,ds in grouped.items():
        m=np.array([d['metrics']['per_farm_mae'] for d in ds]);r=np.array([d['metrics']['per_farm_rmse'] for d in ds])
        rows.append(dict(group=group,seeds=len(ds),mae=float(m.mean()),rmse=float(r.mean()),complete_5seeds=len(ds)==5))
        for s in range(nf):byfarm.append(dict(group=group,station_id=cohort['station_ids'][s],name=cohort['names'][s],seeds=len(ds),mae=float(m[:,s].mean()),rmse=float(r[:,s].mean())))
    for filename,data in [('shared_summary.csv',rows),('per_farm_metrics.csv',byfarm)]:
        if data:
            with (RUN/'analysis'/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    def compare(a,b):
        aa=sorted(grouped[a],key=lambda d:d['spec']['seed']);bb=sorted(grouped[b],key=lambda d:d['spec']['seed']);res=dict(candidate=a,reference=b);rng=np.random.default_rng(1104)
        for key in ['mae','rmse']:
            x=np.array([d['metrics']['per_farm_'+key] for d in aa]);y=np.array([d['metrics']['per_farm_'+key] for d in bb]);xf=x.mean(0);yf=y.mean(0);delta=100*(xf.mean()/yf.mean()-1);wins=int((xf<yf).sum());draw=rng.integers(nf,size=(10000,nf));ci=np.quantile(100*(xf[draw].mean(1)/yf[draw].mean(1)-1),[.025,.975])
            res[key]=dict(change_percent=float(delta),farms_improved=wins,total_farms=nf,paired_farm_bootstrap95_percent=ci.tolist(),per_seed_change_percent=(100*(x.mean(1)/y.mean(1)-1)).tolist(),passed=bool(delta<=-.1 and wins>nf/2))
        hx=np.mean([d['metrics']['high_power_rmse'] for d in aa]);hy=np.mean([d['metrics']['high_power_rmse'] for d in bb]);res['high_power_rmse_change_percent']=float(100*(hx/hy-1));res['passed']=res['mae']['passed'] and res['rmse']['passed'] and res['high_power_rmse_change_percent']<=.1
        return res
    comparisons=[]
    pairs=[('W0_RM','W0_R3'),('TF_RM','TF_R3'),('W0_RM','TF_RM'),('W0_RM','TF_R3'),('W0_R3','TF_R3'),('M1_CB','W0_R3'),('M1_CB','M4_WRONG_CB'),('M2_CBH','W0_R3'),('M2_CBH','M1_CB'),('M2_CBH','M4H_WRONG_CBH'),('M3_ANCHOR','M1_CB'),('M5_TF_CB','TF_R3'),('M5_TF_CB','M5N_TF_NEUTRAL'),('M5_TF_CB','M5W_TF_WRONG_CB')]
    for a,b in pairs:
        if len(grouped.get(a,[]))==5 and len(grouped.get(b,[]))==5:comparisons.append(compare(a,b))
    gate=dict(timestamp=now(),stage='pending',complete_jobs=len(done),total_jobs=len(specs),comparisons=comparisons,personalization_started=False,test2024_read=False)
    needed=['TF_R3','TIDE_R3','TFT_R3','W0_R3','M1_CB','M2_CBH','M4_WRONG_CB','M4H_WRONG_CBH']
    if all(len(grouped.get(g,[]))==5 for g in needed):
        strongest=min(['TF_R3','TIDE_R3','TFT_R3'],key=lambda g:np.mean([x['metrics']['mae'] for x in grouped[g]]));passed=[]
        for cand,ctrl in [('M1_CB','M4_WRONG_CB'),('M2_CBH','M4H_WRONG_CBH')]:
            checks=[compare(cand,ref) for ref in ['W0_R3',ctrl,strongest]];comparisons.extend(checks)
            if all(c['passed'] for c in checks):passed.append(cand)
        gate.update(stage='PASS_READY_FOR_PERSONALIZATION' if passed else 'FAIL_NO_LARGE_SCALE_PERSONALIZATION',eligible=passed,strongest_same_input_baseline=strongest,selected_candidate=min(passed,key=lambda g:np.mean([x['metrics']['mae'] for x in grouped[g]])) if passed else None)
    atomic_json(RUN/'analysis/GATES.json',gate)
    status='# 共享训练阶段结果\n\n'+f'更新时间：{now()}。完成 {len(done)}/{len(specs)} 个任务；未读取2024。\n\n'
    if rows:
        status+='|组别|seeds|MAE|RMSE|\n|---|---:|---:|---:|\n'
        for r in sorted(rows,key=lambda r:r['mae']):status+=f'|{r["group"]}|{r["seeds"]}|{r["mae"]:.6f}|{r["rmse"]:.6f}|\n'
    status+='\n判定：'+gate['stage']+'。未完成5 seeds的组只展示进度，不作验收结论。\n'
    (RUN/'analysis/RESULTS_CN.md').write_text(status)
    return gate
if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
