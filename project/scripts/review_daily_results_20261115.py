"""Audit every completed artifact and compare paired farms/seeds from raw metrics."""
from pathlib import Path
import collections, csv, datetime, hashlib, json, math
import numpy as np

P=Path(__file__).resolve().parents[1]
NAMES=['daily_protocol_restart_20261114_v1','daily_ablation_restart_20261114_v1','daily_site_finetune_20261114_v1']
OUT=P/'analysis/daily_protocol_review_20261115'
FP='bb838b5c77790c527bea81bacbffd12fd9f626d199b502c2cfb435dcc5f1c036'
SEEDS=list(range(42,47));NF=55
COHORT=json.loads((P/'runs'/NAMES[0]/'assets/COHORT.json').read_text())
hash_cache={}
def sha(path):
    path=path.resolve()
    if path not in hash_cache:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        hash_cache[path]=h.hexdigest()
    return hash_cache[path]
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvwrite(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True);groups=collections.defaultdict(dict);audit=[];missing=[]
    def add(group,spec,m,path,epoch,params):
        sites=[spec['station']] if 'station' in spec else list(range(NF))
        for i,station in enumerate(sites):
            row=dict(group=group,seed=spec['seed'],station=station,station_id=COHORT['station_ids'][station],
                     name=COHORT['names'][station],mae=m['per_farm_mae'][i],rmse=m['per_farm_rmse'][i],
                     high_power_rmse=m['high_power_per_farm_rmse'][i],high_count=m['high_power_counts'][i],
                     count=m['counts'][i],selected_epoch=epoch,parameter_count=params,source=str(path.relative_to(P)))
            assert row['count']==29647,(group,row['count'])
            for label,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
                row['mae_'+label]=float(np.mean(m['lead_mae'][i][a:b]))
                row['rmse_'+label]=float(np.sqrt(np.mean(np.square(m['lead_rmse'][i][a:b]))))
            key=(spec['seed'],station);assert key not in groups[group],(group,key)
            groups[group][key]=row
    for name in NAMES:
        r=P/'runs'/name;manifest=json.loads((r/'manifest.json').read_text());count=0;fixed=0;refs=0;norms=[]
        for rel,h in json.loads((r/'SCIENCE_FREEZE.json').read_text())['files'].items():
            assert sha(r/rel)==h,(name,'frozen hash',rel)
        for spec in manifest:
            job=r/'jobs'/spec['id']
            if not (job/'SUCCESS.json').exists():missing.append(name+'/'+spec['id']);continue
            assert not (job/'FAILURE.json').exists()
            success=json.loads((job/'SUCCESS.json').read_text());assert success['passed']
            d=json.loads((job/'metrics.json').read_text());assert d['requested_spec']==spec
            assert not d['smoke_only'];assert spec['preprocess_fingerprint']==FP
            expected_year=2022 if spec['family']=='B0' and spec.get('fold',-1)>=0 else 2023
            assert d['evaluation_year']==expected_year
            assert sha(job/'best.pt')==d['checkpoint_sha256'],spec['id']
            if 'reference_sha256' in d:assert sha(job/'reference_prediction.npy')==d['reference_sha256'];refs+=1
            if 'source_sha256' in d:assert sha(r/spec['pretrained'])==d['source_sha256']
            m=d['metrics'];arr=np.array([m['mae'],m['rmse'],*m['per_farm_mae'],*m['per_farm_rmse']]);assert np.isfinite(arr).all()
            assert np.isclose(np.mean(m['per_farm_mae']),m['mae'],atol=1e-12,rtol=0)
            assert np.isclose(np.mean(m['per_farm_rmse']),m['rmse'],atol=1e-12,rtol=0)
            count+=1
            if spec['stage'] not in ['tune','reference']:
                add(spec['group'],spec,m,job/'metrics.json',d['checkpoint_epoch'],d['parameter_count'])
            if spec['stage']=='personalization':
                for ep in [1,6]:
                    fm=json.loads((job/f'epoch{ep}.json').read_text());assert sha(job/f'epoch{ep}.pt')==fm['checkpoint_sha256'];fixed+=1
                    add(spec['group']+f'_E{ep}',spec,fm['metrics'],job/f'epoch{ep}.json',ep,d['parameter_count'])
                if spec['mode']=='P1_FULL':
                    z=json.loads((job/'Z1.json').read_text());assert z['preprocess_fingerprint']==FP and z['observed_power_delay_minutes']==0
                    for key,mode in [('p0','P0_SHARED_Z1'),('e1','P1_FULL_E1_Z1')]:
                        zm=z['analyses'][key]['bin30'];mm=dict(mae=zm['mae'],rmse=zm['rmse'],per_farm_mae=[zm['mae']],per_farm_rmse=[zm['rmse']],high_power_per_farm_rmse=[zm['high_power_rmse']],high_power_counts=[zm['high_power_count']],counts=[zm['count']],lead_mae=[zm['lead_mae']],lead_rmse=[np.sqrt(zm['lead_mse']).tolist()])
                        add(spec['family']+'_'+mode,spec,mm,job/'Z1.json',0 if key=='p0' else 1,d['parameter_count'])
        audit.append(dict(run=name,planned=len(manifest),completed=count,frozen_hashes_passed=True,
                          selected_checkpoints_verified=count,fixed_checkpoints_verified=fixed,reference_arrays_verified=refs,
                          complete=json.loads((r/'COMPLETE.json').read_text()) if (r/'COMPLETE.json').exists() else None))
        print('audited',name,count,len(manifest),flush=True)
    dump('COMPLETION_AUDIT.json',dict(timestamp=datetime.datetime.now().astimezone().isoformat(),passed=not missing,
         planned=2571,completed=sum(x['completed'] for x in audit),missing=missing,runs=audit,
         hashed_distinct_files=len(hash_cache),data_fingerprint=FP,scope='2571 training jobs including 2023 validation; excludes 2024 and conditional LM models'))
    if missing:print('Waiting for',missing,flush=True);return
    expected={(seed,s) for seed in SEEDS for s in range(NF)}
    for g,v in groups.items():assert set(v)==expected,(g,len(v))
    allrows=[row for g in sorted(groups) for _,row in sorted(groups[g].items())]
    csvwrite('all_farm_seed_metrics.csv',allrows)
    summaries=[];seedrows=[]
    for g in sorted(groups):
        v=list(groups[g].values());s=dict(group=g,cells=len(v),mae=float(np.mean([r['mae'] for r in v])),rmse=float(np.mean([r['rmse'] for r in v])),high_power_rmse=float(np.mean([r['high_power_rmse'] for r in v if r['high_count']>0])),epoch0_selected=sum(r['selected_epoch']==0 for r in v),selected_epoch_median=float(np.median([r['selected_epoch'] for r in v])),parameter_count=v[0]['parameter_count'])
        summaries.append(s)
        for seed in SEEDS:
            vv=[r for r in v if r['seed']==seed]
            seedrows.append(dict(group=g,seed=seed,**{m:float(np.mean([r[m] for r in vv])) for m in ['mae','rmse','high_power_rmse']}))
    csvwrite('group_summary.csv',summaries);csvwrite('per_seed_summary.csv',seedrows)
    pairs=[]
    def pair(a,b):
        if (a,b) not in pairs:pairs.append((a,b))
    for a,b in [('W0_R3','TF_R3'),('SHARED_W0_BASE','TF_BASE'),('INDEPENDENT_W0_BASE','INDEPENDENT_TF_BASE'),('INDEPENDENT_W0_R3','INDEPENDENT_TF_R3'),('W0_R3','SHARED_W0_BASE'),('TF_R3','TF_BASE'),('INDEPENDENT_W0_R3','INDEPENDENT_W0_BASE'),('INDEPENDENT_TF_R3','INDEPENDENT_TF_BASE'),('M2_CBH','TF_R3'),('M2_CBH','W0_R3'),('M1_CB','W0_R3'),('M2_CBH','M1_CB'),('M2_CBH','M4H_WRONG_CBH'),('M1_CB','M4_WRONG_CB'),('W0_RM','W0_R3'),('TF_RM','TF_R3'),('W0_R3','TIDE_R3'),('W0_R3','TFT_R3'),('ENC_neutral_E2L','TF_R3'),('ENC_neutral_E2L','TF_P1_FULL'),('ENC_neutral_E2L','ENC_neutral_E0'),('ENC_neutral_E2L','ENC_neutral_E2N')]:pair(a,b)
    for family in ['W0','TF']:
        for inp in ['BASE','R3']:
            shared='SHARED_W0_BASE' if (family,inp)==('W0','BASE') else family+'_'+inp
            pair('INDEPENDENT_'+family+'_'+inp,shared)
        for mode in ['P1_FULL','P2_HEAD']:
            pair(family+'_'+mode,family+'_R3')
            pair(family+'_'+mode+'_E1',family+'_R3');pair(family+'_'+mode+'_E6',family+'_R3')
            pair('W0_'+mode,'TF_'+mode)
            pair('W0_'+mode+'_E1','TF_'+mode+'_E1');pair('W0_'+mode+'_E6','TF_'+mode+'_E6')
        pair(family+'_P0_SHARED_Z1',family+'_R3');pair(family+'_P1_FULL_E1_Z1',family+'_P1_FULL_E1')
    for g in groups:
        if g.startswith('ENC_'):pair(g,'W0_R3' if '_neutral_' in g else 'M2_CBH')
        if g.startswith(('INPUT_','FACTOR_')):pair(g,g.split('_')[1]+'_R3')
    comparisons=[];pairrows=[];farmrows=[];seedpairs=[];horizon=[]
    rng=np.random.default_rng(20261115);ix=rng.integers(NF,size=(10000,NF))
    def array(g,m):return np.array([[groups[g][seed,s][m] for s in range(NF)] for seed in SEEDS])
    for a,b in pairs:
        res=dict(candidate=a,reference=b)
        for metric in ['mae','rmse','high_power_rmse']:
            x=array(a,metric);y=array(b,metric);delta=100*(x/y-1);xf=x.mean(0);yf=y.mean(0)
            ci=np.quantile(100*(xf[ix].mean(1)/yf[ix].mean(1)-1),[.025,.975])
            res[metric]=dict(candidate_mean=float(x.mean()),reference_mean=float(y.mean()),change_percent=float(100*(x.mean()/y.mean()-1)),
                 farm_mean_change_median=float(np.median(100*(xf/yf-1))),farm_wins=int((xf<yf).sum()),paired_cells_won=int((x<y).sum()),total_cells=275,
                 farms_all5_seeds_improved=int(np.all(x<y,axis=0).sum()),seed_macro_wins=int((x.mean(1)<y.mean(1)).sum()),
                 per_seed_change_percent=(100*(x.mean(1)/y.mean(1)-1)).tolist(),farm_bootstrap95_percent=ci.tolist())
        xm=array(a,'mae');ym=array(b,'mae');xr=array(a,'rmse');yr=array(b,'rmse')
        res['farms_both_metrics_improved']=int(((xm.mean(0)<ym.mean(0))&(xr.mean(0)<yr.mean(0))).sum())
        res['farms_both_all5_seeds_improved']=int(np.all((xm<ym)&(xr<yr),axis=0).sum())
        for i,seed in enumerate(SEEDS):
            seedpairs.append(dict(candidate=a,reference=b,seed=seed,mae_change_percent=res['mae']['per_seed_change_percent'][i],rmse_change_percent=res['rmse']['per_seed_change_percent'][i],mae_farms_won=int((xm[i]<ym[i]).sum()),rmse_farms_won=int((xr[i]<yr[i]).sum()),both_farms_won=int(((xm[i]<ym[i])&(xr[i]<yr[i])).sum())))
            for s in range(NF):
                pairrows.append(dict(candidate=a,reference=b,seed=seed,station_id=COHORT['station_ids'][s],name=COHORT['names'][s],candidate_mae=xm[i,s],reference_mae=ym[i,s],mae_change_percent=100*(xm[i,s]/ym[i,s]-1),candidate_rmse=xr[i,s],reference_rmse=yr[i,s],rmse_change_percent=100*(xr[i,s]/yr[i,s]-1)))
        for s in range(NF):
            farmrows.append(dict(candidate=a,reference=b,station_id=COHORT['station_ids'][s],name=COHORT['names'][s],mae_change_percent=100*(xm[:,s].mean()/ym[:,s].mean()-1),rmse_change_percent=100*(xr[:,s].mean()/yr[:,s].mean()-1),mae_seed_wins=int((xm[:,s]<ym[:,s]).sum()),rmse_seed_wins=int((xr[:,s]<yr[:,s]).sum()),both_seed_wins=int(((xm[:,s]<ym[:,s])&(xr[:,s]<yr[:,s])).sum())))
        for label in ['0_1h','1_3h','3_6h']:
            row=dict(candidate=a,reference=b,horizon=label)
            for metric in ['mae','rmse']:
                x=array(a,metric+'_'+label);y=array(b,metric+'_'+label)
                row[metric+'_change_percent']=float(100*(x.mean()/y.mean()-1));row[metric+'_farms_won']=int((x.mean(0)<y.mean(0)).sum())
            horizon.append(row)
        comparisons.append(res)
    dump('COMPARISONS.json',dict(timestamp=datetime.datetime.now().astimezone().isoformat(),convention='100*(candidate/reference-1); negative is better. Farm means use all5 seeds; cells are dependent, not independent samples.',
         caveat='2023 also selected checkpoints/configs; farm-bootstrap intervals are descriptive, ignore spatial correlation and multiple comparison selection, and do not establish independent significance.',comparisons=comparisons))
    csvwrite('paired_farm_seed_comparisons.csv',pairrows);csvwrite('per_farm_comparisons.csv',farmrows);csvwrite('per_seed_comparisons.csv',seedpairs);csvwrite('horizon_comparisons.csv',horizon)
    csvwrite('comparison_summary.csv',[dict(candidate=c['candidate'],reference=c['reference'],mae_change_percent=c['mae']['change_percent'],rmse_change_percent=c['rmse']['change_percent'],high_power_rmse_change_percent=c['high_power_rmse']['change_percent'],mae_farms_won=c['mae']['farm_wins'],rmse_farms_won=c['rmse']['farm_wins'],both_farms_won=c['farms_both_metrics_improved'],mae_paired_wins=c['mae']['paired_cells_won'],rmse_paired_wins=c['rmse']['paired_cells_won']) for c in comparisons])
    print('Verified',sum(x['completed'] for x in audit),'jobs;',len(groups),'complete groups;',len(comparisons),'paired comparisons',flush=True)

if __name__=='__main__':main()
