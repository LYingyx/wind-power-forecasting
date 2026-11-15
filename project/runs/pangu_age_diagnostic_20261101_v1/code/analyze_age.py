"""Paired 2022-fit / 2023-evaluate forecast-age diagnostic, CPU only."""
import csv,json,time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from prepare_age import OUT,PARENT,PROJECT,BUNDLE,CACHE,atomic,stamp,parent
import diagnose as old

SHIFTS=[0,12,24]
MODELS=['RAW','MOS','LOCAL','LOCAL_REGION','LOCAL_EQ','REGION_EQ','WRONG_LOCAL','WRONG_REGION','ORACLE_PAST_ERROR','LOCAL_UNBOUNDED']
HORIZONS={'1-6':np.arange(6),'1-3':np.arange(3),'4-6':np.arange(3,6)}

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def datasets():
    era=np.load(parent.OUT/'station_surface.npy',mmap_mode='r');data={};cohort={}
    for split in ['train','val']:
        base=old.read_split(split,era);n=len(base['origin']);cycle=np.load(CACHE/split/'weather_cycle_index.npy')[base['rows']]
        fi=np.load(parent.OUT/f'{split}_future_index.npy')[cycle];pi=np.load(parent.OUT/f'{split}_past_index.npy')[cycle]
        ef=np.asarray(era[fi]).transpose(0,2,1,3);ep=np.asarray(era[pi]).transpose(0,2,1,3)
        truef=np.hypot(ef[...,1],ef[...,2]);truep=np.hypot(ep[...,1],ep[...,2])
        mask=np.ones(n,bool);plans={}
        for age in [12,24]:
            plan=dict(np.load(OUT/f'age{age}_{split}_plan.npz'));valid=np.load(OUT/f'age{age}_valid.npy')
            np.testing.assert_array_equal(plan['rows'],base['rows']);np.testing.assert_array_equal(plan['origin'],base['origin'])
            mask &= valid[plan['future_frames']].all(1)&valid[plan['past_frames']].all(1);plans[age]=plan
        keep=np.flatnonzero(mask);np.save(OUT/f'{split}_common_rows.npy',base['rows'][keep])
        cohort[split]=dict(original=n,retained=len(keep),excluded=n-len(keep),earliest_utc=str(base['origin'][keep].min().astype('datetime64[s]')) if len(keep) else None)
        assert len(keep)>.95*n, 'Too many missing older forecasts; do not silently change study cohort'
        base={k:v[keep] for k,v in base.items()};truef,truep=truef[keep],truep[keep]
        for age in SHIFTS:
            d=dict(base)
            if age:
                raw=np.load(OUT/f'age{age}_frames.npy',mmap_mode='r');plan=plans[age]
                d['fu']=np.asarray(raw[plan['future_frames'][keep]]).transpose(0,2,1,3)
                d['pa']=np.asarray(raw[plan['past_frames'][keep]]).transpose(0,2,1,3)
                d['fv']=np.hypot(d['fu'][...,1],d['fu'][...,2]);d['pv']=np.hypot(d['pa'][...,1],d['pa'][...,2])
                d['age']=base['age']+age;d['clock']=base['clock'].copy();d['clock'][:,0]=d['age']/24
                uv=np.concatenate([d['pa'][...,1:3],d['fu'][:,:,:1,1:3]],2)
                d['local_wind']=np.linalg.norm(uv[:,:,1:],axis=-1);d['regional_wind']=np.linalg.norm(uv[:,:,:-1]*.25+uv[:,:,1:]*.75,axis=-1)
            d['y']=truef[:,:,1:7]-d['fv'][:,:,1:7];d['past_error']=truep-d['pv'];data[age,split]=d
        for age in SHIFTS:
            np.testing.assert_array_equal(data[age,split]['h'],base['h'])
    atomic(OUT/'COMMON_COHORT.json',dict(timestamp=stamp(),splits=cohort,same_observations_and_truth_all_ages=True,test_read=False))
    curves_file=np.load(parent.OUT/'inverse_curves_2022.npz')
    curves=[tuple(curves_file[f'{s}_{k}'] for k in range(4)) for s in range(56)]
    for d in data.values():old.equivalents(d,curves)
    return data

def main():
    assert all((OUT/f'AGE{x}_READY.json').exists() for x in [12,24])
    started=time.monotonic();data=datasets();tr=data[0,'train'];va=data[0,'val'];masks=old.distance_masks()
    dt,dv=old.donors(tr,tr),old.donors(va,tr)
    day=(va['origin']+28800).astype('datetime64[s]').astype('datetime64[D]');days,di=np.unique(day,return_inverse=True);counts=np.bincount(di)
    sse=np.zeros((3,len(MODELS),len(days),56,6),'f8');rows=[];corr=[]
    for ai,age in enumerate(SHIFTS):
        train,valid=data[age,'train'],data[age,'val']
        for s in range(56):
            bt,ot,et,rt,qt=old.features(train,s,masks);bv,ov,ev,rv,qv=old.features(valid,s,masks)
            target=valid['y'][:,s];np.add.at(sse[ai,0,:,s],di,target**2)
            obs={
                'MOS':(None,None,False),
                'LOCAL':(ot,ov,True),'LOCAL_REGION':(np.c_[ot,rt],np.c_[ov,rv],True),
                'LOCAL_EQ':(et,ev,False),'REGION_EQ':(np.c_[et,qt],np.c_[ev,qv],False),
                'WRONG_LOCAL':(ot[dt],ot[dv],True),'WRONG_REGION':(np.c_[ot[dt],rt[dt]],np.c_[ot[dv],rt[dv]],True),
                'ORACLE_PAST_ERROR':(train['past_error'][:,s],valid['past_error'][:,s],False),
                'LOCAL_UNBOUNDED':(ot,ov,False)}
            per={}
            for mi,name in enumerate(MODELS[1:],1):
                a,b,bounded=obs[name]
                if a is None:x,y=bt,bv
                else:
                    if bounded:b=np.clip(b,a.min(0),a.max(0))
                    x,y=np.c_[bt,a],np.c_[bv,b]
                assert np.isfinite(x).all() and np.isfinite(y).all()
                prediction=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,train['y'][:,s]).predict(y)
                se=(target-prediction)**2;np.add.at(sse[ai,mi,:,s],di,se);per[name]=se
            for j in range(6):
                for name,se in per.items():
                    raw=float(np.mean(target[:,j]**2));mos=float(np.mean(per['MOS'][:,j]));mse=float(se[:,j].mean())
                    rows.append(dict(shift_hours=age,site=s,station_id=str(parent.load('station_id')[s]),lead_hours=j+1,model=name,samples=len(target),mse=mse,
                        rmse_mps=mse**.5,raw_rmse_mps=raw**.5,mos_rmse_mps=mos**.5,
                        extra_mse_reduction_percent=100*(1-mse/mos),extra_mse_reduction_absolute=mos-mse))
                corr.append(dict(shift_hours=age,site=s,lead_hours=j+1,correlation=float(np.corrcoef(valid['past_error'][:,s,-1],target[:,j])[0,1])))
            atomic(OUT/'STATUS_ANALYSIS.json',dict(timestamp=stamp(),shift_hours=age,completed=(ai*56+s+1),total=168,elapsed_seconds=time.monotonic()-started))
            print('age',age,'site',s+1,'/56',flush=True)
    write_csv(OUT/'per_farm.csv',rows);write_csv(OUT/'past_future_correlation.csv',corr)
    np.savez(OUT/'daily_sufficient_statistics.npz',sse=sse,days=days,counts=counts,models=np.array(MODELS),shifts=np.array(SHIFTS))
    summarize()

def summarize():
    z=np.load(OUT/'daily_sufficient_statistics.npz');sse=z['sse'];days=z['days'];count=z['counts'].sum()
    weeks=days.astype(int)//7;uw=np.unique(weeks);weekly=np.stack([sse[:,:,weeks==w].sum(2) for w in uw],2)
    rng=np.random.default_rng(20261102);draw=rng.integers(0,len(uw),(1000,len(uw)))
    weekcounts=np.array([z['counts'][weeks==w].sum() for w in uw]);bootcounts=weekcounts[draw].sum(1)
    summaries=[];paired=[];boots={};point={};absolute_boots={};absolute_point={}
    for ai,age in enumerate(SHIFTS):
        for horizon,idx in HORIZONS.items():
            total=np.take(sse[ai],idx,axis=-1).sum((1,3))/(count*len(idx)) # model,farm
            raw,base=total[0],total[1]
            w=np.take(weekly[ai],idx,axis=-1).sum(-1) # model,week,farm
            bootbase=w[1][draw].sum(1)
            for mi,name in enumerate(MODELS):
                gain=100*(1-total[mi]/base);bootmodel=w[mi][draw].sum(1);sample=(100*(1-bootmodel/bootbase)).mean(1)
                abssample=((bootbase-bootmodel)/(bootcounts[:,None]*len(idx))).mean(1)
                boots[age,horizon,name]=sample;point[age,horizon,name]=gain
                absolute_boots[age,horizon,name]=abssample;absolute_point[age,horizon,name]=base-total[mi]
                summaries.append(dict(shift_hours=age,forecast_age_hours=[12+age,24+age],horizon=horizon,model=name,
                    macro_rmse_mps=float(np.sqrt(total[mi]).mean()),raw_macro_rmse_mps=float(np.sqrt(raw).mean()),mos_macro_rmse_mps=float(np.sqrt(base).mean()),
                    extra_MSE_reduction_percent=float(gain.mean()),median_extra_MSE_reduction_percent=float(np.median(gain)),positive_farms=int((gain>0).sum()),
                    absolute_MSE_reduction=float((base-total[mi]).mean()),ci95_week_bootstrap=np.quantile(sample,[.025,.975]).tolist(),
                    absolute_ci95_week_bootstrap=np.quantile(abssample,[.025,.975]).tolist()))
    for age in [12,24]:
        for horizon in HORIZONS:
            for name in ['LOCAL','LOCAL_REGION','LOCAL_EQ','REGION_EQ','ORACLE_PAST_ERROR']:
                delta=boots[age,horizon,name]-boots[0,horizon,name];gd=point[age,horizon,name]-point[0,horizon,name]
                absdelta=absolute_boots[age,horizon,name]-absolute_boots[0,horizon,name]
                absdiff=absolute_point[age,horizon,name]-absolute_point[0,horizon,name]
                paired.append(dict(shift_hours=age,horizon=horizon,model=name,change_in_gain_percentage_points=float(gd.mean()),farms_with_increased_gain=int((gd>0).sum()),ci95_paired_week_bootstrap=np.quantile(delta,[.025,.975]).tolist(),
                    change_in_absolute_MSE_reduction=float(absdiff.mean()),absolute_ci95_paired_week_bootstrap=np.quantile(absdelta,[.025,.975]).tolist()))
    atomic(OUT/'SUMMARY.json',dict(timestamp=stamp(),results=summaries,paired_age_effects=paired,all_ages_use_same_samples=True,test_read=False,
        inference='diagnostic weather MSE only; not power forecast MAE/RMSE; initial-source replacement not tested'))
    validate(z,summaries)
    from report_age import main as report
    report()

def validate(z,summary):
    rows=list(csv.DictReader((OUT/'per_farm.csv').open()));assert len(rows)==3*56*6*9
    totals=z['sse'].sum(2);n=z['counts'].sum()
    for r in rows:
        a=SHIFTS.index(int(r['shift_hours']));m=MODELS.index(r['model']);s=int(r['site']);j=int(r['lead_hours'])-1
        np.testing.assert_allclose(float(r['mse']),totals[a,m,s,j]/n,rtol=1e-6)
    cohort=json.loads((OUT/'COMMON_COHORT.json').read_text())
    baseline_check={}
    if not any(v['excluded'] for v in cohort['splits'].values()):
        prior=json.loads((parent.OUT/'D2_ROBUSTNESS_SUMMARY.json').read_text())
        for h in HORIZONS:
            a=next(v for v in summary if v['shift_hours']==0 and v['model']=='LOCAL' and v['horizon']==h)
            b=next(v for v in prior['results'] if v['model']=='LOCAL_RAW_CLIPPED' and v['horizon']==h)
            assert abs(a['extra_MSE_reduction_percent']-b['mean_incremental_MSE_reduction_percent'])<1e-6
        baseline_check['previous_LOCAL_result_reproduced']=True
    atomic(OUT/'COMPLETENESS_AUDIT.json',dict(timestamp=stamp(),passed=True,per_farm_rows=len(rows),farms=56,forecast_age_conditions=3,diagnostic_regressions=3*56*9,
        independent_daily_statistics_reconstruction=True,cohort=cohort,baseline_check=baseline_check,test_read=False))
if __name__=='__main__':main()
