"""Recompute source56 metrics and descriptive strata from 40 prediction archives."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_tables import ROOT, OUT, ART, save, bootstrap

def main():
    cache=ROOT/'data/model_cache_pad5_f8';static=cache/'static'
    ids=np.load(static/'station_id.npy').astype(str)
    capacity=np.load(static/'station_capacity_mw.npy').astype(float)
    desc=pd.read_csv(OUT/'tables/source56_descriptors.csv',dtype={'station_id':str}).set_index('station_id').loc[ids]
    ramp_threshold=desc.ramp_1h_q90.to_numpy()
    reps=pd.read_csv(OUT/'tables/source56_representatives.csv',dtype={'station_id':str}).station_id.tolist()
    # Existing known failure is clearly additional and post-hoc, not a medoid.
    case_farms=reps+['1043056']
    farmrows=[];stratarows=[];leadrows=[];checks=[];cases=[];casemeta=[];blockrows=[]
    for split in ['val','test']:
        origins=np.load(cache/split/'origin_unix.npy')
        sample=np.load(cache/split/'sample_index.npy')
        y=np.asarray(np.load(cache/split/'station_future.npy',mmap_mode='r'),dtype=np.float64)
        history=np.load(cache/split/'station_history.npy',mmap_mode='r')
        t=pd.to_datetime(origins,unit='s',utc=True)
        local=t.tz_convert('Asia/Shanghai')
        month=t.month.to_numpy();hours=local.hour.to_numpy()
        previous=np.asarray(history[:,:,-1],dtype=float)
        past_ramp=previous-np.asarray(history[:,:,-5],dtype=float)
        n=len(origins);assert y.shape==(n,56,24)
        # Origin classification: no future target required except explicitly named diagnostic strata.
        season=np.array(['DJF','MAM','JJA','SON'])[((month%12)//3)]
        timebin=hours//6
        powerbin=np.digitize(previous,[.1,.3,.7])
        rampbin=np.where(past_ramp>ramp_threshold,1,np.where(past_ramp < -ramp_threshold,-1,0))
        extended=np.concatenate([previous[:,:,None],y],axis=2)
        future_event=(np.max(np.abs(extended[:,:,4:]-extended[:,:,:-4]),axis=2)>=.2)
        selectors=[]
        for group in ['DJF','MAM','JJA','SON']:
            selectors.append(('season_origin_UTC',group,season==group))
        for i,label in enumerate(['00-06','06-12','12-18','18-24']):
            selectors.append(('origin_hour_Asia_Shanghai',label,timebin==i))
        for i,label in enumerate(['CF<0.1','0.1<=CF<0.3','0.3<=CF<0.7','CF>=0.7']):
            selectors.append(('observed_power_at_origin',label,powerbin==i))
        for i,label in [(-1,'past_1h_fall'),(0,'past_1h_moderate'),(1,'past_1h_rise')]:
            selectors.append(('observed_1h_ramp_train_q90',label,rampbin==i))
        selectors.extend([('realized_future_1h_ramp_diagnostic','max_abs_delta_CF>=0.2',future_event),
                          ('realized_future_1h_ramp_diagnostic','max_abs_delta_CF<0.2',~future_event)])
        targetvar=y.var(axis=(0,2))
        bidx=(origins-origins.min())//(7*86400)
        unique_blocks=np.unique(bidx)
        case_index=[]
        if split=='test':
            # Earliest common-cache April origin at 06:00 local satisfying observed-state class.
            # No prediction errors or future targets are used to choose these instants.
            for station in case_farms:
                j=list(ids).index(station)
                base=(local.month.to_numpy()==4)&(local.hour.to_numpy()==6)&(local.minute.to_numpy()==0)
                masks={'ordinary':base&(np.abs(past_ramp[:,j])<=ramp_threshold[j])&(previous[:,j]>=.1)&(previous[:,j]<.7),
                       'recent_ramp':base&(np.abs(past_ramp[:,j])>ramp_threshold[j])}
                for typ,mask in masks.items():
                    pos=np.flatnonzero(mask);fallback=False
                    if not len(pos):
                        mask=(local.hour.to_numpy()==6)&(local.minute.to_numpy()==0)
                        mask&=(np.abs(past_ramp[:,j])>ramp_threshold[j]) if typ=='recent_ramp' else ((previous[:,j]>=.1)&(previous[:,j]<.7))
                        pos=np.flatnonzero(mask);fallback=True
                    if not len(pos):continue
                    i=int(pos[0]);caseid=f'{station}_{typ}'
                    case_index.append((caseid,i,j))
                    casemeta.append(dict(case_id=caseid,station_id=station,kind=typ,source='posthoc_failure' if station=='1043056' else 'train_descriptor_medoid',
                        origin_index=i,station_index=j,origin_unix=int(origins[i]),origin_local=str(local[i]),
                        current_cf=previous[i,j],observed_ramp_1h=past_ramp[i,j],train_ramp_q90=ramp_threshold[j],fallback_full_year=fallback))
                    for lead,value in enumerate(history[i,j,:]):
                        cases.append(dict(case_id=caseid,model='Observed_history',seed=-1,lead_hours=(lead-23)/4,value_cf=float(value)))
                    for lead,value in enumerate(y[i,j,:]):
                        cases.append(dict(case_id=caseid,model='Observed_future',seed=-1,lead_hours=(lead+1)/4,value_cf=float(value)))
        for model in ['Persistence','E0','E2C']:
            for seed in ([0] if model=='Persistence' else range(42,52)):
                if model=='Persistence':pred=np.broadcast_to(previous[:,:,None],y.shape)
                else:
                    path=ART/f'column_weather_20260928_v1/runs/main/{model}/seed{seed}/predictions_{split}.npz'
                    with np.load(path) as z:
                        assert np.array_equal(origins,z['origin_unix'])
                        assert np.array_equal(sample,z['sample_index'])
                        truth=np.asarray(z['station_target_mw'],dtype=np.float64)/capacity[None,:,None]
                        target_diff=float(np.max(np.abs(truth-y)));assert target_diff<2e-6,target_diff
                        del truth
                        pred=np.asarray(z['station_prediction_mw'],dtype=np.float64)/capacity[None,:,None]
                    assert np.isfinite(pred).all()
                err=pred-y;ae=np.abs(err)
                mae=ae.mean(axis=(0,2));mse=np.square(err).mean(axis=(0,2))
                for j,station in enumerate(ids):
                    farmrows.append(dict(model=model,split=split,seed=seed,station=station,mae=mae[j],mse=mse[j],rmse=np.sqrt(mse[j]),r2=1-mse[j]/targetvar[j]))
                for lead in range(24):
                    leadrows.append(dict(model=model,split=split,seed=seed,lead_hours=(lead+1)/4,mae=ae[:,:,lead].mean(),mse=np.square(err[:,:,lead]).mean()))
                # Retain station and sample counts so rare strata do not silently change the denominator.
                ae_origin=ae.mean(axis=2);se_origin=np.square(err).mean(axis=2)
                for dim,group,mask in selectors:
                    if mask.ndim==1: mask=np.broadcast_to(mask[:,None],(n,56))
                    count=mask.sum(axis=0);denom=np.maximum(count,1)
                    ma=(ae_origin*mask).sum(axis=0)/denom
                    ms=(se_origin*mask).sum(axis=0)/denom
                    for j,station in enumerate(ids):
                        stratarows.append(dict(model=model,split=split,seed=seed,station=station,dimension=dim,group=group,
                            origins=int(count[j]),target_points=int(count[j])*24,mae=ma[j] if count[j] else np.nan,mse=ms[j] if count[j] else np.nan))
                for block in unique_blocks:
                    mask=bidx==block
                    values=ae_origin[mask].mean(0)
                    for j,station in enumerate(ids):
                        blockrows.append(dict(model=model,split=split,seed=seed,station=station,block=int(block),origins=int(mask.sum()),mae=values[j]))
                for caseid,i,j in case_index:
                    for lead in range(24):
                        cases.append(dict(case_id=caseid,model=model,seed=seed,lead_hours=(lead+1)/4,value_cf=float(pred[i,j,lead])))
                if model!='Persistence':
                    archived=json.loads((path.parent/f'metrics_{split}.json').read_text())['station_mae_cf']
                    diff=float(mae.mean()-archived);assert abs(diff)<2e-7,(model,seed,split,diff)
                    checks.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,model=model,seed=seed,split=split,
                        archived_mae=archived,recomputed_mae=float(mae.mean()),difference=diff,target_max_cf_difference=target_diff,
                        origin_alignment=True,sample_index_alignment=True))
                print(split,model,seed,round(float(mae.mean()),8),flush=True)
                del err,ae,pred
    farm=pd.DataFrame(farmrows);save(farm,'prediction_station_seed',False)
    strata=pd.DataFrame(stratarows);save(strata,'scenario_station_seed',False)
    save(pd.DataFrame(leadrows),'prediction_lead_seed',False)
    save(pd.DataFrame(blockrows),'prediction_week_station_seed',False)
    save(pd.DataFrame(cases),'forecast_cases',False);save(pd.DataFrame(casemeta),'forecast_case_selection')
    summary=strata.groupby(['model','split','dimension','group','station']).agg(mae=('mae','mean'),mse=('mse','mean'),origins=('origins','first')).reset_index()
    save(summary,'scenario_per_station',False)
    ss=summary.groupby(['model','split','dimension','group']).agg(mae=('mae','mean'),rmse_pooled=('mse',lambda x:np.sqrt(x.mean())),farms=('mae','count'),origin_station_pairs=('origins','sum')).reset_index()
    save(ss,'scenario_summary')
    contrasts=[]
    for (split,dim,group),g in summary.groupby(['split','dimension','group']):
        wide=g.pivot(index='station',columns='model',values='mae')
        contrasts.append(dict(split=split,dimension=dim,group=group,unit='farm_after_seed_mean; exploratory strata',**bootstrap(wide.E2C-wide.E0)))
    save(pd.DataFrame(contrasts),'scenario_e2c_minus_e0')
    (OUT/'prediction_verification.json').write_text(json.dumps(dict(archives=len(checks),max_mae_abs_difference=max(abs(c['difference']) for c in checks),
            dimensions=['season UTC','origin time Asia/Shanghai','observed current CF','observed past 1h ramp with training q90','realized future ramp diagnostic'],
            checks=checks),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
