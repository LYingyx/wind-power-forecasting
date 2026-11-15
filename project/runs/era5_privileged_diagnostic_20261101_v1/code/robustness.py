"""Selection-year follow-up: train-range clipping of observation inputs only.

Triggered by the audited 2022->2023 input-support shift, explicitly secondary.
No rows/farms are dropped and no validation-derived clipping threshold is used.
"""
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from diagnose import read_split,distance_masks,features,donors,write_csv
from prepare_era5 import OUT,atomic,stamp,load

def main():
    era=np.load(OUT/'station_surface.npy',mmap_mode='r');tr=read_split('train',era);va=read_split('val',era)
    # features() also offers inverse features; dummy arrays are never supplied to a fitted model here.
    for d in [tr,va]:
        for k in ['local_eq','regional_eq']:d[k]=np.zeros((len(d['origin']),56,6),'f4')
        for k in ['local_valid','regional_valid']:d[k]=np.zeros((len(d['origin']),56,6),bool)
    names=['LOCAL_RAW_CLIPPED','LOCAL_REGION_RAW_CLIPPED','WRONGDAY_RAW_CLIPPED']
    atomic(OUT/'ROBUSTNESS_PROTOCOL.json',dict(timestamp=stamp(),selection_year_followup=True,
        motivation='2022->2023 observed-power support shift at site51; primary D2 retained',
        transform='clip each observation feature to its2022 training min/max; weather features untouched; retain all farms/rows',
        models=names,ridge_alpha=10,test_read=False))
    z=np.load(OUT/'D2_daily_sufficient_statistics.npz');days=z['days'];counts=z['counts'];dayidx=np.searchsorted(days,(va['origin']+28800).astype('datetime64[s]').astype('datetime64[D]'))
    base=z['sse'][1];sse=np.zeros((3,len(days),56,6));rows=[];drift=[];masks=distance_masks();dt,dv=donors(tr,tr),donors(va,tr)
    for s in range(56):
        bt,ot,_,rt,_=features(tr,s,masks);bv,ov,_,rv,_=features(va,s,masks)
        for j,name in enumerate(names):
            a=ot if j==0 else np.c_[ot,rt];b=ov if j==0 else np.c_[ov,rv]
            if j==2:a,b=a[dt],a[dv]
            lo=a.min(0);hi=a.max(0);fraction=float(((b<lo)|(b>hi)).mean())
            pred=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(np.c_[bt,a],tr['y'][:,s]).predict(np.c_[bv,np.clip(b,lo,hi)])
            se=(va['y'][:,s]-pred)**2;np.add.at(sse[j,:,s],dayidx,se)
            for lead in range(6):rows.append(dict(site=s,model=name,lead_hours=lead+1,rmse_mps=float(np.mean(se[:,lead])**.5),incremental_vs_MOS_percent=float(100*(1-se[:,lead].sum()/base[:,s,lead].sum()))))
            drift.append(dict(site=s,station_id=str(load('station_id')[s]),model=name,out_of_train_range_fraction=fraction,
                train_power_max=float(tr['h'][:,s].max()),validation_power_max=float(va['h'][:,s].max())))
    write_csv(OUT/'D2_robustness_per_farm.csv',rows);write_csv(OUT/'D2_input_range_shift.csv',drift)
    np.savez(OUT/'D2_robustness_daily.npz',sse=sse,days=days,counts=counts,models=np.array(names))
    weeks=days.astype(int)//7;uw=np.unique(weeks);ws=np.stack([sse[:,weeks==w].sum(1) for w in uw],1);bs=np.stack([base[weeks==w].sum(0) for w in uw],0)
    rng=np.random.default_rng(20261101);draw=rng.integers(0,len(uw),(1000,len(uw)));summary=[]
    for c,name in enumerate(names):
        for horizon,idx in [('1-6',np.arange(6)),('1-3',np.arange(3)),('4-6',np.arange(3,6))]:
            b=base[:,:,idx].sum((0,2));a=sse[c,:,:,idx].sum((0,1));gain=100*(1-a/b)
            wb=bs[:,:,idx].sum(2);wa=ws[c,:,:,idx].sum(0)
            boots=(100*(1-wa[draw].sum(1)/wb[draw].sum(1))).mean(1)
            summary.append(dict(model=name,horizon=horizon,mean_incremental_MSE_reduction_percent=float(gain.mean()),median_incremental_MSE_reduction_percent=float(np.median(gain)),positive_farms=int((gain>0).sum()),ci95_week_bootstrap=np.quantile(boots,[.025,.975]).tolist()))
    atomic(OUT/'D2_ROBUSTNESS_SUMMARY.json',dict(timestamp=stamp(),results=summary,secondary_selection_year_analysis=True,test_read=False))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
