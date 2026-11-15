"""Older issue, unchanged valid times and observations; no new ERA5 reads."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time
import numpy as np

RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];OUT=(RUN/'output').resolve()
PARENT=PROJECT/'runs/era5_privileged_diagnostic_20261101_v1'
sys.path.insert(0,str(PARENT/'code'))
import prepare_era5 as parent
sys.path.insert(0,str(PROJECT/'src'))
from wind_forcing.data.operational import source_path
BUNDLE=parent.BUNDLE;CACHE=BUNDLE/'cache';SOURCE=Path('/hj102_data_large/ERA5_forecasting/Pangu/Forecast_data')
ASSETS=PROJECT/'runs/final_mechanism_20261030_v1/assets';stamp=parent.stamp

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def selected(split):
    o=np.load(CACHE/split/'origin_unix.npy');keep=o%3600==0
    if split=='val':keep &= np.load(ASSETS/'val_keep.npy')
    rows=np.flatnonzero(keep);cycle=np.load(CACHE/split/'weather_cycle_index.npy')[rows]
    return dict(rows=rows,origin=o[rows],cycle=cycle,issue=np.load(CACHE/split/'origin_issue_unix.npy')[rows],
                future=np.load(CACHE/split/'weather_valid_unix.npy')[cycle],
                past=np.load(BUNDLE/'physical'/split/'past_valid_unix.npy')[cycle])

def plan(shift):
    plans={};keys=set()
    for split in ['train','val']:
        d=selected(split);d['issue']=d['issue']-shift*3600
        assert np.all((d['origin']-d['issue'])>= (12+shift)*3600)
        assert np.all((d['origin']-d['issue'])< (24+shift)*3600)
        assert np.all(d['issue']+12*3600<=d['origin'])
        for it,valid in zip(d['issue'],np.concatenate([d['past'],d['future']],1)):
            keys.update((int(it),int((v-it)//3600)) for v in valid)
        plans[split]=d
    keys=sorted(keys);lookup={k:i for i,k in enumerate(keys)}
    for split,d in plans.items():
        for name in ['past','future']:
            d[name+'_frames']=np.array([[lookup[(int(it),int((v-it)//3600))] for v in vs] for it,vs in zip(d['issue'],d[name])])
        np.savez(OUT/f'age{shift}_{split}_plan.npz',**d)
    np.save(OUT/f'age{shift}_keys.npy',np.array(keys,dtype='i8'))
    return keys

def extract(shift,workers):
    OUT.mkdir(parents=True,exist_ok=True);keys=plan(shift);prefix=f'age{shift}'
    ids=parent.load('bilinear_patch_index');weights=parent.load('bilinear_weight');mean=parent.load('nwp_mean')[:4];std=parent.load('nwp_std')[:4]
    path=OUT/f'{prefix}_frames.npy';donepath=OUT/f'{prefix}_done.npy'
    if not path.exists():
        raw=np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=(len(keys),56,4));raw[:]=np.nan;raw.flush()
        done=np.lib.format.open_memmap(donepath,mode='w+',dtype='?',shape=(len(keys),));done[:]=False;done.flush()
    else:
        raw=np.lib.format.open_memmap(path,mode='r+');done=np.lib.format.open_memmap(donepath,mode='r+');assert len(raw)==len(keys)
    def read(i):
        issue,lead=keys[i];path=source_path(SOURCE,issue,lead)
        if not path.is_file():return int(i),None,dict(path=str(path),missing=True)
        before=path.stat();data=np.load(path,mmap_mode='r',allow_pickle=False)
        if data.shape!=(1,14,721,1440) or data.dtype!=np.float32:raise ValueError(f'Unexpected source format {path}')
        crop=np.array(data[0,:4,171:195,483:508]);assert np.isfinite(crop).all()
        # Reproduce the current operational order: normalize grid, then interpolate.
        normal=(crop-mean[:,None,None])/std[:,None,None]
        loc=(normal.reshape(4,-1)[:,ids]*weights).sum(-1).T
        physical=(loc*std+mean).astype('f4');after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        return int(i),physical,dict(path=str(path),size=before.st_size,mtime_ns=before.st_mtime_ns,sha256=hashlib.sha256(physical.tobytes()).hexdigest())
    pending=np.flatnonzero(~done);start=time.monotonic();initial=int(done.sum());missing=[]
    with (OUT/f'{prefix}_receipts.jsonl').open('a') as receipt,ThreadPoolExecutor(max_workers=workers) as pool:
        for k,(i,value,meta) in enumerate(pool.map(read,pending),1):
            if value is None:missing.append(meta['path'])
            else:raw[i]=value
            receipt.write(json.dumps(dict(index=i,**meta))+'\n');done[i]=True
            if k%128==0 or k==len(pending):
                raw.flush();done.flush();receipt.flush();count=int(done.sum());elapsed=time.monotonic()-start
                atomic(OUT/f'STATUS_age{shift}.json',dict(timestamp=stamp(),done=count,total=len(keys),missing_this_pass=len(missing),elapsed_seconds=elapsed,eta_seconds=(len(keys)-count)*elapsed/max(1,count-initial)))
                print('age',shift,count,'/',len(keys),'missing',len(missing),flush=True)
    valid=np.isfinite(raw).all((1,2));np.save(OUT/f'{prefix}_valid.npy',valid)
    atomic(OUT/f'AGE{shift}_READY.json',dict(timestamp=stamp(),passed=True,frames=len(keys),missing_frames=int((~valid).sum()),minimum_lead=min(k[1] for k in keys),maximum_lead=max(k[1] for k in keys),missing_policy='same paired origin intersection across all ages; never use a newer fallback'))

def write_protocol():
    atomic(OUT/'PROTOCOL.json',dict(timestamp=stamp(),question='Does an additional12h of forecast age increase the value of observed power for weather correction?',
        issue_shifts_hours=[0,12,24],forecast_age_intervals_hours=[[12,24],[24,36],[36,48]],
        train_year=2022,validation_year=2023,validation_excludes_november=True,test_year_read=False,
        power_origins='identical hourly origins and paired common coverage for all ages; reference truth never shifted',
        meteorological_target='ERA5 minus Pangu 10m wind speed at1..6h after power origin',
        backbone='diagnostic multioutput Ridge alpha10; identical training algorithm at each age; no power forecasting network trained',
        input_safety='all NWP nodes from the older issue; real local history to origin and regional history delayed15min; ERA5 inputs only in separately named nondeployable oracle',
        comparison='weather-only MOS vs real local power vs regional power vs equivalent wind vs same-month/hour wrong-day controls',
        robustness='clip observed-power feature columns to2022 training min/max; same rule at all ages; report unbounded local sensitivity',
        inference='report both relative and absolute meteorological MSE reductions; synchronize calendar-week bootstrap across farms and forecast ages',
        limitations='older ERA5-initialized forecasts do not reproduce GFS initial-state error; negative linear results do not prove nonlinear impossibility',
        raw_source=str(SOURCE),era5_labels=str(parent.OUT/'station_surface.npy')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shift',type=int,choices=[12,24],required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    if not (OUT/'PROTOCOL.json').exists():write_protocol()
    extract(args.shift,args.workers)
