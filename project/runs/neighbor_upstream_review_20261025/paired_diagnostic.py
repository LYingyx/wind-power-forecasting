"""Supplementary OLS diagnostic on the SAME 2022/2023 origins as the neural screen.

Reads only train/val caches. All directed pairs, common rows for nested models,
as-of-origin linearly interpolated NWP. Midpoint wind is approximated by the
mean of the two interpolated endpoint vectors: this is a revised diagnostic,
not an exact reproduction of the old nearest-grid midpoint calculation.
"""
from pathlib import Path
import json, csv, os, time
import numpy as np

P=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
OUT=Path(__file__).resolve().parent
CACHE=P/'data/model_cache_pad5_f8'
OLD=P/'runs/independent_single_farm_20261021_v1'
PREP=P/'runs/physics_observation_single_farm_20261024_v1/prepared'
LOCAL=P/'artifacts/shared_all_station_baselines_20260922_v1/data/local_weather'

def load(split):
    assert split in ['train','val']
    root=CACHE/split
    t=np.load(root/'origin_unix.npy')
    h=np.load(root/'station_history.npy',mmap_mode='r')
    ci=np.load(root/'weather_cycle_index.npy')
    future_time=np.load(root/'weather_time_offset_hours.npy')
    past_time=(np.load(PREP/split/'past_valid_unix.npy')[ci]-t[:,None])/3600
    nodes=np.load(LOCAL/f'{split}.npy',mmap_mode='r')
    past=np.load(PREP/split/'past_weather.npy',mmap_mode='r')
    mean=np.load(CACHE/'static/nwp_mean.npy')[[1,2]]
    std=np.load(CACHE/'static/nwp_std.npy')[[1,2]]
    times=np.concatenate((past_time,future_time),axis=1)
    uv=np.empty((len(t),56,4,2),dtype='f8')
    # 0h is interpolated through the archived past nodes and the future nodes;
    # future 1/2/3h use the same as-of-origin product as the neural models.
    allnodes=np.concatenate((np.asarray(past[:,:,:,[1,2]]),np.asarray(nodes[:,:,:,[1,2]])),axis=2)
    issue0=np.load(PREP/split/'past_issue_unix.npy')[ci]
    issue1=np.load(PREP/split/'future_issue_unix.npy')[ci]
    assert (issue0+900<=t[:,None]).all() and (issue1+900<=t[:,None]).all()
    for j,hour in enumerate([0.,1.,2.,3.]):
        x=np.clip(hour-times[:,0],0,times.shape[1]-1)
        lo=np.minimum(np.floor(x).astype(int),times.shape[1]-2)
        fraction=x-lo
        a=allnodes[ci,:,lo,:];b=allnodes[ci,:,lo+1,:]
        uv[:,:,j,:]=(a+(b-a)*fraction[:,None,None])*std+mean
    y=np.stack([np.load(OLD/f'prepared/{split}/{site:02d}/target.npy',mmap_mode='r')[:,[3,7,11]] for site in range(56)],axis=1).astype('f8')
    # Retain the declared dataset splits, never re-derive a UTC-year split.
    local_year=(t+8*3600).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    target_year=(t+6*3600+8*3600).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    assert np.all(local_year==(2022 if split=='train' else 2023))
    assert np.array_equal(local_year,target_year)
    assert np.isfinite(h).all() and np.isfinite(y).all() and np.isfinite(uv).all()
    return dict(t=t,h=np.asarray(h,dtype='f8'),y=y,uv=uv,speed=np.linalg.norm(uv,axis=-1))

def paired_fit(x0,x1,y,tx0,tx1,ty):
    mask=np.isfinite(x1).all(1)&np.isfinite(x0).all(1)&np.isfinite(y)
    test=np.isfinite(tx1).all(1)&np.isfinite(tx0).all(1)&np.isfinite(ty)
    if mask.sum()<500 or test.sum()<300:return None
    results=[]
    for x,xt in [(x0,tx0),(x1,tx1)]:
        a=np.column_stack((x[mask],np.ones(mask.sum())))
        b=np.column_stack((xt[test],np.ones(test.sum())))
        w=np.linalg.lstsq(a,y[mask],rcond=None)[0]
        e=ty[test]-b@w
        results.append(dict(mse=float(np.mean(e*e)),mae=float(np.mean(np.abs(e)))))
    return dict(n_train=int(mask.sum()),n_test=int(test.sum()),**{k+'_'+label:v for label,d in zip(['base','full'],results) for k,v in d.items()})

def main():
    os.nice(10);started=time.monotonic()
    train=load('train');val=load('val')
    lat=np.load(CACHE/'static/station_latitude.npy');lon=np.load(CACHE/'static/station_longitude.npy')
    rows=[];pairs=[]
    for a in range(56):
        for b in range(a+1,56):
            phi=np.radians([lat[a],lat[b]])
            dphi=phi[1]-phi[0];dlam=np.radians(lon[b]-lon[a])
            distance=2*6371*np.arcsin(np.sqrt(np.sin(dphi/2)**2+np.cos(phi).prod()*np.sin(dlam/2)**2))
            if not 20<=distance<=150:continue
            pairs.append((a,b,distance))
    assert len(pairs)==1064
    for pair_index,(a,b,distance) in enumerate(pairs):
        dx=(lon[b]-lon[a])*np.pi/180*6371*np.cos(np.radians((lat[a]+lat[b])/2))
        dy=(lat[b]-lat[a])*np.pi/180*6371
        masks=[]
        for d in [train,val]:
            wind=(d['uv'][:,a,0]+d['uv'][:,b,0])/2
            speed=np.linalg.norm(wind,axis=-1)
            cosine=(wind[:,0]*dx+wind[:,1]*dy)/(distance*np.maximum(speed,1e-6))
            masks.append(dict(up=(speed>=4)&(cosine>=.7),down=(speed>=4)&(cosine<=-.7),cross=(speed>=4)&(abs(cosine)<=.3)))
        for source,target,reverse in [(a,b,False),(b,a,True)]:
            for hi in range(3):
                arrays=[]
                for d in [train,val]:
                    h=d['h'];current=h[:,target,-1];sp0=d['speed'][:,target,0];sp1=d['speed'][:,target,hi+1]
                    base=np.column_stack((current,current-h[:,target,-2],current-h[:,target,-5],sp0,sp1,sp1-sp0,(sp1-sp0)*(1-current)))
                    neigh=np.column_stack((h[:,source,-2],h[:,source,-2]-h[:,source,-6],h[:,source,-2]-h[:,source,-10]))
                    full=np.column_stack((base,neigh))
                    # Additional control: baseline gets ALL 24 own-history points.
                    strong=np.column_stack((h[:,target],sp0,sp1,(sp1-sp0)*(1-current)))
                    arrays.append(dict(sparse=(base,full),history24=(strong,np.column_stack((strong,neigh))),y=d['y'][:,target,hi]-current))
                for cls in ['up','down','cross']:
                    key=('down' if cls=='up' else 'up') if reverse and cls!='cross' else cls
                    m0,m1=[x[key] for x in masks]
                    if m0.sum()<500 or m1.sum()<300:continue
                    for baseline in ['sparse','history24']:
                        x0,x1=arrays[0][baseline];t0,t1=arrays[1][baseline]
                        result=paired_fit(x0[m0],x1[m0],arrays[0]['y'][m0],t0[m1],t1[m1],arrays[1]['y'][m1])
                        if result:rows.append(dict(source=source,target=target,distance_km=float(distance),cls=cls,horizon=f'{hi+1}h',baseline=baseline,**result))
        if pair_index%100==0:print('pairs',pair_index+1,'/',len(pairs),'rows',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
    with (OUT/'paired_predictability_corrected.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    status=dict(completed=True,seconds=time.monotonic()-started,unordered_pairs=len(pairs),directed_pairs=2*len(pairs),rows=len(rows),train_origins=len(train['t']),val_origins=len(val['t']),test_values_read=False,common_rows_between_models=True,year_boundary='existing train/val cache, local power timezone',neighbor_latency_minutes=15,midpoint_proxy='mean of endpoint forecast wind vectors, as-of-origin interpolation',baselines=['sparse same span as original OLS','history24 includes full 6h local history'],interpretation='Conditional OLS association; not a lower bound or causal test. No neural models fitted.')
    (OUT/'CORRECTED_DIAGNOSTIC_AUDIT.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)

if __name__=='__main__':main()
