"""Paired 15-minute power samples; all weather routes use the same older issue."""
from settings import *
from operational import source_path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,shutil,time,argparse
BUNDLE=PROJECT/'data/operational_bundle_v1'
OLD=PROJECT/'runs/final_mechanism_20261030_v1/assets'
ERA=PROJECT/'runs/era5_privileged_diagnostic_20261101_v1/output'
SOURCE=Path('/hj102_data_large/ERA5_forecasting/Pangu/Forecast_data')
def save(path,x):
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,np.asarray(x))
def prepare(workers=8):
    (CACHE/'static').mkdir(parents=True,exist_ok=True)
    for p in (BUNDLE/'cache/static').glob('*.npy'):shutil.copy2(p,CACHE/'static'/p.name)
    shutil.copy2(BUNDLE/'physical/elevation.npy',DATA/'elevation.npy')
    mu=np.load(CACHE/'static/nwp_mean.npy');sd=np.load(CACHE/'static/nwp_std.npy')
    ids=np.load(CACHE/'static/bilinear_patch_index.npy');wt=np.load(CACHE/'static/bilinear_weight.npy')
    lat=np.load(CACHE/'static/weather_latitude.npy');lon=np.load(CACHE/'static/weather_longitude.npy')
    phi=np.deg2rad(np.load(CACHE/'static/station_latitude.npy'));f=2*7.2921159e-5*np.sin(phi)
    dy=np.deg2rad(float(lat[1]-lat[0]))*6371000;dx=np.deg2rad(float(lon[1]-lon[0]))*6371000
    point=lambda x:(x.reshape(*x.shape[:-2],-1)[...,ids]*wt).sum(-1)
    plans={};keys=set()
    for sp in ['train','val']:
        c=BUNDLE/'cache'/sp;orig=np.load(c/'origin_unix.npy');keep=np.ones(len(orig),bool)
        issue=np.load(c/'origin_issue_unix.npy')
        if sp=='val':
            keep &= np.load(OLD/'val_keep.npy')
            # Include the extra old-issue boundary in the common no-November cohort.
            for shift in [0,12]:
                for tz in [0,8*3600]:
                    month=(issue-shift*3600+tz).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
                    keep &= month!=11
        rows=np.flatnonzero(keep);cycles=np.load(c/'weather_cycle_index.npy')[rows]
        uc,first,inv=np.unique(cycles,return_index=True,return_inverse=True)
        future=np.load(c/'weather_valid_unix.npy')[uc];past=np.load(BUNDLE/'physical'/sp/'past_valid_unix.npy')[uc]
        it=issue[rows[first]]-12*3600
        assert np.all((orig[rows]-issue[rows])>=12*3600) and np.all((orig[rows]-issue[rows])<24*3600)
        for a,vs in zip(it,np.concatenate([past,future],1)):keys.update((int(a),int((v-a)//3600)) for v in vs)
        plans[sp]=dict(rows=rows,cycles=uc,inverse=inv,issue=it,past=past,future=future,origin=orig[rows])
        folder=DATA/'common'/sp
        for name in ['station_history','station_future','origin_phase','origin_unix','weather_time_offset_hours']:
            save(folder/(name+'.npy'),np.load(c/(name+'.npy'),mmap_mode='r')[rows])
        save(folder/'cycle.npy',inv);save(folder/'source_rows.npy',rows)
        save(folder/'source_cycles.npy',uc);save(folder/'issue0.npy',issue[rows])
        save(folder/'rings.npy',np.load(OLD/f'rings_{sp}.npy',mmap_mode='r')[rows])
        # Delayed, capacity-weighted regional histories; target farm always excluded.
        a=np.deg2rad(np.load(CACHE/'static/station_latitude.npy'));b=np.deg2rad(np.load(CACHE/'static/station_longitude.npy'))
        z=np.sin((a[:,None]-a)/2)**2+np.cos(a[:,None])*np.cos(a)*np.sin((b[:,None]-b)/2)**2
        dist=2*6371*np.arcsin(np.sqrt(z));np.fill_diagonal(dist,np.inf)
        cap=np.load(CACHE/'static/station_capacity_mw.npy');hist=np.load(folder/'station_history.npy',mmap_mode='r')
        regional=np.empty((len(rows),56,3,22),'f4');mask=[]
        for lo,hi in [(0,20),(20,60),(60,150)]:
            w=((dist>=lo)&(dist<hi))*cap[None];mask.append(w.sum(1)>0);w=w/w.sum(1,keepdims=True).clip(1e-9)
            regional[:,:,len(mask)-1]=np.einsum('ij,njt->nit',w.astype('f4'),hist[:,:,1:23])
        save(folder/'region_history.npy',regional);save(DATA/'region_mask.npy',np.array(mask).T)
        # Same-month and Beijing quarter-hour, different-day donors from training only.
        # Validation donor uses only2022, making the negative control deployable.
    train_origin=plans['train']['origin'];trlocal=(train_origin+8*3600).astype('datetime64[s]')
    trmonth=trlocal.astype('datetime64[M]').astype(int)%12+1;trclock=(train_origin+8*3600)%86400;trday=trlocal.astype('datetime64[D]').astype(int)
    for seed in SEEDS:
        for sp,d in plans.items():
            rng=np.random.default_rng(seed);o=d['origin'];local=(o+8*3600).astype('datetime64[s]')
            month=local.astype('datetime64[M]').astype(int)%12+1;clock=(o+8*3600)%86400;day=local.astype('datetime64[D]').astype(int)
            donors=np.empty(len(o),'i8')
            for i in range(len(o)):
                candidates=np.flatnonzero((trmonth==month[i])&(trclock==clock[i])&((np.abs(trday-day[i])>1) if sp=='train' else True))
                assert len(candidates);donors[i]=rng.choice(candidates)
            save(DATA/'common'/sp/f'donor_r{seed}.npy',donors)
    keys=sorted(keys);lookup={k:i for i,k in enumerate(keys)}
    save(DATA/'aged_keys.npy',np.array(keys,'i8'))
    framepath=DATA/'aged_frames.npy';spacepath=DATA/'aged_spatial.npy';donepath=DATA/'aged_done.npy'
    def mm(path,shape,dtype='f4'):
        if path.exists():return np.lib.format.open_memmap(path,mode='r+')
        v=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);v[:]=False if dtype=='?' else np.nan;v.flush();return v
    frames=mm(framepath,(len(keys),56,14));space=mm(spacepath,(len(keys),56,3));done=mm(donepath,(len(keys),),'?')
    def read(i):
        it,lead=keys[i];p=source_path(SOURCE,it,lead);before=p.stat();raw=np.load(p,mmap_mode='r',allow_pickle=False)
        assert raw.shape==(1,14,721,1440) and raw.dtype==np.float32
        crop=np.array(raw[0,:,171:195,483:508]);assert np.isfinite(crop).all()
        w=crop.astype('f8');normalized=point((crop-mu[:,None,None])/sd[:,None,None]).T.astype('f4')
        ug=-point(np.gradient(w[5],dy,axis=-2,edge_order=2))/f
        vg=point(np.gradient(w[5],dx,axis=-1,edge_order=2))/np.cos(phi)/f
        dty=point(np.gradient(w[9],dy,axis=-2,edge_order=2));dtx=point(np.gradient(w[9],dx,axis=-1,edge_order=2))/np.cos(phi)
        physical=point(w).T;adv=-(physical[:,11]*dtx+physical[:,13]*dty)*3600
        after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        return i,normalized,np.stack([ug/10,vg/10,adv],-1).astype('f4'),dict(path=str(p),size=before.st_size,mtime_ns=before.st_mtime_ns)
    pending=np.flatnonzero(~done);start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool,(DATA/'source_receipts.jsonl').open('a') as rec:
        for k,(i,x,s,receipt) in enumerate(pool.map(read,pending),1):
            frames[i]=x;space[i]=s;done[i]=True;rec.write(json.dumps(dict(index=int(i),**receipt))+'\n')
            if k%128==0 or k==len(pending):
                frames.flush();space.flush();done.flush();rec.flush()
                atomic_json(RUN/'PREPARATION_STATUS.json',dict(timestamp=now(),stage='older14channel_extraction',done=int(done.sum()),total=len(done),eta_seconds=(len(pending)-k)*(time.monotonic()-start)/k))
    assert done.all() and np.isfinite(frames).all()
    surface=np.load(ERA/'station_surface.npy',mmap_mode='r')
    for sp,d in plans.items():
        for age in [0,12]:
            folder=DATA/f'age{age}'/sp;folder.mkdir(parents=True,exist_ok=True)
            if age==0:
                future=np.load(BUNDLE/'local_weather'/f'{sp}.npy',mmap_mode='r')[d['cycles']]
                past=np.load(BUNDLE/'physical'/sp/'past_weather.npy',mmap_mode='r')[d['cycles']]
                dyn=np.load(BUNDLE/'physical'/sp/'dynamics.npy',mmap_mode='r')[d['cycles']]
            else:
                ix=lambda part:np.array([[lookup[(int(it),int((v-it)//3600))] for v in vs] for it,vs in zip(d['issue'],d[part])])
                fi=ix('future');pi=ix('past');future=frames[fi].transpose(0,2,1,3);past=frames[pi].transpose(0,2,1,3)
                spatial=space[fi].transpose(0,2,1,3);phys=future*sd+mu
                dyn=np.stack([spatial[...,0],spatial[...,1],phys[...,1]/10-spatial[...,0],phys[...,2]/10-spatial[...,1],spatial[...,2],
                    np.gradient(phys[...,0],axis=2)/100,np.gradient(np.hypot(phys[...,1],phys[...,2]),axis=2)],-1).astype('f4')
            for name,x in [('future',future),('past',past),('dynamics',dyn)]:save(folder/(name+'.npy'),x)
        idx=np.load(ERA/f'{sp}_future_index.npy')[d['cycles']]
        targets=surface[idx][:,:,:,1:3].transpose(0,2,1,3)
        save(DATA/'labels'/sp/'future_uv.npy',(targets-mu[1:3])/sd[1:3])
    audit=dict(timestamp=now(),passed=True,source_frames=len(keys),min_lead=min(k[1] for k in keys),max_lead=max(k[1] for k in keys),
        train_rows=len(plans['train']['rows']),val_rows=len(plans['val']['rows']),farms=56,test_read=False,
        same_samples_all_ages=True,train_only_statistics=True,era5_surface_final_only=True,
        forecast_ages=[[12,24],[24,36]],cadence_minutes=15,regional_delay_minutes=15)
    for sp in plans:
        assert np.isfinite(np.load(DATA/'labels'/sp/'future_uv.npy',mmap_mode='r')).all()
    atomic_json(DATA/'READY.json',audit);atomic_json(RUN/'PREPARATION_STATUS.json',dict(stage='complete',**audit))
    print(json.dumps(audit),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args();prepare(a.workers)
