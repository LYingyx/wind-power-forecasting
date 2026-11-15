"""Train-2022 topology and negative controls; no 2024 access or copied histories."""
from settings import *
from physics import neighbor_geometry
import numpy as np,collections

def main():
    setup();ASSETS.mkdir(exist_ok=True);static=CACHE/'static'
    lat=np.load(static/'station_latitude.npy');lon=np.load(static/'station_longitude.npy');cap=np.load(static/'station_capacity_mw.npy')
    assert np.isfinite(cap).all() and (cap>0).all()
    ids,geo,mask=neighbor_geometry(lat,lon,16,150.)
    allids,allgeo,allmask=neighbor_geometry(lat,lon,55,150.)
    h=np.load(CACHE/'train/station_history.npy',mmap_mode='r')
    power=np.asarray(h[:,:,-1],dtype='f8');assert np.isfinite(power).all()
    centered=power-power.mean(0);den=np.sqrt(np.square(centered).sum(0));corr=(centered.T@centered)/np.maximum(den[:,None]*den[None,:],1e-15)
    np.fill_diagonal(corr,-np.inf);corids=[];corgeo=[];cormask=[]
    for site in range(56):
        rank=sorted(range(55),key=lambda j:(not bool(allmask[site,j]),-corr[site,allids[site,j]],float(allgeo[site,j,2]),int(allids[site,j])))[:16]
        corids.append(allids[site,rank]);corgeo.append(allgeo[site,rank]);cormask.append(allmask[site,rank])
        assert np.array_equal(h[:64,site],np.load(OLD/f'prepared/train/{site:02d}/history.npy',mmap_mode='r')[:64])
    np.savez(ASSETS/'neighbor_tables.npz',distance_ids=ids,distance_geometry=geo,distance_mask=mask,corr_ids=np.asarray(corids),corr_geometry=np.asarray(corgeo),corr_mask=np.asarray(cormask),capacity_mw=cap,near50=(allgeo[:,:,2]<=50).any(1),correlation=corr)
    train_times=np.load(CACHE/'train/origin_unix.npy');train_local=train_times+8*3600
    trmonth=train_local.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
    trslot=(train_local%86400)//900;trday=train_local//86400
    pools={}
    for mo in range(1,13):
        for slot in range(96):pools[mo,slot]=np.flatnonzero((trmonth==mo)&(trslot==slot))
    checks={}
    for split in ['train','val']:
        times=np.load(CACHE/split/'origin_unix.npy');local=times+8*3600
        months=local.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1;slots=local%86400//900;days=local//86400
        for seed in SEEDS:
            rng=np.random.default_rng(20261025+seed);out=np.empty(len(times),dtype='i8')
            for i,(mo,slot,day) in enumerate(zip(months,slots,days)):
                pool=pools[int(mo),int(slot)]
                if split=='train':pool=pool[np.abs(trday[pool]-day)>=2]
                assert len(pool)>0,(split,i,mo,slot)
                out[i]=pool[rng.integers(len(pool))]
            assert np.array_equal(trmonth[out],months) and np.array_equal(trslot[out],slots)
            assert np.all(abs(trday[out]-days)>=2)
            if split=='val':assert np.all(train_times[out]<times)
            np.save(ASSETS/f'mismatch_{split}_r{seed}.npy',out)
            checks[f'{split}_{seed}']=dict(n=len(out),minimum_days_apart=int(abs(trday[out]-days).min()),donor_year=2022,validation_donors_always_past=bool(np.all(train_times[out]<times)) if split=='val' else None)
    bins=json.loads((ATTRIBUTION_RUN/'analysis/train_only_bins.json').read_text())
    for site in range(56):
        history=np.load(OLD/f'prepared/train/{site:02d}/history.npy',mmap_mode='r');target=np.load(OLD/f'prepared/train/{site:02d}/target.npy',mmap_mode='r')
        delta=target-np.concatenate((history[:,-4:],target[:,:20]),axis=1)
        bins['bins'][str(site)]['outcome_power_change_1h']=float(np.quantile(abs(delta),.75))
    bins['note']='Existing train bins reused; added train-only realized ramp bins for outcome diagnostics, never model inputs.'
    atomic_json(RUN/'analysis/train_only_bins.json',bins)
    atomic_json(ASSETS/'READY.json',dict(timestamp=now(),training_year=2022,test_year_accessed=False,nearest_valid_counts=mask.sum(1).tolist(),has_neighbor_50km=(allgeo[:,:,2]<=50).any(1).tolist(),mismatch_checks=checks,training_surrogate_note='Training-only synthetic negative control may use other days later within 2022; validation draws exclusively from 2022 archive. Not an operational alternative; N2_SELF separately controls capacity.',files={p.name:sha(p) for p in ASSETS.glob('*') if p.suffix in ['.npz','.npy']}))
    print(json.dumps(dict(neighbor_min=int(mask.sum(1).min()),neighbor_max=int(mask.sum(1).max()),farms_within_50km=int((allgeo[:,:,2]<=50).any(1).sum()),mismatch_checked=True)))

if __name__=='__main__':main()
