"""Power-only frozen QC, retained-neighbor recomputation, train-only statistics."""
from settings import *
import numpy as np,h5py,csv,shutil
from physics import thermodynamics,operating_state,neighbor_geometry
from operational import interpolate_nodes,select_issue

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);np.save(p,np.asarray(x))

def column(p,e):
    e=np.full(p.shape[:-1],e,dtype='f4')
    return np.stack([p[...,0],p[...,1],p[...,2],p[...,3],e,p[...,10]-p[...,1],p[...,12]-p[...,2],p[...,8],p[...,6],p[...,4]/9.80665-e,p[...,11]-p[...,10],p[...,13]-p[...,12],p[...,9]*(1000/925)**.2854-p[...,8],p[...,7]-p[...,6],(p[...,5]-p[...,4])/9.80665],-1)

def qc():
    rules=json.loads((RUN/'QC_RULES.json').read_text());rows=[]
    with h5py.File(PROJECT/'data/jilin_56farm_pad8.h5','r') as f:
        times=f['power/timestamp_unix'][:]
        years=(times+28800).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
        # Slice before reading, never load 2024 power values.
        ii=np.flatnonzero(np.isin(years,rules['cohort_years']))
        raw=f['power/station_mw'][ii]
        cap=f['stations/capacity_mw'][:]
        ids=f['stations/id'].asstr()[:];names=f['stations/name'].asstr()[:]
        power=raw/cap
    for s in range(len(ids)):
        reasons=[];stats={}
        for y in rules['cohort_years']:
            v=power[years[ii]==y,s];finite=np.isfinite(v);z=v[finite]
            q=float(np.quantile(z,.99)) if len(z) else 0.
            constant=np.zeros(len(v),bool)
            breaks=np.flatnonzero(np.r_[True,~(np.isfinite(v[1:])&np.isfinite(v[:-1])&(np.abs(np.diff(v))<=rules['constant_tolerance_capacity_factor'])),True])
            for a,b in zip(breaks[:-1],breaks[1:]):
                if b-a>=rules['constant_minimum_run_steps'] and rules['constant_nonzero_range'][0]<v[a]<rules['constant_nonzero_range'][1]:constant[a:b]=True
            gross=float(((z<rules['gross_out_of_range'][0])|(z>rules['gross_out_of_range'][1])).mean()) if len(z) else 1.
            d=dict(finite_fraction=float(finite.mean()),q99=q,gross_fraction=gross,constant_nonzero_fraction=float(constant.sum()/max(1,finite.sum())))
            stats[str(y)]=d
            if d['finite_fraction']<rules['minimum_finite_fraction_each_year']:reasons.append(f'{y}:missing')
            if not rules['minimum_q99_capacity_factor_each_year']<=q<=rules['maximum_q99_capacity_factor_each_year']:reasons.append(f'{y}:q99')
            if gross>rules['maximum_gross_out_of_range_fraction']:reasons.append(f'{y}:gross_range')
            if d['constant_nonzero_fraction']>rules['maximum_nonzero_constant_run_fraction']:reasons.append(f'{y}:constant_meter')
        q=[d['q99'] for d in stats.values()];shift=rules['q99_shift_exclude_if_both']
        if max(q)-min(q)>shift['absolute_change_above'] and max(q)/max(min(q),1e-9)>shift['max_min_ratio_above']:reasons.append('q99_cross_year_shift')
        rows.append(dict(original_site=s,station_id=str(ids[s]),name=str(names[s]),keep=not reasons,reasons=reasons,years=stats))
    keep=[r['original_site'] for r in rows if r['keep']]
    audit=dict(timestamp=now(),rules_sha256=sha(RUN/'QC_RULES.json'),retained=keep,removed=[r for r in rows if not r['keep']],records=rows,read_years=[2022,2023],model_errors_read=False)
    atomic_json(ASSETS/'QC_AUDIT.json',audit)
    atomic_json(ASSETS/'COHORT.json',dict(original_indices=keep,station_ids=[rows[s]['station_id'] for s in keep],names=[rows[s]['name'] for s in keep],count=len(keep),qc_rules_sha256=audit['rules_sha256']))
    with (ASSETS/'qc_by_farm.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['id','name','keep','reasons','q99_2022','q99_2023','coverage2022','coverage2023','constant2022','constant2023'])
        for r in rows:w.writerow([r['station_id'],r['name'],r['keep'],';'.join(r['reasons']),*[r['years'][str(y)][k] for k in ['q99','finite_fraction','constant_nonzero_fraction'] for y in [2022,2023]]])
    print('QC',len(keep),'retained',[(r['name'],r['reasons']) for r in audit['removed']],flush=True)
    return np.asarray(keep)

def prepare():
    protocol=json.loads((SOURCE/'PROTOCOL.json').read_text())
    fingerprint=protocol['preprocess_fingerprint']
    assert protocol['protocol']=='daily_08_BJT_latency15h'
    assert protocol['regional_power_delay_minutes']==0
    for name in ['SUCCESS.json','INDEPENDENT_AUDIT.json']:
        marker=json.loads((SOURCE/name).read_text())
        assert marker['preprocess_fingerprint']==fingerprint
        if name=='INDEPENDENT_AUDIT.json':assert marker['passed']
    (DATA/'READY.json').unlink(missing_ok=True)
    keep=qc();static=SOURCE/'cache/static'
    frozen=json.loads((RUN/'selection_inputs/COHORT.json').read_text())
    assert keep.tolist()==frozen['original_indices'], 'Preserve the existing power-QC cohort'
    assert json.loads((ASSETS/'COHORT.json').read_text())==frozen
    for name in ['station_id','station_name','station_latitude','station_longitude','station_capacity_mw']:
        save(ASSETS/(name+'.npy'),np.load(static/(name+'.npy'))[keep])
    for name in ['nwp_mean','nwp_std','nwp_channel_name']:shutil.copy2(static/(name+'.npy'),ASSETS/(name+'.npy'))
    elevation=np.load(SOURCE/'physical/elevation.npy')[keep];save(ASSETS/'elevation.npy',elevation)
    lat=np.load(ASSETS/'station_latitude.npy');lon=np.load(ASSETS/'station_longitude.npy');cap=np.load(ASSETS/'station_capacity_mw.npy')
    ids,geo,mask=neighbor_geometry(lat,lon)
    a=np.deg2rad(lat);b=np.deg2rad(lon);z=np.sin((a[:,None]-a)/2)**2+np.cos(a[:,None])*np.cos(a)*np.sin((b[:,None]-b)/2)**2
    distance=2*6371*np.arcsin(np.sqrt(z));np.fill_diagonal(distance,np.inf)
    ring_weights=np.asarray([((distance>=lo)&(distance<hi))*cap for lo,hi in [(0,20),(20,60),(60,150.0001)]],dtype='f4')
    ring_valid=ring_weights.sum(-1)>0;ring_weights/=ring_weights.sum(-1,keepdims=True).clip(1e-9)
    near_weights=np.zeros((len(keep),len(keep)),'f4')
    for s in range(len(keep)):near_weights[s,ids[s]]=mask[s]*cap[ids[s]]
    near_weights/=near_weights.sum(-1,keepdims=True).clip(1e-9)
    np.savez(ASSETS/'topology.npz',ids=ids,geometry=geo,mask=mask,capacity=cap[ids],ring_weights=ring_weights,ring_valid=ring_valid,distance=distance,near_weights=near_weights)
    desc=np.stack([lat,lon,np.log(cap),elevation],-1).astype('f4');mean=desc.mean(0);std=desc.std(0).clip(1e-4)
    save(ASSETS/'site_static.npy',(desc-mean)/std)
    stats={};mu=np.load(ASSETS/'nwp_mean.npy');sd=np.load(ASSETS/'nwp_std.npy')
    colsum=np.zeros(15);colsq=np.zeros(15);colcount=0
    for sp in ['train','val']:
        c=SOURCE/'cache'/sp;out=DATA/sp;out.mkdir(exist_ok=True)
        origin=np.load(RUN/f'selection_inputs/{sp}_origin.npy')
        all_origins=np.load(c/'origin_unix.npy');rows=np.searchsorted(all_origins,origin)
        assert (rows<len(all_origins)).all()
        np.testing.assert_array_equal(all_origins[rows],origin)
        hist=np.load(c/'station_history.npy',mmap_mode='r')[rows][:,keep].copy();y=np.load(c/'station_future.npy',mmap_mode='r')[rows][:,keep].copy()
        issue=np.load(c/'origin_issue_unix.npy')[rows];phase=np.load(c/'origin_phase.npy')[rows]
        source_cycle=np.load(c/'weather_cycle_index.npy')[rows];cycles,cyc=np.unique(source_cycle,return_inverse=True)
        offset=np.load(c/'weather_time_offset_hours.npy')[rows]
        assert np.array_equal(select_issue(origin),issue)
        assert np.all((origin-issue>=54000)&(origin-issue<140400))
        assert np.all(issue%86400==0)
        assert np.all((origin+28800).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970==(2022 if sp=='train' else 2023))
        w=np.load(SOURCE/f'local_weather/{sp}.npy',mmap_mode='r')[cycles][:,keep].copy();dyn=np.load(SOURCE/f'physical/{sp}/dynamics.npy',mmap_mode='r')[cycles][:,keep].copy()
        near=np.einsum('ij,njt->nit',near_weights,hist[:,:,2:24]);features=[]
        for r in range(3):
            weight=ring_weights[r];avg=np.einsum('ij,njt->nit',weight,hist[:,:,2:24]);second=np.einsum('ij,nj->ni',weight,hist[:,:,23]**2)
            sig=np.sqrt(np.maximum(second-avg[:,:,-1]**2,0));features.append(np.stack([avg[:,:,-1],sig,avg[:,:,-1]-avg[:,:,-5],np.broadcast_to(ring_valid[r],avg.shape[:2])],-1))
        rings=np.concatenate(features,-1)
        for name,x in [('h',hist),('y',y),('origin',origin),('phase',phase),('cycle',cyc),('offset',offset),('issue',issue),('nodes',w),('dynamics',dyn),('near',near),('rings',rings),('source_row',rows),('source_cycle',cycles)]:save(out/(name+'.npy'),x)
        if sp=='train':
            for s in range(len(keep)):
                wi=interpolate_nodes(w[cyc,s],offset,np.arange(1,25)/4);di=interpolate_nodes(dyn[cyc,s],offset,np.arange(1,25)/4)
                phys=wi*sd+mu;den,stab=thermodynamics(phys,float(elevation[s]));flat=float(np.quantile(abs(np.diff(hist[:,s],axis=-1)),.1));state=operating_state(hist[:,s],flat)
                raw=np.concatenate([den,di,stab,np.broadcast_to(state[:,None],(len(hist),24,7))],-1)
                rr=rings[:,s];stats[str(s)]=dict(flat_threshold=flat,sel_mean=raw.mean((0,1),dtype='f8').tolist(),sel_std=raw.std((0,1),dtype='f8').clip(1e-4).tolist(),ring_mean=rr.mean(0,dtype='f8').tolist(),ring_std=rr.std(0,dtype='f8').clip(1e-4).tolist())
                col=column(w[:,s]*sd+mu,float(elevation[s])).reshape(-1,15).astype('f8');colsum+=col.sum(0);colsq+=(col*col).sum(0);colcount+=len(col)
            cm=colsum/colcount;cs=np.sqrt(np.maximum(colsq/colcount-cm**2,1e-8))
            atomic_json(ASSETS/'normalization.json',dict(fitted_year=2022,sites=stats,column_mean=cm.tolist(),column_std=cs.tolist(),static_mean=mean.tolist(),static_std=std.tolist()))
        # Same-month and local clock donors are fixed for both training and evaluation.
        tr=np.load(DATA/'train/origin.npy');trm=(tr+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12
        trclock=(tr+28800)%86400;trday=(tr+28800)//86400;m=(origin+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12
        for seed in SEEDS:
            rng=np.random.default_rng(seed+20261104);donor=np.empty(len(origin),'i8')
            for month in np.unique(m):
                for slot in np.unique((origin+28800)%86400):
                    ix=np.flatnonzero((m==month)&((origin+28800)%86400==slot));candidates=np.flatnonzero((trm==month)&(trclock==slot))
                    for i in ix:
                        valid=candidates[np.abs(trday[candidates]-(origin[i]+28800)//86400)>=2];assert len(valid)>0;donor[i]=rng.choice(valid)
            save(out/f'donor_r{seed}.npy',donor)
        print('prepared',sp,len(origin),len(keep),flush=True)
    # Quarter-block OOF; 12h purge covers 6h history + 6h future overlap.
    t=np.load(DATA/'train/origin.npy');m=(t+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12;fold=m//3
    save(ASSETS/'oof_fold.npy',fold)
    for f in range(4):
        held=np.flatnonzero(fold==f);lower=t[held].min()-43200;upper=t[held].max()+43200
        allowed=(t<lower)|(t>upper)
        # Internal validation is a 21-day block within allowed months, disjoint from OOF target block.
        end=t[allowed].max();iv=allowed&(t>=end-21*86400);valid=np.flatnonzero(iv)
        train=np.flatnonzero(allowed&((t<t[valid].min()-43200)|(t>t[valid].max()+43200)))
        assert len(train)>0 and len(valid)>0 and not np.intersect1d(train,held).size and not np.intersect1d(valid,held).size
        np.savez(ASSETS/f'fold{f}.npz',train=train,valid=valid,heldout=held)
    from prepare_rm import main as prepare_regional_history
    prepare_regional_history()
    files={str(p.relative_to(RUN)):sha(p) for root in [DATA,ASSETS] for p in root.rglob('*') if p.is_file() and p.name!='READY.json'}
    atomic_json(RUN/'DATA_FREEZE.json',dict(timestamp=now(),files=files,qc_rules_sha256=sha(RUN/'QC_RULES.json'),preprocess_fingerprint=fingerprint))
    atomic_json(DATA/'READY.json',dict(timestamp=now(),passed=True,farms=len(keep),train_rows=len(np.load(DATA/'train/origin.npy')),val_rows=len(np.load(DATA/'val/origin.npy')),read_2024=False,source=str(SOURCE),regional_delay_minutes=0,regional_recomputed=True,protocol=protocol['protocol'],preprocess_fingerprint=fingerprint))

if __name__=='__main__':prepare()
