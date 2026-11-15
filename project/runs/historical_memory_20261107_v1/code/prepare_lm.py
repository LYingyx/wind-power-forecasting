"""Hourly causal forecast/power archive. CPU only; no ERA5 observations read."""
from settings import *
from operational import select_issue,source_path,BAD_START,BAD_END,BAD_UTC_START,BAD_UTC_END
import numpy as np,h5py,csv,argparse,time,fcntl

def build(raw_root):
    out=RUN/'lm_archive';out.mkdir(exist_ok=True)
    lock=(out/'build.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cohort=json.loads((ASSETS/'COHORT.json').read_text());keep=cohort['original_indices']
    bundle=PROJECT/'data/operational_bundle_v1';static=bundle/'cache/static'
    mu=np.load(ASSETS/'nwp_mean.npy');sd=np.load(ASSETS/'nwp_std.npy')
    ids=np.load(static/'bilinear_patch_index.npy')[keep];weights=np.load(static/'bilinear_weight.npy')[keep]
    lat=np.load(static/'weather_latitude.npy');lon=np.load(static/'weather_longitude.npy')
    rr=np.rint((90-lat)/.25).astype(int);cc=np.rint(lon/.25).astype(int)
    assert np.all(np.diff(rr)==1) and np.all(np.diff(cc)==1)
    start=int(np.datetime64('2022-01-01T00:00:00','s').astype(int))-28800
    end=int(np.datetime64('2025-01-01T00:00:00','s').astype(int))-28800
    times=np.arange(start,end,3600,dtype='i8');issues=select_issue(times);leads=(times-issues)//3600
    assert np.all(issues+43200<=times) and np.all((leads>=12)&(leads<24))
    # Mask bad valid times AND issue files in either calendar convention.
    bad=((times>=BAD_START)&(times<BAD_END))|((times>=BAD_UTC_START)&(times<BAD_UTC_END))|((issues>=BAD_START)&(issues<BAD_END))|((issues>=BAD_UTC_START)&(issues<BAD_UTC_END))
    weather=np.full((len(times),55,14),np.nan,'f4');power=np.full((len(times),55),np.nan,'f4');source=np.zeros(len(times),'i1')
    fi=np.load(bundle/'frame_issue_unix.npy');fl=np.load(bundle/'frame_lead_hours.npy');done=np.load(bundle/'raw_done.npy')
    lookup={(int(a),int(b)):i for i,(a,b) in enumerate(zip(fi,fl)) if done[i]}
    raw=np.load(bundle/'raw_frames.npy',mmap_mode='r');missing=[];receipts=[]
    def sample(crop):
        assert crop.shape==(14,len(lat),len(lon)) and np.isfinite(crop).all()
        normalized=(crop-mu[:,None,None])/sd[:,None,None]
        return (normalized.reshape(14,-1)[:,ids]*weights[None]).sum(-1).T.astype('f4')
    begin=time.monotonic()
    for i,(t,it,lead) in enumerate(zip(times,issues,leads)):
        if bad[i]:continue
        key=(int(it),int(lead))
        if key in lookup:weather[i]=sample(np.asarray(raw[lookup[key]]));source[i]=1
        else:
            path=source_path(raw_root,int(it),int(lead))
            if path.exists():
                before=path.stat();a=np.load(path,mmap_mode='r',allow_pickle=False)
                assert a.shape==(1,14,721,1440) and a.dtype==np.float32
                crop=np.array(a[0,:,rr[0]:rr[-1]+1,cc[0]:cc[-1]+1]);weather[i]=sample(crop);source[i]=2
                after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                receipts.append(dict(time=int(t),path=str(path),mtime_ns=before.st_mtime_ns,size=before.st_size))
            else:missing.append(dict(time=int(t),issue=int(it),lead=int(lead),path=str(path)))
        if i%1000==0:atomic_json(out/'STATUS.json',dict(timestamp=now(),stage='hourly_forecast_sampling',completed=i,total=len(times),seconds=time.monotonic()-begin))
    with h5py.File(PROJECT/'data/jilin_56farm_pad8.h5','r') as h5:
        ht=h5['power/timestamp_unix'][:];index=np.searchsorted(ht,times)
        assert (index<len(ht)).all() and np.array_equal(ht[index],times)
        # This is archive construction, not model evaluation or normalization.
        # Excluded November2024 power is not read, even for memory.
        ix=np.flatnonzero(~bad);values=h5['power/station_mw'][index[ix]][:,keep]
        power[ix]=values/np.load(ASSETS/'station_capacity_mw.npy')
    valid=np.isfinite(weather).all(-1)&np.isfinite(power)&(~bad[:,None])
    years=(times+28800).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    speed=np.hypot(weather[...,10]*sd[10]+mu[10],weather[...,12]*sd[12]+mu[12])
    curve=np.full(power.shape,np.nan,'f4');edges=[];means=[]
    for s in range(55):
        ok=valid[:,s]&(years==2022);e=np.quantile(speed[ok,s],np.linspace(0,1,31));binid=np.clip(np.searchsorted(e,speed[ok,s])-1,0,29)
        v=power[ok,s];m=np.array([v[binid==j].mean() if np.any(binid==j) else v.mean() for j in range(30)])
        use=np.flatnonzero(valid[:,s]);curve[use,s]=m[np.clip(np.searchsorted(e,speed[use,s])-1,0,29)]
        edges.append(e);means.append(m)
    for name,a in dict(time=times,issue=issues,lead_hours=leads,weather=weather,power=power,residual=power-curve,mapped_power=curve,valid=valid,bad_period=bad,source=source).items():np.save(out/f'{name}.npy',a)
    np.savez(out/'curve_2022.npz',edges=np.array(edges),means=np.array(means))
    atomic_json(out/'MISSING_SOURCES.json',dict(count=len(missing),raw_root=str(raw_root),raw_root_accessible=raw_root.exists(),missing=missing))
    atomic_json(out/'RAW_RECEIPTS.json',receipts)
    # Exact same-time sampled source values must reproduce old cache nodes.
    audits=[]
    for sp in ['train','val']:
        origin=np.load(DATA/sp/'origin.npy');cy=np.load(DATA/sp/'cycle.npy');offset=np.load(DATA/sp/'offset.npy');nodes=np.load(DATA/sp/'nodes.npy',mmap_mode='r')
        rows=np.flatnonzero(origin%3600==0);rows=rows[np.linspace(0,len(rows)-1,min(300,len(rows)),dtype=int)]
        assert np.all(offset[rows,0]==0)
        ii=np.searchsorted(times,origin[rows]);assert np.array_equal(times[ii],origin[rows])
        usable=valid[ii].all(1);err=float(np.max(abs(weather[ii[usable]]-nodes[cy[rows[usable]],:,0,:])))
        assert err<3e-5,(sp,err);audits.append(dict(split=sp,samples=int(usable.sum()),max_difference=err))
    coverage={}
    for a,b,label in [('2022-12-16','2023-01-04','requested_year_boundary'),('2023-11-01','2023-12-01','november2023'),('2024-11-01','2024-12-01','november2024')]:
        lo=int(np.datetime64(a,'s').astype(int))-28800;hi=int(np.datetime64(b,'s').astype(int))-28800;sel=(times>=lo)&(times<hi)
        coverage[label]=dict(hours=int(sel.sum()),weather_hours=int(np.isfinite(weather[sel]).all((1,2)).sum()),fully_paired_hours=int(valid[sel].all(1).sum()),masked_hours=int(bad[sel].sum()))
    audit=dict(timestamp=now(),forecast_archive_complete=len(missing)==0,training_ready=len(missing)==0,total_hours=len(times),cached_hours=int((source==1).sum()),raw_hours=int((source==2).sum()),missing_hours=len(missing),masked_hours=int(bad.sum()),
        per_year={str(y):dict(total_hours=int((years==y).sum()),valid_pairs=int(valid[years==y].sum()),possible_pairs=int((years==y).sum()*55)) for y in [2022,2023,2024]},
        coverage=coverage,cache_equivalence=audits,power_read_years=[2022,2023,2024],evaluation_performed=False,era5_observations_read=False,curve_fit_year=2022,
        coordinates=dict(latitude=[float(lat.min()),float(lat.max())],longitude=[float(lon.min()),float(lon.max())]),
        convention='UTC Unix timestamps; local power UTC+8; available=issue+12h; bad target and issue November2024 masked',
        invalid_policy='explicit masks; no substitution with ERA5, newer issues, or another forecast source')
    atomic_json(out/'AUDIT.json',audit);atomic_json(out/'STATUS.json',dict(stage='complete' if not missing else 'awaiting_missing_raw_source',**audit))
    (out/'README_CN.md').write_text(f"# LM 小时配对档案\n\n已有缓存覆盖 {audit['cached_hours']} 小时；缺少原始预报 {len(missing)} 小时。缺测均掩码，不以 ERA5 或其他期次填充。\n\n{json.dumps(coverage,ensure_ascii=False,indent=2)}\n\n仅数据准备，没有运行 2024 模型评价。原始源补齐前 training_ready=false。\n")
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw-root',type=Path,default=Path('/hj102_data_large/ERA5_forecasting/Pangu/Forecast_data'));args=ap.parse_args();build(args.raw_root)
