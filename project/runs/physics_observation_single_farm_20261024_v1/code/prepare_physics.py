"""Compact, reusable cycle-level additions. Never duplicate existing weather grids."""
from settings import *
from physics import *
import argparse
import h5py


def save(path, array):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f: np.save(f,array)
    os.replace(tmp,path)


def point_indices(lat,lon,slat,slon):
    # Order-independent bilinear interpolation on rectangular grid.
    ids=[]; weights=[]
    for y,x in zip(slat,slon):
        yy=np.argsort(abs(lat-y))[:2];xx=np.argsort(abs(lon-x))[:2]
        yy=yy[np.argsort(lat[yy])];xx=xx[np.argsort(lon[xx])]
        a=(y-lat[yy[0]])/(lat[yy[1]]-lat[yy[0]])
        b=(x-lon[xx[0]])/(lon[xx[1]]-lon[xx[0]])
        assert 0<=a<=1 and 0<=b<=1
        ids.append([i*len(lon)+j for i in yy for j in xx])
        weights.append([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
    return np.array(ids),np.array(weights,dtype='f4')


def extract_points(f,frames,ids,weights):
    unique=np.unique(frames[frames>=0]); points=np.empty((len(unique),56,14),dtype='f4')
    for start in range(0,len(unique),256):
        raw=f['nwp/values'][unique[start:start+256]].reshape(-1,14,29*30)
        points[start:start+len(raw)]=(raw[:,:,ids]*weights[None,None]).sum(-1).transpose(0,2,1)
    return unique,points


def prepare(split):
    setup(); out=PREPARED/split;out.mkdir(exist_ok=True)
    if (out/'READY.json').exists():return
    assert split in ['train','val'], 'Test preparation requires separate frozen evaluation'
    static=CACHE/'static';slat=np.load(static/'station_latitude.npy');slon=np.load(static/'station_longitude.npy')
    mean=np.load(static/'nwp_mean.npy');std=np.load(static/'nwp_std.npy')
    ci=np.load(CACHE/split/'weather_cycle_index.npy');orig=np.load(CACHE/split/'origin_unix.npy')
    with h5py.File(H5,'r') as f:
        g=f['stage1_splits/'+split]
        frames=g['nwp_frame_index'][:];valid=g['nwp_valid_unix'][:];issue=g['nwp_issue_unix'][:]
        assert len(frames)==len(np.load(LOCAL_WEATHER/(split+'.npy'),mmap_mode='r'))
        cached_valid=np.load(CACHE/split/'weather_valid_unix.npy')
        assert np.array_equal(valid[:,6:14],cached_valid)
        assert np.all(issue[ci]+900<=orig[:,None]),'NWP publication latency violated'
        ids,weights=point_indices(f['crop/latitude'][:],f['crop/longitude'][:],slat,slon)
        # Earlier run, SAME target valid times. Missing packed frames remain masked.
        runs=g['nwp_run_index'][:,6];earlier=np.maximum(runs-1,0)
        releases=f['nwp/issue_unix'][:];lookup=f['nwp/run_lead_to_frame'][:]
        lead=((valid[:,6:14]-releases[earlier,None])//3600-1).astype(int)
        available=(runs>0)&(releases[earlier]+900<=np.minimum.reduceat(orig,np.flatnonzero(np.r_[True,np.diff(ci)!=0])))
        mask=(lead>=0)&(lead<lookup.shape[1])&available[:,None]
        alt=lookup[earlier[:,None],np.clip(lead,0,lookup.shape[1]-1)];alt=np.where(mask,alt,-1);mask=alt>=0
        unique,points=extract_points(f,np.concatenate([frames[:,:6].ravel(),alt.ravel(),frames[:1,6:14].ravel()]),ids,weights)
        history=points[np.searchsorted(unique,frames[:,:6])].transpose(0,2,1,3)
        save(out/'past_weather.npy',history)
        save(out/'past_valid_unix.npy',valid[:,:6]);save(out/'past_issue_unix.npy',issue[:,:6])
        save(out/'future_issue_unix.npy',issue[:,6:14]);save(out/'overlap_mask.npy',mask)
        future=np.load(LOCAL_WEATHER/(split+'.npy'),mmap_mode='r')
        replay=points[np.searchsorted(unique,frames[0,6:14])].transpose(1,0,2)
        difference=float(np.abs(replay-future[0]).max());assert difference<2e-4, difference
        if mask.any():
            alternate=points[np.searchsorted(unique,np.maximum(alt,unique[0]))].transpose(0,2,1,3)
            disagreement=np.where(mask[:,None,:,None],future-alternate,0)
            save(out/'disagreement.npy',disagreement.astype('f4'))
        audit=dict(split=split,n_samples=len(orig),n_cycles=len(frames),historical_nodes=6,
            future_nodes=8,all_issue_plus_15min_before_origin=True,historical_future_issue_same=False,
            point_replay_max_abs=difference,overlap_node_fraction=float(mask.mean()),
            older_run_gap_hours_min=float(((issue[:,6]-releases[earlier])/3600).min()),
            time_reference='UTC Unix; calendar converted to UTC+8',finished=now())
    # Differential features use the frozen 7x7 physical patch; only derived values saved.
    bw=np.load(PREVIOUS/'prepared/bilinear_weights.npy')
    lat=np.load(static/'weather_latitude.npy');lon=np.load(static/'weather_longitude.npy')
    dyn=np.lib.format.open_memmap(out/'dynamics.tmp.npy',mode='w+',dtype='f4',shape=(len(frames),56,8,7))
    for site in range(56):
        patch=np.load(PREVIOUS/f'prepared/{split}/site{site:02d}.npy',mmap_mode='r')
        for start in range(0,len(frames),128):
            raw=np.array(patch[start:start+128]).reshape(-1,8,7,7,14)*std+mean
            dyn[start:start+len(raw),site]=dynamic_fields(raw,slat[site],float(lat[1]-lat[0]),float(lon[1]-lon[0]),bw[site])
        if site%8==0:print(split,'dynamics site',site,flush=True)
    dyn.flush();del dyn;os.replace(out/'dynamics.tmp.npy',out/'dynamics.npy')
    atomic_json(out/'READY.json',audit);print(json.dumps(audit),flush=True)


def main():
    setup();static=CACHE/'static'
    ids,geo,mask=neighbor_geometry(np.load(static/'station_latitude.npy'),np.load(static/'station_longitude.npy'))
    save(PREPARED/'neighbor_ids.npy',ids);save(PREPARED/'neighbor_geometry.npy',geo);save(PREPARED/'neighbor_mask.npy',mask)
    dem=json.loads((COLUMN_DATA/'station_dem_srtm30m.json').read_text())['records']
    save(PREPARED/'elevation.npy',np.array([r['terrain_elevation_m'] for r in dem],dtype='f4'))
    for split in ['train','val']:prepare(split)

if __name__=='__main__':main()
