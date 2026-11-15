"""Materialize canonical interpolation, once, without fitting on val/test."""
from common import *
import shutil
from wind_forcing.models.p0i_legacy import interpolate
from wind_forcing.models.components import build_station_patch_metadata

def main():
    configure_torch(); s = static(); out = RUN/'prepared'; out.mkdir(exist_ok=True)
    audit = {'splits':{}, 'weather_nodes_used':7, 'method':'canonical bilinear then clamped time interpolation',
             'normalization':'frozen source 2022 statistics', 'source_hashes':{}}
    ids = list(map(str,s['station_id']))
    for split in ['train','val','test']:
        dest = out/split; dest.mkdir(exist_ok=True)
        h = np.load(CACHE/split/'station_history.npy',mmap_mode='r')
        y = np.load(CACHE/split/'station_future.npy',mmap_mode='r')
        cyc = np.load(CACHE/split/'weather_cycle_index.npy')
        times = np.load(CACHE/split/'weather_time_offset_hours.npy')
        local = np.load(LOCAL/f'{split}.npy',mmap_mode='r')
        assert h.shape == y.shape == (len(h),56,24)
        assert np.all(np.diff(times,axis=1)>0)
        assert cyc.min()>=0 and cyc.max()<len(local)
        for name in ['origin_phase','origin_unix','sample_index','weather_time_offset_hours','weather_cycle_index']:
            shutil.copy2(CACHE/split/f'{name}.npy',dest/f'{name}.npy')
        # Independently verify the old local cache against original spatial grids.
        ix,_,wt=build_station_patch_metadata(s['weather_latitude'],s['weather_longitude'],s['station_latitude'],s['station_longitude'],'bilinear',1)
        grid=np.load(CACHE/split/'weather.npy',mmap_mode='r')
        sample=np.array([0,len(grid)//2,len(grid)-1])
        values=np.asarray(grid[sample]).reshape(3,8,14,-1)[...,ix.flatten()].reshape(3,8,14,56,-1)
        reproduced=(values*wt[None,None,None]).sum(-1).transpose(0,3,1,2)
        spatial_error=float(np.max(np.abs(reproduced-local[sample])))
        assert spatial_error<1e-5, spatial_error
        for site in range(56):
            sd=dest/f'{site:02d}';sd.mkdir(exist_ok=True)
            if (sd/'SUCCESS.json').exists(): continue
            hh=np.ascontiguousarray(h[:,site]); yy=np.ascontiguousarray(y[:,site])
            assert np.isfinite(hh).all() and np.isfinite(yy).all()
            np.save(sd/'history.npy',hh);np.save(sd/'target.npy',yy)
            arr=np.lib.format.open_memmap(sd/'weather.npy',mode='w+',dtype='float32',shape=(len(h),24,14))
            for st in range(0,len(h),1024):
                sl=slice(st,min(st+1024,len(h)))
                nodes=torch.from_numpy(np.array(local[cyc[sl],site,:7]))[:,None]
                ts=torch.from_numpy(times[sl,:7].copy())
                leads=torch.arange(1,25,dtype=torch.float32)[None].expand(len(nodes),-1)/4
                arr[sl]=interpolate(nodes,ts,leads)[:,0].numpy()
            arr.flush(); assert np.isfinite(arr).all()
            if split=='train':
                np.save(sd/'weather_climatology.npy',arr.mean(axis=0,dtype=np.float64).astype(np.float32))
                atomic_json(sd/'train_descriptors.json',dict(ramp_q90=float(np.quantile(np.abs(hh[:,-1]-hh[:,-5]),.9)),
                        mean_cf=float(yy.mean()),std_cf=float(yy.std())))
            del arr
            atomic_json(sd/'SUCCESS.json',dict(station_id=ids[site],rows=len(h)))
        audit['splits'][split]=dict(rows=len(h),spatial_max_error=spatial_error,clamped_left_fraction=float(np.mean(times[:,0]>.25)),
                min_origin=int(np.load(dest/'origin_unix.npy').min()),max_origin=int(np.load(dest/'origin_unix.npy').max()))
        print(split,audit['splits'][split],flush=True)
    for p in [COLUMN/'column_train_stats.json',COLUMN/'station_dem_srtm30m.json',LOCAL/'manifest.json']:
        audit['source_hashes'][str(p.relative_to(ROOT))]=sha(p)
    atomic_json(out/'AUDIT.json',audit)

if __name__=='__main__': main()
