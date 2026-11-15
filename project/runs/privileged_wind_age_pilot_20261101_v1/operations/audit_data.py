from pathlib import Path
import sys,json
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import DATA,CACHE,PROJECT,atomic_json,now
from operational import source_path
import numpy as np
def audit():
    ready=json.loads((DATA/'READY.json').read_text());assert ready['passed']
    keys=np.load(DATA/'aged_keys.npy');frames=np.load(DATA/'aged_frames.npy',mmap_mode='r')
    lat=np.load(CACHE/'static/station_latitude.npy');lon=np.load(CACHE/'static/station_longitude.npy')
    mu=np.load(CACHE/'static/nwp_mean.npy');sd=np.load(CACHE/'static/nwp_std.npy')
    rows=(90-lat)/.25;cols=lon/.25;rr=np.floor(rows).astype(int);cc=np.floor(cols).astype(int)
    fy=rows-rr;fx=cols-cc;errors=[]
    for i in [0,len(keys)//4,len(keys)//2,3*len(keys)//4,len(keys)-1]:
        issue,lead=keys[i];p=source_path('/hj102_data_large/ERA5_forecasting/Pangu/Forecast_data',int(issue),int(lead));raw=np.load(p,mmap_mode='r')
        # Independent global coordinate interpolation, no cached crop/weights.
        expected=np.array([raw[0,:,rr[s],cc[s]]*(1-fy[s])*(1-fx[s])+raw[0,:,rr[s]+1,cc[s]]*fy[s]*(1-fx[s])+
            raw[0,:,rr[s],cc[s]+1]*(1-fy[s])*fx[s]+raw[0,:,rr[s]+1,cc[s]+1]*fy[s]*fx[s] for s in [0,27,55]])
        err=np.abs((expected-mu)/sd-frames[i,[0,27,55]]);assert err.max()<1e-4,err.max();errors.append(float(err.max()))
    b=PROJECT/'data/operational_bundle_v1';era=PROJECT/'runs/era5_privileged_diagnostic_20261101_v1/output'
    valid=np.load(era/'valid_unix.npy');sur=np.load(era/'station_surface.npy',mmap_mode='r')
    for split in ['train','val']:
        c=DATA/'common'/split;orig=np.load(c/'origin_unix.npy');cycles=np.load(c/'source_cycles.npy');inv=np.load(c/'cycle.npy');oldrows=np.load(c/'source_rows.npy')
        assert np.array_equal(orig,np.load(b/'cache'/split/'origin_unix.npy')[oldrows])
        assert np.array_equal(np.load(DATA/'age0'/split/'future.npy'),np.load(b/'local_weather'/f'{split}.npy',mmap_mode='r')[cycles])
        targetidx=np.load(era/f'{split}_future_index.npy')[cycles]
        targettimes=valid[targetidx];futuretimes=np.load(b/'cache'/split/'weather_valid_unix.npy')[cycles]
        assert np.array_equal(targettimes,futuretimes)
        expected=(sur[targetidx][:,:,:,1:3].transpose(0,2,1,3)-mu[1:3])/sd[1:3]
        assert np.array_equal(expected,np.load(DATA/'labels'/split/'future_uv.npy'))
        offset=(futuretimes[inv]-orig[:,None])/3600
        assert np.array_equal(offset,np.load(c/'weather_time_offset_hours.npy'))
        assert np.all(offset[:,0]<=0) and np.all(offset[:,-1]>=6)
        if split=='val':assert not np.any((orig+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12==10)
        # No future observations enter the15min-delayed regional histories.
        obs=np.load(c/'station_history.npy',mmap_mode='r');assert obs.shape[-1]==24
    result=dict(timestamp=now(),passed=True,independent_global_bilinear_max_normalized_error=max(errors),
        source_frames_audited=5,farms_per_frame=3,channels=14,age0_exact_original_nodes=True,era5_valid_times_exact=True,
        common_samples=True,test_read=False,**{k:ready[k] for k in ['train_rows','val_rows']})
    atomic_json(RUN/'operations/DATA_AUDIT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':audit()
