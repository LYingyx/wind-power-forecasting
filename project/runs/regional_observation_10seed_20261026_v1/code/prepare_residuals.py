"""Train-only OOF isotonic curves and a single missing historical NWP endpoint."""
from settings import *
from physics import neighbor_geometry
from residual_inputs import mapping_history,month_index
from sklearn.isotonic import IsotonicRegression
import numpy as np,h5py,shutil,csv

def save(path,array):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.save(f,array)
    os.replace(tmp,path)

def boundary(split):
    path=ASSETS/f'boundary_weather_{split}.npy'
    if path.exists():return json.loads((ASSETS/f'boundary_{split}.json').read_text())
    valid=np.load(PREPARED/split/'past_valid_unix.npy')[:,0]-3600
    original_issue=np.load(PREPARED/split/'past_issue_unix.npy')[:,0]
    ci=np.load(CACHE/split/'weather_cycle_index.npy');origin=np.load(CACHE/split/'origin_unix.npy')
    # Older run predicts the predecessor valid hour with one-hour forecast lead.
    with h5py.File(H5,'r') as f:
        issues=f['nwp/issue_unix'][:];lookup=f['nwp/run_lead_to_frame'][:]
        run=np.searchsorted(issues,valid-3600);ok=(run<len(issues))&(issues[np.minimum(run,len(issues)-1)]==valid-3600)
        frame=np.where(ok,lookup[np.minimum(run,len(issues)-1),0],-1);ok &= frame>=0
        assert np.all(valid<1704038400),'No 2024 local-time weather valid dates'
        assert np.all(np.where(ok[ci],issues[np.minimum(run[ci],len(issues)-1)],original_issue[ci])+900<=origin)
        lat=f['crop/latitude'][:];lon=f['crop/longitude'][:]
        slat=np.load(CACHE/'static/station_latitude.npy');slon=np.load(CACHE/'static/station_longitude.npy')
        ids=[];weights=[]
        for y,x in zip(slat,slon):
            yy=np.argsort(abs(lat-y))[:2];xx=np.argsort(abs(lon-x))[:2];yy=yy[np.argsort(lat[yy])];xx=xx[np.argsort(lon[xx])]
            a=(y-lat[yy[0]])/(lat[yy[1]]-lat[yy[0]]);b=(x-lon[xx[0]])/(lon[xx[1]]-lon[xx[0]])
            ids.append([i*len(lon)+j for i in yy for j in xx]);weights.append([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
        ids=np.asarray(ids);weights=np.asarray(weights,dtype='f4')
        unique=np.unique(frame[ok]);points=np.empty((len(unique),56,4),dtype='f4')
        for start in range(0,len(unique),128):
            selected=unique[start:start+128]
            # Read only explicitly referenced historical frames and four channels.
            raw=f['nwp/values'][selected,:4].reshape(-1,4,len(lat)*len(lon))
            points[start:start+len(raw)]=(raw[:,:,ids]*weights[None,None]).sum(-1).transpose(0,2,1)
        data=np.asarray(np.load(PREPARED/split/'past_weather.npy',mmap_mode='r')[:,:,0,:4]).copy()
        data[ok]=points[np.searchsorted(unique,frame[ok])]
    # Earliest archive boundary can lack the predecessor run. Explicit endpoint
    # hold, not fabricated observations, is limited to those listed cycles.
    report=dict(split=split,cycles=len(valid),exact_predecessor=int(ok.sum()),endpoint_hold_cycles=np.flatnonzero(~ok).tolist(),latest_valid_unix=int(valid.max()),all_issue_latency_checked=True)
    save(path,data);atomic_json(ASSETS/f'boundary_{split}.json',report);return report

def main():
    setup();ASSETS.mkdir(exist_ok=True);os.nice(10)
    for split in ['train','val']:print('boundary',boundary(split),flush=True)
    static=CACHE/'static';lat=np.load(static/'station_latitude.npy');lon=np.load(static/'station_longitude.npy');cap=np.load(static/'station_capacity_mw.npy')
    ids,geo,mask=neighbor_geometry(lat,lon,55,150)
    np.savez(ASSETS/'topology.npz',ids=ids,geometry=geo,mask=mask,capacity=cap[ids])
    shutil.copy2(NEIGHBOR_RUN/'assets/neighbor_tables.npz',ASSETS/'neighbor_tables.npz')
    origin=np.load(CACHE/'train/origin_unix.npy');time=origin-900;month=month_index(time)
    year=(time+8*3600).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970;assert np.all(year==2022)
    power=np.load(CACHE/'train/station_history.npy',mmap_mode='r')
    valtime=np.load(CACHE/'val/origin_unix.npy')-900;valpower=np.load(CACHE/'val/station_history.npy',mmap_mode='r')
    table={};rows=[];folds={}
    for mo in range(12):
        start=int(np.datetime64(f'2022-{mo+1:02d}-01','s').astype('i8'))-8*3600
        end=int((np.datetime64(f'2022-{mo+1:02d}','M')+1).astype('datetime64[s]').astype('i8'))-8*3600
        fit=(time<start-86400)|(time>=end+86400)
        assert not np.any(fit&(month==mo));assert fit.sum()>1000
        folds[str(mo)]=dict(excluded_start_unix=start-86400,excluded_end_unix=end+86400,n_fit=int(fit.sum()),n_held=int((month==mo).sum()))
    for site in range(56):
        _,speed=mapping_history('train',site,map_power=False);x=speed[:,-2];y=np.asarray(power[:,site,-2],dtype='f8')
        assert np.isfinite(x).all() and np.isfinite(y).all()
        def fit_curve(selected):
            model=IsotonicRegression(increasing=True,y_min=0,y_max=1.2,out_of_bounds='clip').fit(x[selected],y[selected])
            return dict(x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())
        final=fit_curve(np.ones(len(x),dtype=bool));oof={};prediction=np.empty(len(x))
        for mo in range(12):
            f=folds[str(mo)];fit=(time<f['excluded_start_unix'])|(time>=f['excluded_end_unix']);c=fit_curve(fit);oof[str(mo)]=c
            held=month==mo;prediction[held]=np.interp(x[held],c['x'],c['y'])
        table[str(site)]=dict(final=final,oof=oof)
        _,v=mapping_history('val',site,map_power=False);vx=v[:,-2];vy=np.asarray(valpower[:,site,-2],dtype='f8')
        for label,pred,truth in [('train_in_sample',np.interp(x,final['x'],final['y']),y),('train_OOF',prediction,y),('val_2023',np.interp(vx,final['x'],final['y']),vy)]:
            e=pred-truth;rows.append(dict(site=site,split=label,n=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean())))
        if site%8==0:print('curves',site,flush=True)
    atomic_json(ASSETS/'curves.json',dict(training_year=2022,input='density-adjusted 10m wind proxy in m/s; same dry proxy as SEL',output='capacity factor, isotonic bounded [0,1.2]',oof='Leave local observation month plus 24h buffer out; observation-month applied to every history token',folds=folds,stations=table))
    with (RUN/'analysis/curve_quality.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(ASSETS/'CURVES_READY.json',dict(timestamp=now(),passed=True,train_points=len(time),validation_points=len(valtime),stations=56,folds=12,test_year_accessed=False,source_weather='Only train/val caches and explicitly indexed predecessor frames',interpretation='Power mapping innovations mix weather, calibration, availability and dispatch effects; not pure NWP error',files={p.name:sha(p) for p in ASSETS.iterdir() if p.is_file()}))

if __name__=='__main__':main()
