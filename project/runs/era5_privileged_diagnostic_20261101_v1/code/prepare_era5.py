"""Read-only GRIB extraction, coordinate/time/field guards, no 2024 access."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import eccodes as ec
import numpy as np
from functools import lru_cache

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
OUT = (RUN / 'output').resolve()
BUNDLE = (PROJECT / 'data/operational_bundle_v1').resolve()
SOURCE = Path('/hj102_data_large/ERA5')
STATIC = BUNDLE / 'cache/static'
SURFACE = [('mean_sea_level_pressure','msl',151,'Pa'),
           ('10m_u_component_of_wind','10u',165,'m s**-1'),
           ('10m_v_component_of_wind','10v',166,'m s**-1'),
           ('2m_temperature','2t',167,'K')]
PRESSURE = [(n,lev,p,u) for n,p,u in [('z',129,'m**2 s**-2'),
            ('q',133,'kg kg**-1'),('t',130,'K'),('u',131,'m s**-1'),
            ('v',132,'m s**-1')] for lev in [1000,925]]

def stamp(): return datetime.now().astimezone().isoformat()
def atomic(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
@lru_cache(maxsize=16)
def load(name):return np.load(STATIC/(name+'.npy'))
def utc(t):return datetime.fromtimestamp(int(t),timezone.utc)
def source_path(t):
    dt=utc(t)
    if dt.year not in [2022,2023]: raise ValueError('This diagnostic forbids other ERA5 years')
    return SOURCE/str(dt.year)/f'{dt:%Y-%m-%dT%H}:00:00.grib'

def scan(path, pressure):
    """Inspect every GRIB1 PDS, select by parameter/level, never message position.

    Nonselected packed fields are skipped with seek; ecCodes independently
    validates all selected metadata before reading any values.
    """
    result=[]
    with path.open('rb') as f:
        size=os.fstat(f.fileno()).st_size;off=0
        while off<size:
            f.seek(off);head=f.read(128);j=head.find(b'GRIB')
            if j<0:
                if head.strip(b'\0\r\n '):raise ValueError(f'Invalid GRIB padding {path}:{off}')
                if off+len(head)<size:raise ValueError('Excessive padding')
                break
            if head[:j].strip(b'\0\r\n '):raise ValueError('Invalid message separator')
            off+=j;head=head[j:]
            if head[7]!=1:raise ValueError(f'Unsupported GRIB edition in {path}')
            length=int.from_bytes(head[4:7],'big');pds=head[8:]
            if length<20 or length&(1<<23):raise ValueError('Unsupported GRIB length')
            if pds[3]!=128 or pds[4]!=98:raise ValueError('Unexpected parameter table/centre')
            param=pds[8];kind=pds[9];level=int.from_bytes(pds[10:12],'big')
            if not pressure or (param in [129,130,131,132,133] and kind==100 and level in [1000,925]):
                result.append(off)
            off+=length
    return result

def field(g, expected, expected_time):
    name,kind,level,unit=expected
    for key,value in [('shortName',name),('typeOfLevel',kind),('level',level),('units',unit),
                      ('gridType','regular_ll'),('Ni',1440),('Nj',721),
                      ('iScansNegatively',0),('jScansPositively',0),
                      ('jPointsAreConsecutive',0),('alternativeRowScanning',0),
                      ('bitmapPresent',0),('numberOfMissing',0),('stepType','instant')]:
        if ec.codes_get(g,key)!=value:raise ValueError(f'{key}: {ec.codes_get(g,key)} != {value}')
    if str(ec.codes_get(g,'expver')).zfill(4) not in ['0001','0005']:raise ValueError('Unsupported ERA5 experiment version')
    dt=utc(expected_time)
    assert ec.codes_get(g,'validityDate')==int(dt.strftime('%Y%m%d'))
    assert ec.codes_get(g,'validityTime')==dt.hour*100
    lat0=ec.codes_get(g,'latitudeOfFirstGridPointInDegrees');lat1=ec.codes_get(g,'latitudeOfLastGridPointInDegrees')
    lon0=ec.codes_get(g,'longitudeOfFirstGridPointInDegrees');lon1=ec.codes_get(g,'longitudeOfLastGridPointInDegrees')
    di=ec.codes_get(g,'iDirectionIncrementInDegrees');dj=ec.codes_get(g,'jDirectionIncrementInDegrees')
    assert lat0==90 and lat1==-90 and di==dj==.25
    assert np.isclose((lon1-lon0)%360,359.75)
    lat,lon=load('weather_latitude'),load('weather_longitude')
    rows=np.rint((lat0-lat)/dj).astype(int)
    cols=np.rint((lon-lon0)%360/di).astype(int)
    np.testing.assert_allclose(lat0-rows*dj,lat,atol=1e-7)
    np.testing.assert_allclose((lon0+cols*di)%360,lon,atol=1e-7)
    idx=(rows[:,None]*1440+cols).ravel().astype('i4')
    values=np.asarray(ec.codes_get_double_elements(g,'values',idx.tolist()),dtype='f4').reshape(24,25)
    assert np.isfinite(values).all()
    if name=='q':values=np.maximum(values,0)
    return values, float(lon0)

def pressure_one(task):
    i,t=task;path=source_path(t);before=path.stat();offsets=scan(path,True)
    if len(offsets)!=10:raise ValueError(f'{path}: expected 10 unique pressure fields, got {len(offsets)}')
    result=np.full((10,24,25),np.nan,'f4');seen=set();lons=set();versions=set()
    lookup={(n,lev):(c,p,u) for c,(n,lev,p,u) in enumerate(PRESSURE)}
    with path.open('rb') as f:
        for off in offsets:
            f.seek(off);g=ec.codes_grib_new_from_file(f)
            try:
                key=(ec.codes_get(g,'shortName'),ec.codes_get(g,'level'))
                c,p,u=lookup[key];assert key not in seen and ec.codes_get(g,'paramId')==p
                result[c],lon=field(g,(key[0],'isobaricInhPa',key[1],u),t)
                seen.add(key);lons.add(lon);versions.add(str(ec.codes_get(g,'expver')).zfill(4))
            finally:ec.codes_release(g)
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return i,result,dict(path=str(path),size=before.st_size,mtime_ns=before.st_mtime_ns,
                       longitude_starts=sorted(lons),expver=sorted(versions),sha256=hashlib.sha256(result.tobytes()).hexdigest())

def surface_day(task):
    date,rows,times=task;day=utc(times[0]);result=np.full((len(times),4,24,25),np.nan,'f4')
    lookup={int(t):i for i,t in enumerate(times)};receipts=[]
    for c,(directory,name,param,unit) in enumerate(SURFACE):
        path=SOURCE/'surface_new'/directory/str(day.year)/(date+'.grib');before=path.stat();seen=set();lons=set()
        with path.open('rb') as f:
            for off in scan(path,False):
                f.seek(off);g=ec.codes_grib_new_from_file(f)
                try:
                    datecode=ec.codes_get(g,'validityDate');clock=ec.codes_get(g,'validityTime')
                    dt=datetime.strptime(f'{datecode:08d}{clock:04d}','%Y%m%d%H%M').replace(tzinfo=timezone.utc)
                    t=int(dt.timestamp())
                    if t not in lookup:continue
                    assert t not in seen and ec.codes_get(g,'paramId')==param
                    result[lookup[t],c],lon=field(g,(name,'surface',0,unit),t)
                    seen.add(t);lons.add(lon)
                finally:ec.codes_release(g)
        assert seen==set(lookup)
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        receipts.append(dict(path=str(path),size=before.st_size,mtime_ns=before.st_mtime_ns,longitude_starts=sorted(lons)))
    return rows,result,receipts

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    alltimes=[]
    for sp in ['train','val']:
        future=np.load(BUNDLE/'cache'/sp/'weather_valid_unix.npy')
        past=np.load(BUNDLE/'physical'/sp/'past_valid_unix.npy')
        alltimes.extend([future.ravel(),past.ravel()])
    times=np.unique(np.concatenate(alltimes));assert set((times.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970).tolist())=={2022,2023}
    missing=[str(source_path(t)) for t in times if not source_path(t).is_file()]
    days=sorted(set(utc(t).strftime('%Y-%m-%d') for t in times))
    for day in days:
        for directory,*_ in SURFACE:
            f=SOURCE/'surface_new'/directory/day[:4]/(day+'.grib')
            if not f.is_file():missing.append(str(f))
    atomic(OUT/'SOURCE_COVERAGE.json',dict(timestamp=stamp(),times=len(times),days=len(days),missing=missing,years=[2022,2023],test_era5_read=False))
    if missing:raise FileNotFoundError(missing[:10])
    np.save(OUT/'valid_unix.npy',times)
    for sp in ['train','val']:
        for name,path in [('future',BUNDLE/'cache'/sp/'weather_valid_unix.npy'),('past',BUNDLE/'physical'/sp/'past_valid_unix.npy')]:
            t=np.load(path);idx=np.searchsorted(times,t);np.testing.assert_array_equal(times[idx],t);np.save(OUT/f'{sp}_{name}_index.npy',idx)
    for name,shape,dtype,fill in [('era5_crop.npy',(len(times),14,24,25),'f4',np.nan),('surface_done.npy',(len(times),),'?',False),('pressure_done.npy',(len(times),),'?',False)]:
        p=OUT/name
        if not p.exists():
            a=np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape);a[:]=fill;a.flush()
        else:assert np.load(p,mmap_mode='r').shape==shape
    atomic(OUT/'PROTOCOL.json',dict(timestamp=stamp(),stage='D1_D2_only',era5_source=str(SOURCE),operational_bundle=str(BUNDLE),
        channel_order=['msl','u10','v10','t2m']+[f'{n}{lev}' for n,lev,_,_ in PRESSURE],crop=dict(north=47.25,south=41.5,west=120.75,east=126.75,shape=[24,25],resolution=.25),
        train_year=2022,selection_year=2023,excluded_selection_month=11,era5_years_read=[2022,2023],test_era5_read=False,
        normalization='unchanged operational2022 mean/std; no ERA5 fitting',labels='auxiliary losses and offline diagnostics only; D3 requires separately labeled oracle input',
        source_validation='every selected message: key,units,validity,grid,scan,expver,missing; dynamically inspect every PDS; never select by message position',
        no_new_power_training_started=True))
    print('plan',len(times),'hours',len(days),'days',flush=True)

def extract(mode,workers):
    times=np.load(OUT/'valid_unix.npy');raw=np.lib.format.open_memmap(OUT/'era5_crop.npy',mode='r+')
    done=np.lib.format.open_memmap(OUT/f'{mode}_done.npy',mode='r+');todo=np.flatnonzero(~done);start=time.monotonic();initial=int(done.sum())
    if mode=='pressure':
        version_path=OUT/'pressure_expver.npy'
        if not version_path.exists():
            versions=np.lib.format.open_memmap(version_path,mode='w+',dtype='i1',shape=(len(times),));versions[:]=np.where(done,1,0);versions.flush()
        else:versions=np.lib.format.open_memmap(version_path,mode='r+')
    if mode=='surface':
        groups={}
        for i in todo:groups.setdefault(utc(times[i]).strftime('%Y-%m-%d'),[]).append(int(i))
        tasks=[(d,ids,times[ids].tolist()) for d,ids in groups.items()];fn=surface_day
    else:tasks=[(int(i),int(times[i])) for i in todo];fn=pressure_one
    with (OUT/f'{mode}_receipts.jsonl').open('a') as receipt,ProcessPoolExecutor(max_workers=workers) as pool:
        for n,(idx,value,meta) in enumerate(pool.map(fn,tasks,chunksize=1),1):
            if mode=='surface':raw[idx,:4]=value
            else:
                raw[idx,4:]=value
                versions[idx]=1 if meta['expver']==['0001'] else 5
            done[idx]=True;receipt.write(json.dumps(dict(index=idx,source=meta),ensure_ascii=False)+'\n')
            if n%16==0 or n==len(tasks):
                raw.flush();done.flush();receipt.flush();count=int(done.sum());elapsed=time.monotonic()-start
                if mode=='pressure':versions.flush()
                status=dict(timestamp=stamp(),stage=mode,done=count,total=len(times),workers=workers,elapsed_seconds=elapsed,
                    eta_seconds=(len(times)-count)*elapsed/max(count-initial,1))
                atomic(OUT/f'STATUS_{mode}.json',status);print(json.dumps(status),flush=True)
    assert done.all();raw.flush();done.flush()
    if mode=='pressure':
        versions.flush()
        atomic(OUT/'PRESSURE_VERSION_AUDIT.json',dict(timestamp=stamp(),final_frames=int((versions==1).sum()),era5t_frames=int((versions==5).sum()),policy='retain supplied ERA5T as provisional reference with explicit masks; never silently label as final ERA5'))
    atomic(OUT/f'{mode.upper()}_READY.json',dict(timestamp=stamp(),frames=len(times),passed=True))

def labels():
    assert (OUT/'SURFACE_READY.json').exists() and (OUT/'PRESSURE_READY.json').exists()
    sys.path.insert(0,str(PROJECT/'scripts'))
    import prepare_operational_pangu as op
    raw=np.load(OUT/'era5_crop.npy',mmap_mode='r');ids=load('bilinear_patch_index');weights=load('bilinear_weight');mu=load('nwp_mean');sd=load('nwp_std')
    physics=op.load_physics();elevation=np.load(BUNDLE/'physical/elevation.npy')
    counts={}
    for sp in ['train','val']:
        for part in ['future','past']:
            idx=np.load(OUT/f'{sp}_{part}_index.npy');folder=OUT/'labels'/sp;folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/f'era5_{part}_pressure_final_mask.npy',np.load(OUT/'pressure_expver.npy')[idx]==1)
            shape=(len(idx),56,idx.shape[1],14)
            dst=np.lib.format.open_memmap(folder/f'era5_{part}.npy',mode='w+',dtype='f4',shape=shape)
            sel=np.lib.format.open_memmap(folder/f'era5_{part}_weather_sel.npy',mode='w+',dtype='f4',shape=(*shape[:-1],15))
            for start in range(0,len(idx),48):
                end=min(start+48,len(idx));x=np.asarray(raw[idx[start:end]]);assert np.isfinite(x).all()
                physical=op.bilinear(x,ids,weights).transpose(0,2,1,3)
                dst[start:end]=(physical-mu)/sd
                den,stab=physics.thermodynamics(physical,elevation[None,:,None]);dyn=op.dynamics(x,ids,weights,load('station_latitude'))
                sel[start:end]=np.concatenate([den,dyn,stab],axis=-1)
            dst.flush();sel.flush();counts[f'{sp}_{part}']=list(shape)
    atomic(OUT/'LABELS_READY.json',dict(timestamp=stamp(),passed=True,shapes=counts,weather_sel_channels=15,state_channels='retained actual power; not ERA5 labels',test_read=False))
    while not (OUT/'D2_SUMMARY.json').exists():
        if (OUT/'PIPELINE_FAILURE.json').exists():raise RuntimeError('Diagnostic failed; final audit cannot proceed')
        time.sleep(10)
    from audit_final import main as final_audit
    final_audit()

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','surface','pressure','labels']);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='plan':plan()
    elif a.stage=='labels':labels()
    else:extract(a.stage,a.workers)
if __name__=='__main__':main()
