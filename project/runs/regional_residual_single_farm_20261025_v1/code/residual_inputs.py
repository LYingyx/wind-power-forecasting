"""As-of-origin NWP-power innovations; OOF calibration by observation month."""
from settings import *
import numpy as np
from functools import lru_cache

W0_VARIANTS=['R1','R1_RAWMAP','R1_SELF','R2','R2_PAIR','R3','R3_RAW','R4',
             'C1_R1','C1_R2','C1_R3','C1_R4','C2_R1','C2_R2','C2_R3','C2_R4']
TF_VARIANTS=['N3','R1','R2','R3','R4']
RESIDUAL_VARIANTS=set(W0_VARIANTS)
def family(v):
    if v.startswith(('C1_','C2_')):return v[3:]
    return v.split('_')[0]

@lru_cache(None)
def curves():return json.loads((ASSETS/'curves.json').read_text())

@lru_cache(None)
def topology(site,all_neighbors=False):
    with np.load(ASSETS/'topology.npz') as z:
        k=55 if all_neighbors else 8
        return tuple(z[n][site,:k].copy() for n in ['ids','geometry','mask','capacity'])

def month_index(unix):return (np.asarray(unix)+8*3600).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12

def calibrated(speed,observation_times,site,split):
    table=curves()['stations'][str(site)];out=np.empty_like(speed,dtype='f4')
    if split=='val':return np.interp(speed,table['final']['x'],table['final']['y']).astype('f4')
    assert split=='train'
    months=month_index(observation_times)
    for month in np.unique(months):
        mask=months==month;c=table['oof'][str(int(month))]
        out[mask]=np.interp(speed[mask],c['x'],c['y'])
    return out

def adjusted_speed(w,elevation):
    # Identical dry density proxy to physics.thermodynamics, physical [msl,u,v,T].
    w=np.asarray(w,dtype='f8');t=w[...,3];z=float(elevation)
    ps=w[...,0]*np.exp(-9.80665*z/(287.05*np.maximum(t+.0065*z/2,180)))
    thub=np.maximum(t-.65,180);phub=ps*np.exp(-9.80665*100/(287.05*np.maximum((t+thub)/2,180)))
    rho=phub/(287.05*thub)
    return ((rho/1.225)**(1/3)*np.hypot(w[...,1],w[...,2])).astype('f4')

def mapping_history(split,site,rows=None,map_power=True):
    """Map hourly physical weather to the 24 actual observation times.

    The extra predecessor hour fills the missing endpoint; missing issuance cycles
    use an audited hold of the first actual hourly forecast (26 train/32 val cycles).
    Each contributing forecast was issued >=15min before the prediction origin.
    Training curve for EVERY historical point excludes its entire local month
    plus 24h padding, preventing overlap across adjacent rolling windows.
    """
    assert split in ['train','val']
    rows=slice(None) if rows is None else rows
    origin=np.load(CACHE/split/'origin_unix.npy')[rows]
    ci=np.load(CACHE/split/'weather_cycle_index.npy')[rows]
    past=np.load(PREPARED/split/'past_weather.npy',mmap_mode='r')
    future=np.load(LOCAL_WEATHER/f'{split}.npy',mmap_mode='r')
    boundary=np.load(ASSETS/f'boundary_weather_{split}.npy',mmap_mode='r')
    nodes=np.concatenate((boundary[:,site,None,:],past[:,site,:,:4],future[:,site,:,:4]),axis=1)
    first=np.load(PREPARED/split/'past_valid_unix.npy')[ci,0]-3600
    times=origin[:,None]+(np.arange(24)-23)*900
    coordinate=(times-first[:,None])/3600
    assert np.all(coordinate>=0) and np.all(coordinate<=nodes.shape[1]-1)
    lo=np.minimum(np.floor(coordinate).astype(int),nodes.shape[1]-2);fraction=coordinate-lo
    w=nodes[ci[:,None],lo]*(1-fraction[...,None])+nodes[ci[:,None],lo+1]*fraction[...,None]
    mean=np.load(CACHE/'static/nwp_mean.npy')[:4];std=np.load(CACHE/'static/nwp_std.npy')[:4]
    physical=w*std+mean;elevation=np.load(PREPARED/'elevation.npy')[site]
    speed=adjusted_speed(physical,elevation)
    return (calibrated(speed,times,site,split) if map_power else None),speed

def neighbor_arrays(split,site,variant,seed,limit=None):
    n=len(np.load(CACHE/split/'origin_unix.npy')) if limit is None else limit
    ids,geo,mask,cap=topology(site,family(variant) in ['R3','R4'])
    # A joint donor preserves neighbor-to-neighbor and power-to-weather coherence.
    source_split='train' if variant.startswith('C2_') else split
    rows=np.load(NEIGHBOR_RUN/'assets'/f'mismatch_{split}_r{seed}.npy',mmap_mode='r')[:n] if variant.startswith('C2_') else np.arange(n)
    power=np.load(CACHE/source_split/'station_history.npy',mmap_mode='r')
    p=np.asarray(power[rows[:,None],ids[None,:],:],dtype='f4')
    mapped=np.stack([mapping_history(source_split,int(j),rows)[0] for j in ids],axis=1)
    assert np.isfinite(mapped).all() and np.isfinite(p).all()
    return p,mapped,ids,geo,mask,cap
