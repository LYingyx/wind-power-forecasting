"""Compact, common origin-only labels; correlations are descriptive, not causal."""
from settings import *
from residual_inputs import mapping_history,topology
import numpy as np,csv

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-9 and np.std(y)>1e-9 else None

def main():
    acf=[];cross=[]
    for split in ['train','val']:
        origins=np.load(CACHE/split/'origin_unix.npy');n=len(origins)
        power=np.load(CACHE/split/'station_history.npy',mmap_mode='r')[:,:,-2].copy()
        mapped=np.empty_like(power)
        for site in range(56):mapped[:,site]=mapping_history(split,site)[0][:,-2]
        residual=power-mapped
        ci=np.load(CACHE/split/'weather_cycle_index.npy');past=np.load(PREPARED/split/'past_weather.npy',mmap_mode='r')
        mean=np.load(CACHE/'static/nwp_mean.npy');sd=np.load(CACHE/'static/nwp_std.npy')
        uv=past[ci,:,-1,1:3]*sd[1:3]+mean[1:3]
        labels=np.zeros((n,56,7),dtype='i1')
        for site in range(56):
            ids,geo,valid,cap=topology(site,True);weight=cap*valid;unit=geo[:,:2]/np.maximum(geo[:,2:3],.1)
            cos=(uv[:,site,None,:]*unit[None]).sum(-1)/np.sqrt((uv[:,site]**2).sum(-1)+4)[:,None]
            for ring,(lo,hi) in enumerate([(0,20),(20,60),(60,150.0001)]):
                w=weight*((geo[:,2]>=lo)&(geo[:,2]<hi));labels[:,site,ring]=int(w.sum()>0)
                signs=residual[:,ids]>0
                fraction=(signs*w[None]).sum(1)/max(w.sum(),1e-12)
                labels[:,site,3+ring]=np.where(w.sum()>0,(np.maximum(fraction,1-fraction)>=2/3).astype('i1'),2)
            w=weight[:8];alignment=(cos[:,:8]*w[None]).sum(1)/max(w.sum(),1e-12)
            labels[:,site,6]=np.where(alignment<-.3,0,np.where(alignment>.3,2,1))
            for lag in [900,3600,10800,21600]:
                pos=np.searchsorted(origins,origins+lag);ok=pos<n;ok[ok]&=origins[pos[ok]]==origins[ok]+lag
                for kind,values in [('raw',power),('residual',residual)]:
                    acf.append(dict(split=split,site=site,kind=kind,lag_hours=lag/3600,n=int(ok.sum()),correlation=corr(values[ok,site],values[pos[ok],site])))
            for j,dist in zip(ids,geo[:,2]):
                if j<=site or dist>150:continue
                cross.append(dict(split=split,a=site,b=int(j),distance_km=float(dist),raw=corr(power[:,site],power[:,j]),residual=corr(residual[:,site],residual[:,j])))
        np.save(ASSETS/f'diagnostic_labels_{split}.npy',labels)
        print('diagnostic labels',split,labels.shape,flush=True)
    for name,rows in [('residual_autocorrelation.csv',acf),('residual_cross_station_correlation.csv',cross)]:
        with (RUN/'analysis'/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(ASSETS/'DIAGNOSTICS_READY.json',dict(timestamp=now(),years=[2022,2023],label_columns=['ring0_present','ring1_present','ring2_present','ring0_residual_sign_agreement','ring1_residual_sign_agreement','ring2_residual_sign_agreement','near8_alignment'],sign_threshold=2/3,alignment_threshold=.3,acf_matching='exact origin timestamp pairs, no row shifts',wind='latest available past hourly 10m forecast vector',interpretation='OOF mapping for 2022, final 2022 mapping for 2023; residual includes mapping and operational errors'))
if __name__=='__main__':main()
