"""Do large 0-6h power changes come from power extrapolation or weather? Zero training. 2022 fit curves, 2023 eval.
Sources: persistence, B0 (shared power-only model, own history), W0_R3/TF_R3 (shared, seed-mean), Pangu-implied, ERA5-implied."""
import numpy as np, json, sys
sys.path.insert(0,'/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_protocol_restart_20261114_v1/code')
from operational import interpolate_nodes
P='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs'
D=f'{P}/daily_protocol_restart_20261114_v1/data'; S=f'{P}/privileged_wind_age_pilot_20261101_v1/data'; FT=f'{P}/daily_site_finetune_20261114_v1'
A=f'{P}/daily_protocol_restart_20261114_v1/assets'
keep=np.array(json.load(open(f'{A}/COHORT.json'))['original_indices']); emu=np.load(f'{S}/cache/static/nwp_mean.npy')[1:3]; esd=np.load(f'{S}/cache/static/nwp_std.npy')[1:3]; mu=np.load(f'{A}/nwp_mean.npy'); sd=np.load(f'{A}/nwp_std.npy')
LEADS=[3,7,15,23]  # 1h,2h,4h,6h (index into 24 steps)
def load(sp):
    o=np.load(f'{D}/{sp}/origin.npy'); cy=np.load(f'{D}/{sp}/cycle.npy'); off=np.load(f'{D}/{sp}/offset.npy')
    o2=np.load(f'{S}/common/{sp}/origin_unix.npy'); assert np.array_equal(o,o2)
    cy2=np.load(f'{S}/common/{sp}/cycle.npy'); off2=np.load(f'{S}/common/{sp}/weather_time_offset_hours.npy'); assert np.allclose(off,off2)
    nodes=np.load(f'{D}/{sp}/nodes.npy',mmap_mode='r'); era=np.load(f'{S}/labels/{sp}/future_uv.npy',mmap_mode='r')
    q=np.r_[0.,(np.array(LEADS)+1)/4]
    wsP=np.zeros((len(o),55,len(q)),'f4'); wsE=np.zeros_like(wsP)
    for s in range(55):
        n=np.asarray(nodes[cy,s])*sd+mu; w=interpolate_nodes(n[...,1:3],off,q); wsP[:,s]=np.hypot(w[...,0],w[...,1])
        e=(np.asarray(era[cy2,keep[s]])*esd+emu).astype('f4'); w=interpolate_nodes(e,off,q); wsE[:,s]=np.hypot(w[...,0],w[...,1])
    h=np.load(f'{D}/{sp}/h.npy',mmap_mode='r'); y=np.load(f'{D}/{sp}/y.npy',mmap_mode='r')
    return o,np.asarray(h[:,:,-1]),np.asarray(y[:,:,LEADS]),wsP,wsE
o2,h2,y2,P2,E2=load('train'); o3,h3,y3,P3,E3=load('val')
print('ERA5 10m ws mean %.2f, Pangu %.2f (2023)'%(E3.mean(),P3.mean()))
B0=np.mean([np.asarray(np.load(f'{D}/reference_r{sd_}/val.npy',mmap_mode='r')[:,:,LEADS]) for sd_ in range(42,47)],0)
def curve_fit(ws,p):
    e=np.quantile(ws,np.linspace(0,1,41)); b=np.clip(np.searchsorted(e,ws)-1,0,39); m=np.array([p[b==i].mean() if (b==i).any() else np.nan for i in range(40)])
    m=np.where(np.isnan(m),np.nanmean(p),m); return lambda w: m[np.clip(np.searchsorted(e,w)-1,0,39)]
R={}
def add(k,v): R.setdefault(k,[]).append(v)
names=['persist','B0','W0','TF','Pangu','ERA5']
for s in range(55):
    # curves: 2022, weather-only, fitted per source on concurrent (valid-time) power: use lead pairs (ws at lead k, y at lead k) pooled
    fP=curve_fit(P2[:,s,1:].ravel(),y2[:,s].ravel()); fE=curve_fit(E2[:,s,1:].ravel(),y2[:,s].ravel())
    W0=np.mean([np.load(f'{FT}/jobs/W0_P1_FULL_s{s:02d}_r{x}/p0_prediction.npy')[:,0][:,LEADS] for x in range(42,47)],0)
    TF=np.mean([np.load(f'{FT}/jobs/TF_P1_FULL_s{s:02d}_r{x}/p0_prediction.npy')[:,0][:,LEADS] for x in range(42,47)],0)
    hl=h3[:,s][:,None]; dy=y3[:,s]-hl
    pred_change={'persist':np.zeros_like(dy),'B0':B0[:,s]-hl,'W0':W0-hl,'TF':TF-hl,
                 'Pangu':fP(P3[:,s,1:])-fP(P3[:,s,:1]),'ERA5':fE(E3[:,s,1:])-fE(E3[:,s,:1])}
    for j,L in enumerate(LEADS):
        d=dy[:,j]; ev=np.abs(d)>.3; add(('evfrac',L),ev.mean())
        for nme in names:
            pc=pred_change[nme][:,j]
            add((nme,L,'capture'),(np.sign(d)*pc)[ev].sum()/np.abs(d)[ev].sum())
            add((nme,L,'corr'),np.corrcoef(pc,d)[0,1] if pc.std()>0 else 0.)
            # event detection: POD/FAR with |pred change|>0.15 in same direction as event
            hit=(np.abs(pc)>.15)&(np.sign(pc)==np.sign(d))
            add((nme,L,'POD'),hit[ev].mean())
            fa=(np.abs(pc)>.15)&~((np.abs(d)>.15)&(np.sign(d)==np.sign(pc))); add((nme,L,'FAR'),fa.sum()/max((np.abs(pc)>.15).sum(),1))
        # unused info: correlation of model residual with weather-implied change, on all and on events
        for m,pm in [('W0',W0),('TF',TF)]:
            res=y3[:,s,j]-pm[:,j]
            for src in ['Pangu','ERA5']:
                add((m,src,L,'rescorr'),np.corrcoef(res,pred_change[src][:,j])[0,1])
M=lambda *k: np.mean(R[k])
for L in LEADS:
    print(f'\n== lead {(L+1)/4:.0f} h: share of samples with |Δ|>0.3: {M("evfrac",L):.3%}')
    print('   source    capture(|Δ|>.3)  corr(predΔ,Δ)   POD(|predΔ|>.15 same dir)  FAR')
    for nme in names: print(f'   {nme:8s}  {M(nme,L,"capture"):9.3f}  {M(nme,L,"corr"):12.3f}  {M(nme,L,"POD"):14.3f}  {M(nme,L,"FAR"):14.3f}')
    for m in ['W0','TF']: print(f'   corr(residual of {m}, weather-implied Δ): Pangu {M(m,"Pangu",L,"rescorr"):+.3f}  ERA5 {M(m,"ERA5",L,"rescorr"):+.3f}')
