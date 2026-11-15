"""Why weather-implied large changes often do not appear in power: operating point, plateau (possible curtailment), direction, short-lived weather changes.
2023, 4 h lead, ERA5 (true weather) and Pangu; zero training."""
import numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'info.py')).read().split("R={}")[0])
L=15; j=LEADS.index(L)
# extra queries for smoothing: implied change using mean of ws over [3h,5h] vs point at 4h
def ws_at(sp,qs):
    import sys
    o=np.load(f'{D}/{sp}/origin.npy'); cy2=np.load(f'{S}/common/{sp}/cycle.npy'); off=np.load(f'{D}/{sp}/offset.npy')
    era=np.load(f'{S}/labels/{sp}/future_uv.npy',mmap_mode='r'); out=np.zeros((len(o),55,len(qs)),'f4')
    for s in range(55):
        e=(np.asarray(era[cy2,keep[s]])*esd+emu).astype('f4'); w=interpolate_nodes(e,off,qs); out[:,s]=np.hypot(w[...,0],w[...,1])
    return out
Esm=ws_at('val',np.array([3.,3.5,4.,4.5,5.]))
H3=np.asarray(np.load(f'{D}/val/h.npy',mmap_mode='r'))
rows=[]
for s in range(55):
    fE=curve_fit(E2[:,s,1:].ravel(),y2[:,s].ravel()); fP=curve_fit(P2[:,s,1:].ravel(),y2[:,s].ravel())
    hl=h3[:,s]; d=y3[:,s,j]-hl
    impE=fE(E3[:,s,1+j])-fE(E3[:,s,0]); impP=fP(P3[:,s,1+j])-fP(P3[:,s,0]); impEs=fE(Esm[:,s]).mean(1)-fE(E3[:,s,0])
    hist=H3[:,s]; flat=(np.ptp(hist[:,-8:],1)<0.01)&(hl>0.05)&(hl<0.95)   # 2 h nearly constant, mid range: plateau (possible curtailment / limit)
    lvl=np.digitize(hl,[.1,.3,.7])
    rows.append((d,impE,impP,impEs,lvl,flat))
def far(imp,d,mask):
    sig=(np.abs(imp)>.15)&mask; ok=(np.abs(d)>.15)&(np.sign(d)==np.sign(imp)); return (sig&~ok).sum()/max(sig.sum(),1), sig.sum()
def agg(fn):
    a=np.array([fn(*r) for r in rows]); return a
print('4 h lead. FAR = share of weather-implied |Δ|>0.15 signals NOT followed by same-direction power change >0.15')
for name,k in [('ERA5',1),('Pangu',2),('ERA5 smoothed 3-5h',3)]:
    tot=[far(r[k],r[0],np.ones_like(r[0],bool)) for r in rows]; print(f'  {name:20s} all: FAR {np.sum([t[0]*t[1] for t in tot])/np.sum([t[1] for t in tot]):.3f}')
print('  ERA5 by current power level and direction (FAR, signal count):')
for lv,lab in enumerate(['<0.1','0.1-0.3','0.3-0.7','>=0.7']):
    for sgn,sl in [(1,'up'),(-1,'down')]:
        t=[far(r[1],r[0],(r[4]==lv)&(np.sign(r[1])==sgn)) for r in rows]; n=np.sum([x[1] for x in t])
        print(f'     level {lab:8s} {sl:4s}: FAR {np.sum([x[0]*x[1] for x in t])/max(n,1):.3f}  n={n}')
t=[far(r[1],r[0],r[5]) for r in rows]; n=np.sum([x[1] for x in t]); t2=[far(r[1],r[0],~r[5]) for r in rows]; n2=np.sum([x[1] for x in t2])
print(f'  ERA5 when last 2 h power is flat at mid level (plateau): FAR {np.sum([x[0]*x[1] for x in t])/max(n,1):.3f} n={n};  otherwise {np.sum([x[0]*x[1] for x in t2])/max(n2,1):.3f} n={n2}')
# magnitude attenuation: actual / implied for true signals
ratio=[]; 
for d,iE,_,_,lv,_ in rows:
    m=np.abs(iE)>.3; ratio.append(np.sum(d[m]*np.sign(iE[m]))/np.sum(np.abs(iE[m])))
print(f'  On ERA5-implied |Δ|>0.3: realized change in that direction / implied magnitude = {np.mean(ratio):.3f}')

print('\n  State-aware conversions of the same ERA5 / Pangu weather (4 h):')
for src,k in [('ERA5',None),('Pangu',None)]:
    acc={'curve change':[], 'anchored+clipped (headroom)':[], 'level: f(future)-current power':[]}
    for s,(d,iE,iP,_,lv,_) in enumerate(rows):
        hl=h3[:,s]
        if src=='ERA5': fE=curve_fit(E2[:,s,1:].ravel(),y2[:,s].ravel()); fut=fE(E3[:,s,1+j]); imp=iE
        else: fP=curve_fit(P2[:,s,1:].ravel(),y2[:,s].ravel()); fut=fP(P3[:,s,1+j]); imp=iP
        cand={'curve change':imp,'anchored+clipped (headroom)':np.clip(hl+imp,0,1)-hl,'level: f(future)-current power':fut-hl}
        for nme,c in cand.items():
            sig=np.abs(c)>.15; ok=(np.abs(d)>.15)&(np.sign(d)==np.sign(c)); ev=np.abs(d)>.3
            acc[nme].append(((sig&~ok).sum(),sig.sum(),(hit:=(sig&ok&ev)).sum(),ev.sum(),np.corrcoef(c,d)[0,1]))
    for nme,v in acc.items():
        v=np.array(v); print(f'     {src:5s} {nme:32s} FAR {v[:,0].sum()/v[:,1].sum():.3f}  POD(|Δ|>.3) {v[:,2].sum()/v[:,3].sum():.3f}  corr {v[:,4].mean():.3f}')
