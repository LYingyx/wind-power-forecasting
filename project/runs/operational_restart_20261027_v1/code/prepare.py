from settings import *
import numpy as np,time
from operational import interpolate_nodes

def month(t):return (t+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12

def moments(p,weights):
    w=weights/max(weights.sum(),1e-12);m=(p*w[None,:,None]).sum(1)
    sd=np.sqrt(np.maximum(((p[:,:,-1]-m[:,-1:])**2*w).sum(1),0))
    return np.stack([m[:,-1],sd,m[:,-1]-m[:,-5],np.full(len(p),float(weights.sum()>0))],-1).astype('f4')

def mapped_weather(split,site,queries):
    root=CACHE/split;origin=np.load(root/'origin_unix.npy');ci=np.load(root/'weather_cycle_index.npy')
    future=np.load(BUNDLE/f'local_weather/{split}.npy',mmap_mode='r')[ci,site]
    past=np.load(PREPARED/split/'past_weather.npy',mmap_mode='r')[ci,site]
    t=np.concatenate([(np.load(PREPARED/split/'past_valid_unix.npy')[ci]-origin[:,None])/3600,np.load(root/'weather_time_offset_hours.npy')],1)
    w=interpolate_nodes(np.concatenate([past,future],1),t,np.asarray(queries))
    w=w*np.load(CACHE/'static/nwp_std.npy')+np.load(CACHE/'static/nwp_mean.npy')
    return w

def prepare():
    setup();start=time.time()
    # Global E2C moments use new training-issued weather only.
    w=np.load(BUNDLE/'local_weather/train.npy',mmap_mode='r');mu=np.load(CACHE/'static/nwp_mean.npy');sd=np.load(CACHE/'static/nwp_std.npy');elev=np.load(PREPARED/'elevation.npy')
    total=np.zeros(15,'f8');squares=total.copy();count=0
    for site in range(56):
        p=np.asarray(w[:,site],dtype='f8')*sd+mu;e=np.full(p.shape[:-1],elev[site]);g=9.80665;k=.2854
        v=np.stack([p[...,0],p[...,1],p[...,2],p[...,3],e,p[...,10]-p[...,1],p[...,12]-p[...,2],p[...,8],p[...,6],p[...,4]/g-e,p[...,11]-p[...,10],p[...,13]-p[...,12],p[...,9]*(1000/925)**k-p[...,8],p[...,7]-p[...,6],(p[...,5]-p[...,4])/g],-1).reshape(-1,15)
        total+=v.sum(0);squares+=(v*v).sum(0);count+=len(v)
    cm=total/count;cs=np.maximum(squares/count-cm*cm,1e-8)**.5
    atomic_json(ASSETS/'column_stats.json',dict(mean=cm.tolist(),std=cs.tolist(),station_ids=np.load(CACHE/'static/station_id.npy').astype(str).tolist(),weather_channels=np.load(CACHE/'static/nwp_channel_name.npy').astype(str).tolist(),source='new operational train local weather',fingerprint=FINGERPRINT))
    topo=np.load(ASSETS/'topology.npz');origins={s:np.load(CACHE/s/'origin_unix.npy') for s in ['train','val']}
    tr=origins['train'];tr_month=month(tr);tr_slot=((tr+28800)%86400)//900;tr_day=(tr+28800)//86400
    for split,t in origins.items():
        m=month(t);slot=((t+28800)%86400)//900;day=(t+28800)//86400
        for seed in SEEDS:
            rng=np.random.default_rng(seed+20261027);donor=np.empty(len(t),'i8')
            for mm in np.unique(m):
                for ss in np.unique(slot):
                    ix=np.flatnonzero((m==mm)&(slot==ss));candidates=np.flatnonzero((tr_month==mm)&(tr_slot==ss))
                    assert len(candidates)>1
                    for i in ix:
                        good=candidates[abs(tr_day[candidates]-day[i])>=2]
                        assert len(good);donor[i]=rng.choice(good)
            assert np.all(month(tr[donor])==m) and np.all((tr[donor]+28800)%86400==(t+28800)%86400)
            np.save(ASSETS/f'mismatch_{split}_r{seed}.npy',donor)
        raw=np.load(CACHE/split/'station_history.npy',mmap_mode='r')
        out=np.lib.format.open_memmap(ASSETS/f'rings_{split}.npy',mode='w+',dtype='f4',shape=(len(t),56,12))
        for site in range(56):
            ids=topo['ids'][site];dist=topo['geometry'][site,:,2];cap=topo['capacity'][site];valid=topo['mask'][site]
            p=np.asarray(raw[:,ids,1:23]);features=[]
            for lo,hi in [(0,20),(20,60),(60,150.0001)]:features.append(moments(p,cap*valid*(dist>=lo)*(dist<hi)))
            out[:,site]=np.concatenate(features,-1)
        out.flush();print('P0 regional assets',split,flush=True)
    atomic_json(ASSETS/'P0_READY.json',dict(timestamp=now(),seconds=time.time()-start,fingerprint=FINGERPRINT,source_splits=['train','val']))
    p1()

def p1():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge
    origins={s:np.load(CACHE/s/'origin_unix.npy') for s in ['train','val']}
    hist={s:np.load(CACHE/s/'station_history.npy',mmap_mode='r') for s in origins}
    target={s:np.load(CACHE/s/'station_future.npy',mmap_mode='r') for s in origins}
    errors={s:np.lib.format.open_memmap(ASSETS/f'error_history_{s}.npy',mode='w+',dtype='f4',shape=hist[s].shape) for s in origins}
    future={s:np.lib.format.open_memmap(ASSETS/f'mapped_future6_{s}.npy',mode='w+',dtype='f4',shape=(len(origins[s]),56,6)) for s in origins}
    curves={};diagnostics=[]
    for site in range(56):
        speed={};hw={};fw={}
        for s in origins:
            w=mapped_weather(s,site,np.arange(-23,1)/4);hw[s]=np.hypot(w[...,1],w[...,2])
            w=mapped_weather(s,site,np.arange(1,7));fw[s]=np.hypot(w[...,1],w[...,2])
        t=origins['train'];m=month(t);y=np.asarray(hist['train'][:,site,-1]);x=hw['train'][:,-1]
        def fit(mask):
            model=IsotonicRegression(out_of_bounds='clip');model.fit(x[mask],y[mask]);return dict(x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())
        full=fit(np.ones(len(t),bool));oof={}
        # Month block plus 24h padding prevents rolling histories crossing fit folds.
        for mm in range(12):
            held=t[m==mm];keep=(t<held.min()-86400)|(t>held.max()+86400);oof[str(mm)]=fit(keep)
        curves[str(site)]=dict(final=full,oof=oof)
        def mapping(v,valid,s):
            if s=='val':return np.interp(v,full['x'],full['y']).astype('f4')
            z=np.empty(v.shape,'f4');months=month(valid)
            for mm in np.unique(months):
                mask=months==mm;c=oof[str(mm)];z[mask]=np.interp(v[mask],c['x'],c['y'])
            return z
        for s in origins:
            mapped=mapping(hw[s],origins[s][:,None]+np.arange(-23,1)*900,s)
            errors[s][:,site]=hist[s][:,site]-mapped
            future[s][:,site]=mapping(fw[s],origins[s][:,None]+np.arange(1,7)*3600,s)
        diagnostics.append(dict(site=site,correlations=[float(np.corrcoef(errors['val'][:,site,-4:].mean(1),target['val'][:,site,4*j+3]-future['val'][:,site,j])[0,1]) for j in range(6)]))
        print('P1 same-issue OOF mapping',site,flush=True)
    for a in [*errors.values(),*future.values()]:a.flush()
    atomic_json(ASSETS/'curves.json',dict(stations=curves,sign='observed minus mapped power',fit='2022 only; train residuals exclude observation month plus 24h',fingerprint=FINGERPRINT))
    topo=np.load(ASSETS/'topology.npz');er={}
    for s,t in origins.items():
        er[s]=np.lib.format.open_memmap(ASSETS/f'error_rings_{s}.npy',mode='w+',dtype='f4',shape=(len(t),56,12))
        for site in range(56):
            ids=topo['ids'][site];dist=topo['geometry'][site,:,2];cap=topo['capacity'][site];valid=topo['mask'][site];p=np.asarray(errors[s][:,ids,1:23])
            er[s][:,site]=np.concatenate([moments(p,cap*valid*(dist>=lo)*(dist<hi)) for lo,hi in [(0,20),(20,60),(60,150.0001)]],-1)
        er[s].flush()
    for site,d in enumerate(diagnostics):
        xx={};yy={}
        for s,t in origins.items():
            age=(t-np.load(CACHE/s/'origin_issue_unix.npy'))/86400
            xx[s]=np.concatenate([hist[s][:,site],future[s][:,site],age[:,None]],1)
            yy[s]=np.asarray(target[s][:,site,3::4])
        def predict(extra):
            a=np.concatenate([xx['train'],*extra.get('train',[])],1);b=np.concatenate([xx['val'],*extra.get('val',[])],1)
            mu=a.mean(0);sd=a.std(0).clip(1e-4);model=Ridge(alpha=100.).fit((a-mu)/sd,yy['train']);return model.predict((b-mu)/sd)
        base=predict({});own=predict({s:[errors[s][:,site,-12:]] for s in origins});region=predict({s:[errors[s][:,site,-12:],er[s][:,site]] for s in origins})
        def score(p):e=p-yy['val'];return dict(mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())))
        raw=predict({s:[np.load(ASSETS/f'rings_{s}.npy',mmap_mode='r')[:,site]] for s in origins})
        d.update(base=score(base),own=score(own),region=score(region),raw=score(raw))
    atomic_json(RUN/'analysis/P1_DIAGNOSTIC.json',dict(timestamp=now(),rows=diagnostics,model='ridge alpha100 fixed, fit2022/evaluate2023; baseline includes own24history,6mappedNWP,forecastage; separate raw3rings comparator',warning='power mapping residual is not pure meteorological error; correlation alone is not an information lower bound'))
    def gate(a,b):
        effects={k:float(np.mean([r[b][k] for r in diagnostics])/np.mean([r[a][k] for r in diagnostics])-1) for k in ['mae','rmse']}
        wins={k:sum(r[b][k]<r[a][k] for r in diagnostics) for k in effects}
        corr=float(np.nanmedian([np.mean(r['correlations']) for r in diagnostics]))
        return all(effects[k]<=-.001 and wins[k]>28 for k in effects) and corr>=.05,dict(effects=effects,wins=wins,median_mean_error_correlation=corr)
    own,od=gate('base','own');regional,rd=gate('own','region');rawpass,rawdetail=gate('raw','region')
    atomic_json(RUN/'analysis/P1_GATE.json',dict(timestamp=now(),own=own,region=own and regional and rawpass,own_evidence=od,regional_evidence=rd,regional_vs_raw=rawdetail,criteria='conditional ridge both errors >=0.1% lower; >28 farms each; median error correlation >=0.05',selection_year=2023,test_evaluated=False))
    atomic_json(ASSETS/'READY.json',dict(timestamp=now(),fingerprint=FINGERPRINT))
if __name__=='__main__':prepare()
