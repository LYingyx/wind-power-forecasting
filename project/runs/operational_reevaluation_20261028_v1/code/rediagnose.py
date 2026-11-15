from settings import *
import numpy as np
from sklearn.linear_model import Ridge
origins={s:np.load(CACHE/s/'origin_unix.npy') for s in ['train','val']}
hist={s:np.load(CACHE/s/'station_history.npy',mmap_mode='r') for s in origins}
target={s:np.load(CACHE/s/'station_future.npy',mmap_mode='r') for s in origins}
errors={s:np.load(ASSETS/f'error_history_{s}.npy',mmap_mode='r') for s in origins}
future={s:np.load(ASSETS/f'mapped_future6_{s}.npy',mmap_mode='r') for s in origins}
er={s:np.load(ASSETS/f'error_rings_{s}.npy',mmap_mode='r') for s in origins}
prior=json.loads((RUN/'analysis/P1_DIAGNOSTIC.json').read_text())
atomic_json(RUN/'analysis/P1_WITH_RAW_CONDITIONING.json',prior)
diagnostics=prior['rows']
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
