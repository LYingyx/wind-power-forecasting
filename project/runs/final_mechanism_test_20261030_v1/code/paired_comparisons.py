from settings import *
import numpy as np
def comparison(data,base,arm):
    sites=[i for i in range(56) if all((c,i,s) in data for c in [base,arm] for s in SEEDS)]
    result=dict(base=base,arm=arm,paired_farms=len(sites),seeds_per_farm=10,passed=False)
    if len(sites)!=56:return result
    rng=np.random.default_rng(281028);boot=rng.integers(0,56,(4000,56))
    for metric in ['mae','rmse']:
        a=np.array([[data[base,i,s][metric] for s in SEEDS] for i in sites])
        b=np.array([[data[arm,i,s][metric] for s in SEEDS] for i in sites])
        am=a.mean(1);bm=b.mean(1)
        rel=float(bm.mean()/am.mean()-1)
        ci=np.quantile(bm[boot].mean(1)/am[boot].mean(1)-1,[.025,.975]).tolist()
        result[metric]=dict(base=float(am.mean()),arm=float(bm.mean()),relative_change=rel,farm_wins=int((bm<am).sum()),paired_farm_bootstrap95=ci,per_seed_macro_relative_change=(b.mean(0)/a.mean(0)-1).tolist())
    def macro_high(c):
        values=[data[c,i,s]['high_rmse'] for i in sites for s in SEEDS]
        if any(x is None for x in values):return None
        return float(np.mean(values))
    ha,hb=macro_high(base),macro_high(arm)
    result['high_power_rmse_relative_change']=hb/ha-1 if ha is not None and hb is not None and ha>0 else None
    result['short_0_1h_relative_change']={k:float(np.mean([data[arm,i,s]['horizons']['0_1h'][k] for i in sites for s in SEEDS])/np.mean([data[base,i,s]['horizons']['0_1h'][k] for i in sites for s in SEEDS])-1) for k in ['mae','rmse']}
    result['passed']=bool(all(result[k]['relative_change']<=-.001 and result[k]['farm_wins']>=29 for k in ['mae','rmse']) and result['high_power_rmse_relative_change'] is not None and result['high_power_rmse_relative_change']<=.005 and (base!='W0_H32' or max(result['short_0_1h_relative_change'].values())<=.001))
    return result

