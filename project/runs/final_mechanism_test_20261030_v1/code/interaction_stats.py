from settings import *
import numpy as np
def interaction(data,backbone,seeds):
    cells=['H1W1','H1W0','H0W1','H0W0']
    farms=[i for i in range(56) if all((backbone,c,i,s) in data for c in cells for s in seeds)]
    result=dict(backbone=backbone,seeds=seeds,paired_farms=len(farms),complete=len(farms)==56,passed=False)
    if len(farms)!=56:return result
    rng=np.random.default_rng(301030);indices=rng.integers(0,56,(4000,56));measures={}
    for horizon in ['all','0_1h','1_3h','3_6h']:
        measures[horizon]={}
        for metric in ['mae','rmse']:
            def value(c,i,s):
                m=data[backbone,c,i,s]
                return (m if horizon=='all' else m['horizons'][horizon])[metric]
            bycell={c:np.array([[value(c,i,s) for s in seeds] for i in farms]) for c in cells}
            gain_real=bycell['H1W0']-bycell['H1W1'];gain_fixed=bycell['H0W0']-bycell['H0W1'];effect=gain_real-gain_fixed
            byfarm=effect.mean(1);den=float(bycell['H1W1'].mean());inter=float(effect.mean())
            measures[horizon][metric]=dict(cells={c:float(x.mean()) for c,x in bycell.items()},weather_gain_real_history=float(gain_real.mean()),weather_gain_fixed_history=float(gain_fixed.mean()),gain_ratio=float(gain_real.mean()/gain_fixed.mean()) if abs(gain_fixed.mean())>1e-12 else None,interaction=inter,interaction_relative_to_E11=inter/den,positive_farms=int((byfarm>0).sum()),positive_seed_macros=int((effect.mean(0)>0).sum()),per_farm_interaction=byfarm.tolist(),paired_farm_bootstrap95=np.quantile(byfarm[indices].mean(1),[.025,.975]).tolist())
    result['measures']=measures
    result['passed']=all(measures['all'][m]['interaction_relative_to_E11']>=.001 and measures['all'][m]['positive_farms']>=29 and measures['all'][m]['weather_gain_real_history']>0 for m in ['mae','rmse'])
    return result

