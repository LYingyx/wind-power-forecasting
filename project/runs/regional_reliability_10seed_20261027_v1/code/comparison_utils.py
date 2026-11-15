"""Paired, equal-farm effects and predeclared dual-metric gates; no 2024 launch."""
from settings import *
from collect_physics import collect
import pandas as pd,numpy as np
TF='TF_z1_lr0.0002_dp0.05'

def effect_pairs(frame,edges,keys):
    pairs=[];effects=[]
    for cc,cv,rc,rv in edges:
        a=frame[(frame.configuration==cc)&(frame.variant==cv)];b=frame[(frame.configuration==rc)&(frame.variant==rv)]
        if a.empty or b.empty:continue
        pair=a.merge(b,on=['site','seed',*keys],suffixes=('_candidate','_reference'),validate='one_to_one')
        if pair.empty:continue
        assert (pair.n_candidate==pair.n_reference).all()
        for metric in ['mae','rmse']:pair[metric+'_difference']=pair[metric+'_candidate']-pair[metric+'_reference']
        pairs.append(pair)
        for values,rows in pair.groupby(keys,dropna=False):
            if not isinstance(values,tuple):values=(values,)
            complete=rows.groupby('site').seed.nunique();rows=rows[rows.site.isin(complete[complete==(1 if cc=='ARX' else 2)].index)]
            if rows.empty:continue
            for metric in ['mae','rmse']:
                farm=rows.groupby('site')[[metric+'_candidate',metric+'_reference']].mean();ca=farm[metric+'_candidate'];ref=farm[metric+'_reference']
                effects.append(dict(candidate_configuration=cc,candidate_variant=cv,reference_configuration=rc,reference_variant=rv,**dict(zip(keys,values)),metric=metric,farms=len(farm),pairs=len(rows),candidate=float(ca.mean()),reference=float(ref.mean()),effect_percent=float(100*(ca.mean()/ref.mean()-1)),improved_farms=int((ca<ref).sum()),improved_seed42=int((rows[rows.seed==42][metric+'_difference']<0).sum()),improved_seed43=int((rows[rows.seed==43][metric+'_difference']<0).sum())))
    return pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame(),pd.DataFrame(effects)

