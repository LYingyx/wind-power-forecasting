"""Explain the old E2C confidence interval by reproducing its equal-week estimator."""
import json
import numpy as np
import pandas as pd
from audit_tables import ROOT,OUT,ART,save

def main():
    frame=pd.read_csv(OUT/'tables/prediction_week_station_seed.csv',dtype={'station':str})
    old=json.loads((ART/'column_weather_20260928_v1/analysis/FINAL_WEATHER_ENCODER_DECISION.json').read_text())
    rows=[];details=[]
    for split in ['val','test']:
        d=frame[frame.split.eq(split)&frame.model.isin(['E0','E2C'])]
        w=d.pivot(index=['seed','station','block'],columns='model',values='mae').sort_index()
        delta=(w.E2C-w.E0).to_numpy().reshape(10,56,-1).transpose(0,2,1)
        counts=d.groupby('block').origins.first().sort_index().to_numpy(float)
        rng=np.random.default_rng(20260929+(split=='test'));draws=np.empty(5000)
        for i in range(5000):
            si=rng.integers(10,size=10);wi=rng.integers(len(counts),size=len(counts));fi=rng.integers(56,size=56)
            draws[i]=delta[np.ix_(si,wi,fi)].mean()
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(split=split,estimator='equal_weight_each_observed_week',point_estimate=float(delta.mean()),ci_low=lo,ci_high=hi,
            valid_origin_weighted_mean=float((delta.mean((0,2))*counts).sum()/counts.sum()),
            min_origins_per_week=int(counts.min()),max_origins_per_week=int(counts.max()),weeks=len(counts)))
        for b,count,value in zip(sorted(d.block.unique()),counts,delta.mean((0,2))):
            details.append(dict(split=split,block=b,origins=int(count),E2C_minus_E0=value))
    save(pd.DataFrame(rows),'equal_week_sensitivity')
    save(pd.DataFrame(details),'weekly_delta_and_coverage',False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
