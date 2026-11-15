"""Independent source-level validation of the delivered seedwise tables."""
import json, re, math
from pathlib import Path
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parents[1];R=O.parent;T=O/'tables'
read=lambda name:pd.read_csv(T/name,dtype={'station_id':str},float_precision='round_trip',low_memory=False)
actual=read('all_trained_results.csv')
checked=0
for job,g in actual.groupby('source_job_id',sort=False):
    m=json.loads((R/'jobs'/job/'metrics.json').read_text())
    for r in g.itertuples():
        assert r.family==m['family'] and r.site==m['site'] and r.seed==m['seed']
        assert r.width==m['config']['width'] and r.lr==m['config']['lr'] and r.weight_decay==m['config']['weight_decay']
        for key in ['mae','rmse','bias','n_origins','outside_0_1_fraction']:assert getattr(r,key)==m[r.split][key],(job,r.split,key)
        checked+=1
assert checked==18460
pairs=read('paired_results.csv')
for met in ['mae','rmse']:
    expected=pairs[met+'_reference']-pairs[met+'_candidate']
    assert np.array_equal(expected.to_numpy(),pairs[met+'_gain'].to_numpy())
    signs=np.select([expected>0,expected<0],['improved','worsened'],default='tie')
    assert np.array_equal(signs,pairs[met+'_direction'])
counts=read('improvement_counts_all_pairs.csv')
expected_mae={'D01':(271,9),'D02':(170,110),'D03':(168,112),'D04':(139,141),'D05':(122,158),'D06':(158,122),'D07':(176,104),'D08':(187,93),'D09':(271,289),'D10':(289,271),'B02':(345,215)}
for cid,wl in expected_mae.items():
    r=counts[(counts.comparison_id==cid)&(counts.year==2024)&(counts.metric=='mae')].iloc[0]
    assert (r.improved,r.worsened)==wl and r.ties==0

view=read('formal_and_ablation_results.csv')
index=actual.set_index(['source_job_id','split'])
e10=view[view.family=='SELECTED_BASELINE']
for r in e10.itertuples():
    source=index.loc[r.source_job_id,r.split]
    assert source.family==r.actual_family and r.mae==source.mae and r.rmse==source.rmse
assert len(e10)==1120

sample_ids=['formal_E2C_CBH_s06_r42','ablation_DIRECT_s06_r46','reference_B0_s00_r51','tune_E2C_CBH_c0_s06_r40']
detail_checks={}
for name,axis in [('lead_results.csv.gz','lead_step'),('scenario_results.csv.gz','scenario'),('daily_results_utc.csv.gz','utc_day')]:
    n=0
    for chunk in pd.read_csv(T/name,chunksize=100000,float_precision='round_trip'):
        for r in chunk[chunk.source_job_id.isin(sample_ids)].itertuples():
            m=json.loads((R/'jobs'/r.source_job_id/'metrics.json').read_text())[r.split]
            if axis=='lead_step':a,b=m['lead_mae'][r.lead_step-1],m['lead_rmse'][r.lead_step-1]
            elif axis=='scenario':a,b=m['scenarios'][r.scenario]['mae'],m['scenarios'][r.scenario]['rmse']
            else:
                z=m['daily_utc'][str(r.utc_day)];a=z['abs_sum']/(z['n']*24);b=math.sqrt(z['sq_sum']/(z['n']*24))
                assert r.abs_error_sum==z['abs_sum'] and r.squared_error_sum==z['sq_sum'] and r.n_origins==z['n']
            assert r.mae==a and r.rmse==b,(name,r.source_job_id)
            n+=1
    assert n>0;detail_checks[name]=n

cases=read('forecast_cases_per_seed.csv.gz');n=0
for (job,site),g in cases.groupby(['source_job_id','site']):
    y=np.load(R/f'prepared/test/{site:02d}/target.npy',mmap_mode='r')
    if job.startswith('persistence'):
        h=np.load(R/f'prepared/test/{site:02d}/history.npy',mmap_mode='r');p=None
    else:p=np.load(R/'jobs'/job/'test_prediction.npy',mmap_mode='r')
    for r in g.itertuples():
        assert r.observed==float(y[r.origin_index,r.lead_step-1])
        assert r.prediction==float(h[r.origin_index,-1] if p is None else p[r.origin_index,r.lead_step-1])
        n+=1
assert n==65664
# Check relative Markdown links; absolute project links remain valid in this workspace.
bad=[]
for path in [O/'REPORT_CN.md',O/'ARCHITECTURES_CN.md',O/'legacy/README_CN.md',*list((O/'farms').glob('*.md'))]:
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if not target.startswith(('http:','https:','#')) and not (path.parent/target.split('#')[0]).exists():bad.append((str(path),target))
assert not bad,bad
out=dict(passed=True,all_source_metric_rows_checked=checked,all_selected_source_rows_checked=len(e10),all_pair_differences_and_signs_checked=len(pairs),detailed_source_sample_rows=detail_checks,all_case_predictions_checked=n,broken_markdown_links=bad)
(O/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
