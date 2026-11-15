"""Persist the required diagnostic comparisons as soon as both groups are complete."""
from pathlib import Path
import json,sys,csv
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now,JOBS
from collect import records


def report():
    groups=records()
    for group in ['R0','TF_R3']:
        for seed in range(42,47):
            out=JOBS/f'REFERENCE_{group}_r{seed}'
            if (out/'SUCCESS.json').exists():groups[group][seed]=json.loads((out/'metrics.json').read_text())
    complete={g:values for g,values in groups.items() if set(values)==set(range(42,47)) and
              all('diagnostics' in v['metrics'] for v in values.values())}
    rows=[];diversity=[]
    def pooled(group,bin_name):
        bins=[complete[group][seed]['metrics']['diagnostics']['bins'][bin_name] for seed in range(42,47)]
        count=sum(v['count'] for v in bins)
        squared=sum(v['squared_error_sum'] for v in bins)
        cap=sum(v['capture_numerator'] for v in bins);den=sum(v['capture_denominator'] for v in bins)
        return count,squared/count if count else None,cap/den if den else None
    for group,values in complete.items():
        for ref in ['R0','TF_R3','TF_E2L','R2','R0+H','R3','R4-noS','R4-wrongS']:
            if group==ref or ref not in complete:continue
            for name in values[42]['metrics']['diagnostics']['bins']:
                n,x,cx=pooled(group,name);nr,y,cy=pooled(ref,name);assert n==nr
                rows.append(dict(candidate=group,reference=ref,bin=name,count=n,candidate_mse=x,reference_mse=y,
                                 delta_mse=x-y if n else None,mse_change_percent=100*(x/y-1) if n and y else None,
                                 candidate_capture_ratio=cx,reference_capture_ratio=cy))
        for seed,value in values.items():
            d=value['metrics']['diagnostics']
            if 'sample_std_mean' in d:
                diversity.append(dict(group=group,seed=seed,**{k:d[k] for k in ['sample_std_mean','nonzero_sample_std_fraction','sample_median_mae','sample_median_rmse']}))
    atomic_json(RUN/'analysis/DIAGNOSTICS.json',dict(timestamp=now(),complete_groups=sorted(complete),comparisons=rows,
                stochastic_diversity=diversity,note='Pooled per-seed diagnostic errors; not errors of an ensemble prediction. No selection by these diagnostics.'))
    if rows:
        with (RUN/'analysis/diagnostic_comparisons.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':report()
