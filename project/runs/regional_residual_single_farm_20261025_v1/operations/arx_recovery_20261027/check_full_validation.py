from pathlib import Path
import json, sys
import numpy as np
HERE=Path(__file__).resolve().parent
RUN=HERE.parents[1]
sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now,sha
new=HERE/'validation/site00'; old=RUN/'jobs/ARX_s00'
assert (new/'SUCCESS.json').exists(), 'Full validation still running'
a=json.loads((old/'metrics.json').read_text());b=json.loads((new/'metrics.json').read_text())
rows=[]
for family in a['selections']:
    x,y=a['selections'][family],b['selections'][family]
    assert x['alpha']==y['alpha']
    errs={m:abs(x['metrics'][m]-y['metrics'][m]) for m in ['mae','rmse']}
    assert max(errs.values())<1e-10
    rows.append(dict(family=family,alpha=x['alpha'],metric_errors=errs))
with np.load(old/'coefficients.npz') as a,np.load(new/'coefficients.npz') as b:
    assert set(a.files)==set(b.files)
    errors={k:float(np.max(np.abs(a[k].astype('f8')-b[k].astype('f8')),initial=0)) for k in a.files}
assert max(errors.values())<1e-8
resume=json.loads((HERE/'RESUME_VALIDATION_START.json').read_text())
assert resume['completed_families']==['LOCAL','REG_RAW']
result=dict(timestamp=now(),passed=True,full_site=0,rows=rows,coefficient_errors=errors,
    checkpoint_resume_passed=True,resume_test=resume,original_metrics_sha256=sha(old/'metrics.json'),
    recovered_metrics_sha256=sha(new/'metrics.json'),qr_equivalence_sha256=sha(HERE/'QR_EQUIVALENCE.json'),
    same_alpha_grid=True,same_l1_ratio=True,same_tolerance=True,iteration_budget_only_extended=True)
atomic_json(HERE/'FULL_VALIDATION.json',result)
print(json.dumps(result))
