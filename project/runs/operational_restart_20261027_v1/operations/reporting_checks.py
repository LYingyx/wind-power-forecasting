from pathlib import Path
import sys
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from report import comparison,interaction
from design import build
build()
# Effect sizes must use seed averages within matched farms; neither win counts
# nor a ratio-of-pairwise-percentage changes may substitute for this estimand.
d={}
for i in range(56):
 for s in [42,43]:
  a=.1+i*.002+(s-42)*.001
  for c,v in [('base',a),('good',a*.98),('bad_high',a*.98)]:
   d[c,i,s]=dict(mae=v,rmse=v*1.3,high_rmse=a*(1.02 if c=='bad_high' else (0.98 if c=='good' else 1)))
r=comparison(d,'base','good',[42,43]);assert r['passed'] and abs(r['relative_change']['mae']+.02)<1e-10 and r['farm_wins']['mae']==56
assert not comparison(d,'base','bad_high',[42,43])['passed']
d.pop(('good',55,43));assert not comparison(d,'base','good',[42,43])['passed']
for i in range(56):
 for s in [42,43]:
  for c,v in [('W0_SEL',.08),('W0_H1W0',.10),('W0_H0W1',.11),('W0_H0W0',.12)]:d[c,i,s]=dict(mae=v,rmse=v*1.5)
assert interaction(d,[42,43])['passed']
manifest=json.loads((RUN/'manifest.json').read_text());assert len({s['id'] for s in manifest})==len(manifest)==1568
atomic_json(RUN/'operations/REPORTING_CHECKS.json',dict(timestamp=now(),passed=True,paired_seed_farm_estimand=True,partial_pairs_not_accepted=True,high_power_veto=True,interaction_sign=True,manifest_unique=True))
print('Reporting checks passed')
