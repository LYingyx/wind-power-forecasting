"""Freeze the new D round while retaining every prior scientific hash."""
from pathlib import Path
import sys,json
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
assert not (RUN/'SOURCE_FREEZE.json').exists()
checks=json.loads((RUN/'operations/CHECKS.json').read_text());assert checks['passed']
old=json.loads((RESIDUAL_RUN/'SOURCE_FREEZE.json').read_text())['files']
for p,h in old.items():assert sha(p)==h,p
files=dict(old)
paths=list((RUN/'code').glob('*.py'))+list((RUN/'operations').glob('*.py'))+list(ASSETS.glob('*'))
paths += [RUN/'manifest.json',RUN/'PROTOCOL.json',RUN/'README_CN.md',RUN/'analysis/train_only_bins.json',
          RUN/'operations/HOST_POLICY.json',RUN/'operations/CHECKS.json',RUN/'operations/compatibility_fixture.pt']
for p in paths:
    if p.is_file():files[str(p)]=sha(p)
atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files=files,prior_frozen_files=len(old),
    new_science_files=len(files)-len(old),screen_only=True,opens_2024=False))
atomic_json(ATTRIBUTION_RUN/'operations/B1_DEFERRED_REGIONAL_COMPLETION.json',dict(timestamp=now(),
    status='E_dependent_on_D_selection_and_final_freeze',screen_run=str(RUN),opens_2024=False))
print('Frozen',len(files),'files; D only')
