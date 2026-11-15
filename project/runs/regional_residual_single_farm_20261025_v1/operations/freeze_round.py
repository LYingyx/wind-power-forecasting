"""Freeze science sources/assets after checks; old freezes remain immutable."""
from pathlib import Path
import sys,json
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
assert not (RUN/'SOURCE_FREEZE.json').exists(),'Do not silently overwrite a freeze'
checks=json.loads((RUN/'operations/CHECKS.json').read_text());assert checks['passed']
old=json.loads((NEIGHBOR_RUN/'SOURCE_FREEZE.json').read_text())['files']
for path,h in old.items():assert sha(path)==h
files=dict(old)
paths=list((RUN/'code').glob('*.py'))+list(ASSETS.glob('*'))+list((RUN/'operations').glob('*.py'))
paths += [RUN/'manifest.json',RUN/'PROTOCOL.json',RUN/'README_CN.md',RUN/'analysis/train_only_bins.json',RUN/'operations/HOST_POLICY.json',RUN/'operations/CHECKS.json',RUN/'operations/compatibility_fixture.pt']
for p in paths:
    if p.is_file():files[str(p)]=sha(p)
atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files=files,prior_frozen_files=len(old),new_science_files=len(files)-len(old),screen_only=True,opens_2024=False))
previous=ATTRIBUTION_RUN/'FROZEN_SELECTION.json'
atomic_json(ATTRIBUTION_RUN/'operations/B1_DEFERRED_REGIONAL_RESIDUAL.json',dict(timestamp=now(),status='B1_still_not_started',previous_freeze=str(previous),previous_freeze_sha256=sha(previous),screen_run=str(RUN),reason='User authorized residual screening; no 2024 evaluation before a separate review/freeze',new_training=2352,reference_replays=560,arx_jobs=56))
print('Frozen',len(files),'files')
