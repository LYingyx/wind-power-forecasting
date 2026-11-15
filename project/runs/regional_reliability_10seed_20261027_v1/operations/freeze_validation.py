"""Freeze new science and operational code, preserving prior source hashes."""
from pathlib import Path
import sys,json
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *

def freeze():
    assert not (RUN/'SOURCE_FREEZE.json').exists()
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    assert json.loads((RUN/'operations/REPORTING_CHECKS.json').read_text())['passed']
    prior=json.loads((TEN_RUN/'SOURCE_FREEZE.json').read_text())['files']
    files={}
    for path,h in prior.items():
        assert sha(path)==h,path
        files[path]=h
    paths=list((RUN/'code').glob('*.py'))+list((RUN/'operations').glob('*.py'))+list(ASSETS.glob('*'))
    paths += [RUN/n for n in ['manifest.json','PROTOCOL.json','SELECTION_RULES.json','candidate_extension_manifest.json','REUSED_SCREEN.json','ACTIVATION.json','README_CN.md'] if (RUN/n).exists()]
    paths += [RUN/'analysis/train_only_bins.json',RUN/'operations/HOST_POLICY.json',RUN/'operations/CHECKS.json',RUN/'operations/compatibility_fixture.pt']
    paths += [RUN/'operations/REPORTING_CHECKS.json']
    for p in paths:
        if p.is_file():files[str(p)]=sha(p)
    atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files=files,prior_frozen_files=len(prior),opens_2024=False))
    print('Frozen',len(files),'files',flush=True)

if __name__=='__main__':freeze()
