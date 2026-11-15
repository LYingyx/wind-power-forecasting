"""Freeze neighbor screen code, topology, surrogate indices, and 2022 bins."""
from settings import *
def main():
    assert json.loads((RUN/'CHECKS.json').read_text())['passed']
    inherited=json.loads((ATTRIBUTION_RUN/'SOURCE_FREEZE.json').read_text())['files']
    for p,h in inherited.items():assert sha(p)==h,p
    files=list((RUN/'code').glob('*.py'))+list(ASSETS.glob('*'))+[RUN/'manifest.json',RUN/'PROTOCOL.json',RUN/'analysis/train_only_bins.json']
    hashes={**inherited,**{str(p):sha(p) for p in files if p.is_file()}}
    atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files=hashes,purpose='Neighbor 2023-only screen; not revised 2024 permission'))
    print('frozen',len(hashes))
if __name__=='__main__':main()
