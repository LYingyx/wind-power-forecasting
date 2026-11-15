"""Freeze A sources and train-only bins; never authorize 2024 with this file."""
from settings import *

def main():
    assert json.loads((RUN/'CHECKS.json').read_text())['passed']
    files=list((RUN/'code').glob('*.py'))+[RUN/'manifest.json',RUN/'PROTOCOL.json',RUN/'analysis/train_only_bins.json']
    inherited=json.loads((PARENT_RUN/'SOURCE_FREEZE.json').read_text())['files']
    for p,h in inherited.items():assert sha(p)==h
    hashes={**inherited,**{str(p):sha(p) for p in files}}
    atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files=hashes,purpose='A-only experiment/code freeze; not B model-selection permission'))
    print('frozen',len(hashes))

if __name__=='__main__':main()
