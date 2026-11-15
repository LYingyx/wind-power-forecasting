"""Preflight without evaluating2024 or reading test targets."""
from pathlib import Path
import sys,json,compileall
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *

def run():
    assert compileall.compile_dir(RUN/'code',quiet=1) and compileall.compile_dir(RUN/'operations',quiet=1)
    parent_checks=json.loads((PARENT/'operations/CHECKS.json').read_text())
    assert parent_checks['passed']
    for name in ['physical_models.py','operational_farm.py','diagnostics.py','neighbor_inputs.py']:
        assert sha(RUN/'code'/name)==sha(PARENT/'code'/name)
    candidates=json.loads((RUN/'candidate_manifest.json').read_text())
    parent={s['id']:s for s in json.loads((PARENT/'manifest.json').read_text())}
    assert len(candidates)==4480
    for s in candidates:
        q=dict(s);q['id']=q.pop('source_job');q['task']='train';assert q==parent[q['id']]
    assert not (RUN/'TEST_FREEZE.json').exists()
    from physical_data import FarmData
    try:FarmData('test',0,'R3_RAW',limit=1)
    except RuntimeError:pass
    else:raise AssertionError('Premature2024 access')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,parent_checks_sha256=sha(PARENT/'operations/CHECKS.json'),checks=['same audited model and data implementation','all evaluation candidates map exactly to parent training specifications','2024 access rejected before selection freeze'],test_evaluated=False))
    print('Follow-up preflight passed')

if __name__=='__main__':run()
