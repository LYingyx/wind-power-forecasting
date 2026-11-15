"""Audit the new eight seeds, combine references, never start a test-year run."""
from pathlib import Path
import sys,json,time,fcntl,subprocess
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,sha,now,atomic_json

def main():
    lock=(RUN/'operations/finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((RUN/'manifest.json').read_text());assert manifest
    while not all((JOBS/s['id']/'SUCCESS.json').exists() for s in manifest):time.sleep(30)
    activation=json.loads((RUN/'ACTIVATION.json').read_text())
    assert sha(RUN/'manifest.json')==activation['manifest_sha256']
    assert sha(RUN/'REUSED_SOURCES.json')==activation['reused_sources_sha256']
    weights=set()
    for s in manifest:
        out=JOBS/s['id'];assert json.loads((out/'spec.json').read_text())==s
        result=json.loads((out/'metrics.json').read_text());assert not result['smoke_only']
        assert (out/'diagnostics.json').exists() and not (out/'latest.pt').exists()
        assert len(list(out.glob('epoch_*.pt')))<=2
        for sel in result['selections'].values():
            path=out/sel['checkpoint'];assert sha(path)==sel['sha256'];weights.add(str(path))
    for p,h in json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(p)==h,p
    for p,h in json.loads((RUN/'operations/OPERATIONAL_OVERLAY_V2.json').read_text())['files'].items():assert sha(p)==h,p
    for p,h in json.loads((RUN/'operations/OPERATIONAL_OVERLAY_V3.json').read_text())['files'].items():assert sha(p)==h,p
    subprocess.run([sys.executable,'-s',str(RUN/'code/analyze_extension.py')],check=True)
    atomic_json(RUN/'COMPLETION_AUDIT.json',dict(timestamp=now(),passed=True,new_seed_fits=len(manifest),
        selected_weights_verified=len(weights),open_2024=False,report=str(RUN/'analysis/REPORT_CN.md')))
    atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True))

if __name__=='__main__':main()
