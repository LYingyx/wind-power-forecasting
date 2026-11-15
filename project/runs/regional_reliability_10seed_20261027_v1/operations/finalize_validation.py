"""Durable completion audit and conditional eight-seed handoff."""
from pathlib import Path
import sys,json,time,fcntl,subprocess,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *

def main():
    lock=(RUN/'operations/finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=json.loads((RUN/'manifest.json').read_text())
    while not all((JOBS/s['id']/'SUCCESS.json').exists() for s in specs):
        failures=[s['id'] for s in specs if (JOBS/s['id']/'FAILURE.json').exists() and not (JOBS/s['id']/'SUCCESS.json').exists()]
        atomic_json(RUN/'operations/FINALIZER_STATUS.json',dict(timestamp=now(),state='waiting_for_all_success',failed=failures,completed=sum((JOBS/s['id']/'SUCCESS.json').exists() for s in specs),planned=len(specs)))
        time.sleep(30)
    weights=set()
    for s in specs:
        out=JOBS/s['id'];assert json.loads((out/'spec.json').read_text())==s
        m=json.loads((out/'metrics.json').read_text());assert not m['smoke_only']
        assert (out/'diagnostics.json').exists() and not (out/'latest.pt').exists()
        if s['task']=='train':
            assert len(list(out.glob('epoch_*.pt')))<=2
            for sel in m['selections'].values():
                assert sha(out/sel['checkpoint'])==sel['sha256'];weights.add(str(out/sel['checkpoint']))
        else:assert sha(m['source_checkpoint'])==m['source_sha256']
    for path,h in json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(path)==h,path
    subprocess.run([sys.executable,str(RUN/'code/analyze_validation.py')],check=True)
    atomic_json(RUN/'COMPLETION_AUDIT.json',dict(timestamp=now(),passed=True,tasks=len(specs),selected_weights_verified=len(weights),open_2024=False))
    atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True))
    if RUN.name==SCREEN_RUN.name:
        # Once complete, let dispatchers drain before starting the dependent round.
        while True:
            try:
                subprocess.run([sys.executable,str(RUN/'operations/activate_replication.py')],check=True)
                break
            except subprocess.CalledProcessError:
                atomic_json(RUN/'operations/REPLICATION_STATUS.json',dict(timestamp=now(),state='retryable_activation_error',error=traceback.format_exc()))
                time.sleep(30)

if __name__=='__main__':main()
