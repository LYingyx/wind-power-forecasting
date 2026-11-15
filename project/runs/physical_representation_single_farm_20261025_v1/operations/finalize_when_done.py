"""Finish the A audit and recommendation automatically; never read/open 2024."""
from pathlib import Path
import sys,json,time,fcntl,subprocess
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,PARENT_RUN,sha,now,atomic_json

def main():
    lock=(RUN/'operations/finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((RUN/'manifest.json').read_text())
    while not all((JOBS/s['id']/'SUCCESS.json').exists() for s in manifest):time.sleep(30)
    selected=set();diagnoses=0
    for spec in manifest:
        out=JOBS/spec['id'];assert json.loads((out/'spec.json').read_text())==spec
        m=json.loads((out/'metrics.json').read_text());assert not m['smoke_only']
        assert (out/'diagnostics.json').exists()
        if spec['task']=='train':
            assert not (out/'latest.pt').exists()
            for value in m['selections'].values():
                path=out/value['checkpoint'];assert sha(path)==value['sha256'];selected.add(str(path))
            assert len(list(out.glob('epoch_*.pt')))<=2
        else:
            assert not list(out.glob('*.pt'));assert sha(m['source_checkpoint'])==m['source_sha256'];diagnoses+=1
    for p,h in json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(p)==h
    subprocess.run([sys.executable,'-s',str(RUN/'code/analyze_round.py')],check=True)
    atomic_json(RUN/'COMPLETION_AUDIT.json',dict(timestamp=now(),passed=True,completed=len(manifest),new_training=len(manifest)-diagnoses,reference_diagnostics=diagnoses,unique_selected_weights_verified=len(selected),test_year_accessed=False,selection_frozen=False,next='Review SELECTION_RECOMMENDATION and freeze B comparison list.'))

if __name__=='__main__':main()
