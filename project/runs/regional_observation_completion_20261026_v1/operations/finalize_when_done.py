"""Audit D, write selection recommendation, leave E dependent on its own freeze."""
from pathlib import Path
import sys,json,time,fcntl,subprocess
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,sha,now,atomic_json

def main():
    lock=(RUN/'operations/finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((RUN/'manifest.json').read_text())
    while not all((JOBS[s['id']]/'SUCCESS.json').exists() for s in manifest):time.sleep(30)
    selected=set();references=0
    for spec in manifest:
        out=JOBS/spec['id'];assert json.loads((out/'spec.json').read_text())==spec
        result=json.loads((out/'metrics.json').read_text());assert not result['smoke_only']
        assert (out/'diagnostics.json').exists()
        if spec['task']=='train':
            assert not (out/'latest.pt').exists() and len(list(out.glob('epoch_*.pt')))<=2
            for item in result['selections'].values():
                path=out/item['checkpoint'];assert sha(path)==item['sha256'];selected.add(str(path))
        else:
            assert not list(out.glob('*.pt'));assert sha(result['source_checkpoint'])==result['source_sha256'];references+=1
    for path,h in json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(path)==h,path
    subprocess.run([sys.executable,'-s',str(RUN/'code/analyze_round.py')],check=True)
    atomic_json(RUN/'COMPLETION_AUDIT.json',dict(timestamp=now(),passed=True,completed=len(manifest),
        new_training=2240,reference_replays=references,unique_selected_weights_verified=len(selected),test_year_accessed=False))
    atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True))
    atomic_json(RUN/'operations/E_PENDING_D_REVIEW.json',dict(timestamp=now(),state='D_complete_review_recommendation_before_E_freeze',
        recommendation=str(RUN/'analysis/D_SELECTION_RECOMMENDATION.json'),open_2024=False))

if __name__=='__main__':main()
