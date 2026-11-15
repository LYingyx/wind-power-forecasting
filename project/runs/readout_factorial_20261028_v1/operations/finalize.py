"""Export periodically; audit every checkpoint and freeze a complete2023 decision."""
from pathlib import Path
import sys,time,json,fcntl,traceback,os
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from collect_physics import collect
from factorial_report import run as report

def audit_complete(result):
    specs=json.loads((RUN/'manifest.json').read_text());protocol=require_protocol();checkpoints={}
    assert len(specs)==2800 and result['complete']
    for s in specs:
        p=JOBS/s['id'];assert json.loads((p/'spec.json').read_text())==s
        assert (p/'SUCCESS.json').exists() and (p/'FIT_COMPLETE.json').exists()
        m=json.loads((p/'metrics.json').read_text())
        assert not m['smoke_only'] and m['data_fingerprint']==FINGERPRINT
        assert m['n_val']==protocol['validation_rows'] and m['n_train']==31190
        assert m['validation_mask_sha256']==protocol['validation_mask_sha256'] and m['evaluation_year']==2023
        assert json.loads((p/'normalization.json').read_text())['fitted_split']=='train'
        for sel in m['selections'].values():assert sha(p/sel['checkpoint'])==sel['sha256']
        assert not (p/'test.json').exists()
        if s['configuration'] in ['W0_H32','TF_R3_RAW',result['selected']]:
            sel=m['selections']['mae']
            checkpoints[s['id']]=dict(checkpoint=sel['checkpoint'],sha256=sel['sha256'],spec_sha256=sha(p/'spec.json'),normalization_sha256=sha(p/'normalization.json'))
    assert len(list((RUN/'analysis/stations').glob('*.csv')))==56
    atomic_json(RUN/'SELECTION_FREEZE.json',dict(timestamp=now(),selected=result['selected'],references=['W0_H32','TF_R3_RAW'],checkpoints=checkpoints,protocol_sha256=sha(RUN/'PROTOCOL.json'),source_freeze_sha256=sha(RUN/'SOURCE_FREEZE.json'),manifest_sha256=sha(RUN/'manifest.json'),results_sha256=sha(RUN/'analysis/FACTORIAL_RESULTS.json'),data_fingerprint=FINGERPRINT,test_year_evaluated=False,future_test_interpretation='exploratory after prior2024 inspection; preserve original frozen main table'))
    atomic_json(RUN/'operations/COMPLETE_AUDIT.json',dict(timestamp=now(),passed=True,jobs=2800,farms=56,seeds=10,evaluation_year=2023,selected=result['selected'],test_evaluated=False))

def main():
    setup();lock=(RUN/'operations/finalizer.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    last=0.
    while True:
        try:
            state=collect(False)
            complete=state['completed']==state['planned'] and not state['failed']
            if time.monotonic()-last>300 or complete:
                result=report();last=time.monotonic()
            atomic_json(RUN/'operations/FOLLOWUP_STATUS.json',dict(timestamp=now(),pid=os.getpid(),state='RUNNING',**{k:state[k] for k in ['completed','planned','failed']}))
            if state['failed']:atomic_json(RUN/'operations/ATTENTION_REQUIRED.json',dict(timestamp=now(),failed_jobs=state['failed_jobs']))
            if complete:
                audit_complete(result)
                atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True))
                atomic_json(RUN/'operations/FOLLOWUP_STATUS.json',dict(timestamp=now(),state='COMPLETE',completed=2800,planned=2800,selected=result['selected']))
                return
            (RUN/'operations/FINALIZER_ERROR.json').unlink(missing_ok=True)
        except Exception:
            atomic_json(RUN/'operations/FINALIZER_ERROR.json',dict(timestamp=now(),traceback=traceback.format_exc()));traceback.print_exc()
        time.sleep(30)

if __name__=='__main__':main()
