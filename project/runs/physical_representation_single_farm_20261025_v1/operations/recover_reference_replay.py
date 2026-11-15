"""Replay one CPU roundoff failure on CUDA, keeping the original evidence."""
from pathlib import Path
import os,sys,json,fcntl
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now
from scheduled_worker import run

if __name__=='__main__':
    jid='A4_input_TF_DENS_MAE_s29_r42_d15';out=JOBS/jid
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evidence=json.loads((RUN/'operations/REFERENCE_REPLAY_INVESTIGATION.json').read_text())
    assert abs(evidence['results']['cuda:0']['rmse']['difference'])<2e-6
    old_failure=json.loads((out/'FAILURE.json').read_text())
    old_runtime=json.loads((out/'runtime.json').read_text())
    rc=run(jid)
    assert rc==0 and (out/'SUCCESS.json').exists()
    atomic_json(RUN/'operations/REFERENCE_REPLAY_RECOVERY.json',dict(timestamp=now(),job=jid,reason='CPU/CUDA FP32 arithmetic difference; original 2e-6 tolerance retained',original_failure=old_failure,original_runtime=old_runtime,investigation=evidence,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),retrained=False,passed=True))
    (out/'FAILURE.json').unlink()
    print('Recovered reference replay without retraining or tolerance changes',flush=True)
