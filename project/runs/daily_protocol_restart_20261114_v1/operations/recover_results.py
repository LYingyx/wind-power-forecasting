"""Recover completed remote jobs without ever counting partial checkpoints."""
from pathlib import Path
import sys,json,subprocess,shlex,collections
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np

REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
def recover():
    manifest=json.loads((RUN/'manifest.json').read_text());specs={s['id']:s for s in manifest};inventories={};returned=[]
    for host in ['v1006001','v1006002']:
        script=f"from pathlib import Path;import json;r=Path({REMOTE!r});print(json.dumps([p.parent.name for p in (r/'jobs').glob('*/SUCCESS.json')]))"
        inventories[host]=json.loads(subprocess.check_output(['ssh',host,'python3 -c '+shlex.quote(script)],text=True,timeout=30))
        for jid in inventories[host]:
            assert jid in specs;out=JOBS/jid
            if (out/'SUCCESS.json').exists():continue
            out.mkdir(exist_ok=True)
            subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=YIELD_REQUEST.json',host+':'+REMOTE+'/jobs/'+jid+'/',str(out)+'/'],check=True,timeout=120)
            d=json.loads((out/'metrics.json').read_text());assert d['requested_spec']==specs[jid] and not d['smoke_only'];assert sha(out/'best.pt')==d['checkpoint_sha256']
            if 'reference_sha256' in d:assert sha(out/'reference_prediction.npy')==d['reference_sha256']
            atomic_json(out/'SUCCESS.json',dict(timestamp=now(),source_host=host,verified_return=True,passed=True,recovered=True));(out/'REMOTE.json').unlink(missing_ok=True);returned.append(jid)
    verified=[];groups=collections.defaultdict(list);validation_counts=set()
    for spec in manifest:
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        d=json.loads((out/'metrics.json').read_text());assert d['requested_spec']==spec and not d['smoke_only'];assert sha(out/'best.pt')==d['checkpoint_sha256'],spec['id']
        m=d['metrics'];ma=np.asarray(m['per_farm_mae']);rm=np.asarray(m['per_farm_rmse']);lm=np.asarray(m['lead_mae']);lr=np.asarray(m['lead_rmse'])
        assert ma.shape==rm.shape==(55,) and lm.shape==lr.shape==(55,24)
        assert np.isfinite(ma).all() and np.isfinite(rm).all() and (ma>=0).all() and (rm>=0).all()
        assert np.allclose(lm.mean(1),ma,rtol=1e-10,atol=1e-12)
        assert np.allclose(np.sqrt((lr**2).mean(1)),rm,rtol=1e-10,atol=1e-12)
        assert abs(ma.mean()-m['mae'])<1e-12 and abs(rm.mean()-m['rmse'])<1e-12
        if d['evaluation_year']==2023:
            assert set(m['counts'])=={29647};validation_counts.update(m['counts'])
        if 'reference_sha256' in d:assert sha(out/'reference_prediction.npy')==d['reference_sha256']
        verified.append(spec['id']);groups[spec['stage']].append(spec['id']);(out/'REMOTE.json').unlink(missing_ok=True)
    report=dict(timestamp=now(),passed=True,planned=len(manifest),verified_complete=len(verified),newly_recovered=returned,by_stage={k:len(v) for k,v in groups.items()},missing=[s['id'] for s in manifest if s['id'] not in verified],remote_successes=inventories,shared_validation_samples_per_farm=list(validation_counts),checksum_and_metric_consistency=True)
    atomic_json(RUN/'analysis/RECOVERY_AUDIT.json',report)
    sys.path.insert(0,str(RUN/'operations'));from collect import collect
    collect();print(json.dumps({k:v for k,v in report.items() if k not in ['remote_successes','missing']},ensure_ascii=False),flush=True)
if __name__=='__main__':recover()
