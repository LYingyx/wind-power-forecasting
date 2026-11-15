"""Durable D -> eight additional seeds dependency, with an audited manifest at activation."""
from pathlib import Path
import sys,json,time,fcntl,subprocess,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from extend_seeds import select_arms,build_manifest
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name

def materialize():
    audit=SCREEN_RUN/'COMPLETION_AUDIT.json';recommendation=SCREEN_RUN/'analysis/D_SELECTION_RECOMMENDATION.json'
    assert json.loads(audit.read_text())['passed']
    rec=json.loads(recommendation.read_text());arms=select_arms(rec)
    if (RUN/'ACTIVATION.json').exists():
        a=json.loads((RUN/'ACTIVATION.json').read_text())
        assert a['source_recommendation_sha256']==sha(recommendation)
        assert a['manifest_sha256']==sha(RUN/'manifest.json')
        return a
    allowed={s['id']:s for s in json.loads((RUN/'candidate_manifest.json').read_text())}
    specs=build_manifest(arms);assert all(allowed[s['id']]==s for s in specs)
    reuse=[]
    for spec in json.loads((SCREEN_RUN/'manifest.json').read_text()):
        if (spec['configuration'],spec['variant']) not in arms:continue
        path=SCREEN_RUN/'jobs'/spec['id'];assert (path/'SUCCESS.json').exists()
        reuse.append(dict(spec=spec,path=str(path),metrics_sha256=sha(path/'metrics.json'),
                          diagnostics_sha256=sha(path/'diagnostics.json')))
    assert len(reuse)==len(arms)*56*2
    atomic_json(RUN/'manifest.json',specs);atomic_json(RUN/'REUSED_SOURCES.json',reuse)
    record=dict(timestamp=now(),arms=arms,new_seed_fits=len(specs),reused_results=len(reuse),
        manifest_sha256=sha(RUN/'manifest.json'),reused_sources_sha256=sha(RUN/'REUSED_SOURCES.json'),
        source_recommendation_sha256=sha(recommendation),source_audit_sha256=sha(audit),open_2024=False)
    atomic_json(RUN/'ACTIVATION.json',record)
    return record

def main():
    lock=(RUN/'operations/selection_watcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            for p,h in json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files'].items():
                # Large initializers are checked at deployment and activation, not every heartbeat.
                if '/reference_B0_' not in p:assert sha(p)==h,p
            for p,h in json.loads((RUN/'operations/OPERATIONAL_OVERLAY_V2.json').read_text())['files'].items():assert sha(p)==h,p
            status=json.loads((SCREEN_RUN/'STATUS.json').read_text())
            audit=SCREEN_RUN/'COMPLETION_AUDIT.json'
            if audit.exists() and json.loads(audit.read_text()).get('passed'):
                record=materialize()
                for host in ['v1006001','v1006002']:
                    subprocess.run(['rsync','-a',*[str(RUN/n) for n in ['manifest.json','ACTIVATION.json','REUSED_SOURCES.json']],
                                    host+':'+REMOTE+'/'],check=True,timeout=90)
                subprocess.run([sys.executable,'-s',str(RUN/'operations/launch_extension_v2.py')],check=True)
                atomic_json(RUN/'PENDING_STATUS.json',dict(record,state='activated',activated_at=now()))
                return
            atomic_json(RUN/'PENDING_STATUS.json',dict(timestamp=now(),state='waiting_for_complete_D_audit',
                pid=__import__('os').getpid(),parent_completed=status['completed'],parent_planned=status['planned'],
                parent_failed=status['failed'],min_new_fits=1792,max_new_fits=5376,open_2024=False))
        except Exception:
            atomic_json(RUN/'PENDING_STATUS.json',dict(timestamp=now(),state='retryable_dependency_error',
                error=traceback.format_exc(),open_2024=False))
        time.sleep(30)

if __name__=='__main__':main()
