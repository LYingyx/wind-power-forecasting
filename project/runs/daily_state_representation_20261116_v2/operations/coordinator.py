"""Collect remote evidence and activate only predeclared, gate-qualified stages."""
from pathlib import Path
import sys,os,json,time,subprocess,shlex,fcntl,datetime
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,ASSETS,atomic_json,now,sha
from collect import collect
from report_diagnostics import report as report_diagnostics
HOST='v1006001';REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name


def remote(code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,'python3 -c '+shlex.quote(code)],
                     check=True,capture_output=True,text=True,timeout=40)
    return json.loads(p.stdout)


def main():
    lock=(RUN/'operations/coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=json.loads((RUN/'manifest.json').read_text());diagnostic=json.loads((RUN/'diagnostic_manifest.json').read_text())
    by_id={s['id']:s for s in specs+diagnostic};assignments=json.loads((RUN/'operations/ASSIGNMENTS.json').read_text())
    remote_ids=[j for j,h in assignments.items() if h==HOST]
    while True:
        errors=[];state={}
        try:
            state=remote(f'''from pathlib import Path
import json,time
r=Path({REMOTE!r}); ids={remote_ids!r}
p=r/'schedulers/status.json'
print(json.dumps(dict(done=[j for j in ids if (r/'jobs'/j/'SUCCESS.json').exists()],
 failed=[j for j in ids if (r/'jobs'/j/'FAILURE.json').exists()],
 status=json.loads(p.read_text()) if p.exists() else {{}},fresh=p.exists() and time.time()-p.stat().st_mtime<120)))''')
            subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--include=*/','--include=*.json','--exclude=*',
                            HOST+':'+REMOTE+'/jobs/',str(JOBS)+'/'],check=True,timeout=45,stdout=subprocess.DEVNULL)
            for jid in state['done']:
                out=JOBS/jid
                if (out/'SUCCESS.json').exists():continue
                subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=YIELD_REQUEST.json',
                                HOST+':'+REMOTE+'/jobs/'+jid+'/',str(out)+'/'],check=True,timeout=90,stdout=subprocess.DEVNULL)
                value=json.loads((out/'metrics.json').read_text())
                assert value['requested_spec']==by_id[jid] and not value['smoke_only'] and value['evaluation_year']==2023
                if value.get('no_training'):assert sha(RUN/by_id[jid]['checkpoint'])==value['checkpoint_sha256']
                else:assert sha(out/'best.pt')==value['checkpoint_sha256']
                atomic_json(out/'SUCCESS.json',dict(timestamp=now(),passed=True,source_host=HOST,verified_return=True))
            gates=collect()
            for p in [ASSETS/'selected_candidate.json',*sorted((RUN/'operations').glob('ENABLE_*.json'))]:
                if p.exists():
                    rel=str(p.parent.relative_to(RUN))
                    subprocess.run(['rsync','-a',str(p),HOST+':'+REMOTE+'/'+rel+'/'],check=True,timeout=30,stdout=subprocess.DEVNULL)
        except Exception as exc:
            errors.append(repr(exc));gates=collect()
        localfile=RUN/'schedulers/status.json'
        local=json.loads(localfile.read_text()) if localfile.exists() else {}
        local_fresh=localfile.exists() and time.time()-localfile.stat().st_mtime<120
        active=(local.get('active',[]) if local_fresh else [])+(state.get('status',{}).get('active',[]) if state.get('fresh') else [])
        active=[x for x in active if not (JOBS/x['id']/'SUCCESS.json').exists()]
        progress={}
        for item in active:
            path=JOBS/item['id']/'progress.json'
            if path.exists():progress[item['id']]=json.loads(path.read_text())
        required=[s for s in specs if all((RUN/d).exists() for d in s['dependencies'])]
        finished=[s['id'] for s in required if (JOBS/s['id']/'SUCCESS.json').exists()]
        diagnostic_done=sum((JOBS/s['id']/'SUCCESS.json').exists() for s in diagnostic)
        failures=[s['id'] for s in specs+diagnostic if (JOBS/s['id']/'FAILURE.json').exists()]
        report=dict(timestamp=now(),run=RUN.name,stage1_planned=25,maximum_training_jobs=55,
                    currently_enabled_training_jobs=len(required),completed_training=len(finished),
                    diagnostic_evaluations_completed=diagnostic_done,active=active,progress=progress,
                    failures=failures,errors=errors,gates=gates,hosts=dict(local4090=local,v1006001=state))
        atomic_json(RUN/'STATUS.json',report)
        report_diagnostics()
        atomic_json(RUN.parents[1]/'ACTIVE_EXPERIMENT.json',dict(timestamp=now(),run=str(RUN),status_path=str(RUN/'STATUS.json'),
                    scope='v2 state representation; conditional stages; local4090 and v1006001'))
        if gates['finished'] and diagnostic_done==len(diagnostic):
            atomic_json(RUN/'COMPLETE.json',dict(timestamp=now(),training_completed=len(finished),diagnostic_completed=diagnostic_done,gates=gates))
            atomic_json(RUN/'operations/STOP.json',{})
            remote(f"from pathlib import Path;Path({(REMOTE+'/operations/STOP.json')!r}).write_text('{{}}');print('true')")
            break
        time.sleep(30)

if __name__=='__main__':main()
