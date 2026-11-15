"""One-time, identity-checked restart of only this run's unfinished CPU ARX."""
from pathlib import Path
import os,sys,json,time,subprocess,shlex,socket,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1];PROJECT=RUN.parents[1]
sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now
REMOTE_PROJECT='/home/liying/wind_physics_project/project'
REMOTE_RUN=REMOTE_PROJECT+'/runs/'+RUN.name
HOSTS=['v1006001','v1006002']
STAMP='20261027_QR_RECOVERY'


def execute(host,code):
    if host=='local':
        args=[sys.executable,'-c',code]
    else:
        args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(code)]
    r=subprocess.run(args,capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(host+' '+r.stdout[-2000:]+r.stderr[-2000:])
    return json.loads(r.stdout)


def main():
    assert json.loads((HERE/'FULL_VALIDATION.json').read_text())['passed']
    assert not (HERE/'RESTART_COMPLETE.json').exists(), 'Already restarted; inspect live state instead'
    manifest=json.loads((RUN/'manifest.json').read_text())
    targets=[s['id'] for s in manifest if s['task']=='arx' and not (JOBS/s['id']/'SUCCESS.json').exists()]
    before=json.loads((RUN/'STATUS.json').read_text())
    assert before['training_completed']==before['planned_training']
    assert before['reference_diagnostics_completed']==before['planned_reference_diagnostics']
    atomic_json(HERE/'RESTART_START.json',dict(timestamp=now(),targets=targets,original_status=before,
                authorized='User requested diagnosis and restart of unfinished ARX; retain scientific objective and original attempts'))
    # Only exact script arguments from this run, owned by this UID, are eligible.
    stops={}
    for host in ['local',*HOSTS]:
        root=str(RUN) if host=='local' else REMOTE_RUN
        allowed_names=['cpu_scheduler_v2.py','cpu_scheduler_qr_v3.py','cpu_remote_controller.py']
        if host=='local':allowed_names.append('remote_controller.py')
        code=f'''from pathlib import Path
import os,json,signal,time
root=Path({root!r});allowed={{str(root/'operations'/n) for n in {allowed_names!r}}}
targets=set({targets!r});items=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=p.joinpath('cmdline').read_bytes().decode().split('\\0')
  if not allowed.intersection(argv):continue
  worker='--job-id' in argv
  jid=argv[argv.index('--job-id')+1] if worker else None
  if worker and jid not in targets:continue
  fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
  items.append(dict(pid=int(p.name),start=int(fields[19]),job_id=jid,worker=worker,argv=argv))
 except (OSError,ValueError,IndexError):continue
audit=root/'operations/arx_recovery_20261027/STOP_IDENTITIES.json';audit.write_text(json.dumps(items,indent=2)+'\\n')
for item in sorted(items,key=lambda x:x['worker']):
 try:
  fields=Path('/proc/'+str(item['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
  assert int(fields[19])==item['start']
  os.kill(item['pid'],signal.SIGTERM)
 except ProcessLookupError:pass
print(json.dumps(items))'''
        stops[host]=execute(host,code)
    # Reuse the failed-site pilot's checkpoint rather than restarting its solved alphas.
    pilot=HERE/'validation/site09';heartbeat=pilot/'QR_HEARTBEAT.json'
    if heartbeat.exists() and not (pilot/'SUCCESS.json').exists():
        h=json.loads(heartbeat.read_text());pid=h['pid'];proc=Path(f'/proc/{pid}')
        if proc.exists():
            cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert proc.stat().st_uid==os.getuid() and 'recover_arx.py --site 9 --output' in cmd and 'validation/site09' in cmd
            import signal
            os.kill(pid,signal.SIGTERM)
    # No mutation of job artifacts until the old controllers have received exits.
    deadline=time.monotonic()+60
    while True:
        alive={}
        for host,items in stops.items():
            alive[host]=execute(host,f'''from pathlib import Path
import json
live=[]
for item in {items!r}:
 try:
  f=Path('/proc/'+str(item['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
  if int(f[19])==item['start'] and f[0] not in ['Z','X']:live.append(item['pid'])
 except OSError:pass
print(json.dumps(live))''')
        if not any(alive.values()):break
        if time.monotonic()>deadline:raise RuntimeError('Controllers still draining; no artifacts changed: '+str(alive))
        time.sleep(2)
    targets=[j for j in targets if not (JOBS/j/'SUCCESS.json').exists()]
    atomic_json(HERE/'STOP_AUDIT.json',dict(timestamp=now(),stopped=stops,restart_targets=targets,all_old_cpu_processes_exited=True))
    for host in ['local',*HOSTS]:
        root=str(RUN) if host=='local' else REMOTE_RUN
        result=execute(host,f'''from pathlib import Path
import json,os,time
r=Path({root!r});stamp={STAMP!r};targets={targets!r};archived=[]
for jid in targets:
 p=r/'jobs'/jid;p.mkdir(parents=True,exist_ok=True)
 assert not (p/'SUCCESS.json').exists(),jid
 hold=p/'CPU_RETRY_HOLD.json';hold.write_text(json.dumps(dict(reason='verified QR recovery staging')))
 archive=p/'attempts'/stamp;archive.mkdir(parents=True,exist_ok=True)
 for f in list(p.iterdir()):
  if f.name in ['attempts','worker.lock','CPU_RETRY_HOLD.json']:continue
  assert not (archive/f.name).exists(),str(f)
  f.rename(archive/f.name)
 archived.append(jid)
archive=r/'operations/arx_recovery_20261027/old_operational_state';archive.mkdir(exist_ok=True)
for name in ['CPU_GLOBAL_COMPLETE.json','GLOBAL_COMPLETE.json','CPU_ADOPT.json','CPU_LEASED_JOBS.json']:
 p=r/'operations'/name
 if p.exists():p.rename(archive/name)
for name in ['cpu_scheduler_v2.json','cpu_remote_v1006001.json','cpu_remote_v1006002.json']:
 p=r/'schedulers'/name
 if p.exists():
  d=json.loads(p.read_text());(archive/name).write_text(p.read_text());d.update(enabled=False,active=[],replaced_by='cpu_scheduler_qr_v3.py');p.write_text(json.dumps(d))
print(json.dumps(dict(archived=archived)))''')
        print(host,'archived',len(result['archived']),flush=True)
    # Promote only the same-spec pilot, preserving its last committed checkpoint.
    if 'ARX_s09' in targets and pilot.exists():
        out=JOBS/'ARX_s09'
        import shutil
        for f in pilot.iterdir():
            if f.is_file():shutil.move(str(f),out/f.name)
        atomic_json(pilot/'PROMOTED.json',dict(timestamp=now(),destination=str(out),same_scientific_spec=True))
        # Keep this one local: its checkpoint is already here.
    for host in ['local',*HOSTS]:
        root=str(RUN) if host=='local' else REMOTE_RUN
        execute(host,f'''from pathlib import Path
import json
r=Path({root!r})
for jid in {targets!r}:(r/'jobs'/jid/'CPU_RETRY_HOLD.json').unlink(missing_ok=True)
print(json.dumps(dict(ready=True)))''')
    launches={}
    # Start remote schedulers, then coordinator, then local scheduler. Resource guards remain unchanged.
    for host in HOSTS:
        workers=56 if host=='v1006001' else 20
        reserve=6 if host=='v1006001' else 4
        launches[host]=execute(host,f'''from pathlib import Path
import subprocess,os,json
r=Path({REMOTE_RUN!r});log=(r/'operations/arx_recovery_20261027/scheduler.log').open('a')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
p=subprocess.Popen([{(REMOTE_PROJECT+'/.runtime/chronos/bin/python3.11')!r},'-s',str(r/'operations/cpu_scheduler_qr_v3.py'),'--max-workers',{str(workers)!r},'--reference-slots',{str(reserve)!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,max_workers={workers})))''')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (HERE/'controller.log').open('a') as log:
        controller=subprocess.Popen([sys.executable,'-s',str(RUN/'operations/cpu_remote_controller.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (HERE/'scheduler.log').open('a') as log:
        scheduler=subprocess.Popen([sys.executable,'-s',str(RUN/'operations/cpu_scheduler_qr_v3.py'),'--max-workers','32','--reference-slots','4'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launches['local']=dict(controller_pid=controller.pid,scheduler_pid=scheduler.pid,max_workers=32)
    atomic_json(HERE/'RESTART_COMPLETE.json',dict(timestamp=now(),targets=targets,launches=launches,
        status='launched; verify real worker heartbeats and solver progress',original_successes_preserved=True,
        objective_unchanged=True,tolerance=1e-4,alpha_grid=[.003,.001,.0003],l1_ratio=.8,
        original_failed_attempts_archived=STAMP,open_2024=False))
    print(json.dumps(launches),flush=True)


if __name__=='__main__':main()
