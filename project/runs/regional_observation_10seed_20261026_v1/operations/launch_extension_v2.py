"""Idempotent launch of the authorized three-host extension after activation."""
from pathlib import Path
import os,sys,subprocess,json,shlex,fcntl
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now

def main():
    lock=(RUN/'operations/activation_launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (RUN/'ACTIVATION.json').exists() and (RUN/'operations/DEPLOYED.json').exists()
    remote='/home/liying/wind_physics_project/project/runs/'+RUN.name
    python='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python3.11'
    records={}
    for host in ['v1006001','v1006002']:
        script=f'''from pathlib import Path
import subprocess,os,json,fcntl
r=Path({remote!r});lock=(r/'operations/extension_launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
marker=r/'operations/EXTENSION_LAUNCH.json';old=json.loads(marker.read_text()) if marker.exists() else {{}}
pid=old.get('pid',0);proc=Path('/proc')/str(pid)
alive=proc.exists() and proc.stat().st_uid==os.getuid() and str(r).encode() in (proc/'cmdline').read_bytes()
if not alive:
 env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (r/'operations/bootstrap_efficient_v2.log').open('a') as f:
  p=subprocess.Popen([{python!r},'-s',str(r/'operations/bootstrap_efficient_v2.py'),'--warmup-seconds','60'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 old=dict(pid=p.pid,host={host!r});marker.write_text(json.dumps(old))
print(json.dumps(old))'''
        r=subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
        records[host]=json.loads(r.stdout)
    atomic_json(RUN/'operations/LAUNCH_REMOTES.json',records)
    localfile=RUN/'operations/LAUNCH_LOCAL.json';local=json.loads(localfile.read_text()) if localfile.exists() else {}
    env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for name,args in [('remote_controller',[]),('bootstrap_efficient_v2',['--warmup-seconds','60']),('finalize_extension_v2',[])]:
        pid=local.get(name,{}).get('pid',0);proc=Path('/proc')/str(pid)
        alive=proc.exists() and proc.stat().st_uid==os.getuid() and str(RUN).encode() in (proc/'cmdline').read_bytes()
        if alive:continue
        cmd=[sys.executable,'-s',str(RUN/'operations'/(name+'.py')),*args]
        with (RUN/'operations'/(name+'.log')).open('a') as f:
            p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        local[name]=dict(pid=p.pid,args=cmd,timestamp=now());atomic_json(localfile,local)
    print(json.dumps(dict(remote=records,local=local)))

if __name__=='__main__':main()
