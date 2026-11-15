from pathlib import Path
import sys,subprocess,json,shlex,os,hashlib
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE_PROJECT='/home/liying/wind_physics_project/project';REMOTE_RUN=REMOTE_PROJECT+'/runs/'+RUN.name
HOSTS=['v1006001','v1006002']
def ssh(host,script):
 return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=180).stdout

def freeze():
 assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
 paths=list((RUN/'code').rglob('*.py'))+list((RUN/'operations').glob('*.py'))+list(ASSETS.glob('*'))
 paths += [RUN/n for n in ['candidate_manifest.json','screen_manifest.json','PROTOCOL.json']]
 paths += [RUN/'operations'/n for n in ['CHECKS.json','HOST_POLICY.json','compatibility_fixture.pt']]
 files={str(p):sha(p) for p in paths if p.is_file()}
 # Hash only2022/2023 inputs; test data are outside this selection run.
 for folder in [CACHE/'static',CACHE/'train',CACHE/'val',BUNDLE/'local_weather',BUNDLE/'physical']:
  for p in folder.rglob('*'):
   if not p.is_file() or p.name in ['weather.npy','test.npy'] or 'test' in p.relative_to(BUNDLE).parts:continue
   # canonical bundle paths are represented beneath project for portable verification.
   dest=PROJECT/'data/operational_bundle_v1'/p.relative_to(BUNDLE)
   files[str(dest)]=sha(p)
 files[str(PROJECT/'data/operational_bundle_v1/cache/manifest.json')]=sha(CACHE/'manifest.json')
 atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files=files,data_fingerprint=FINGERPRINT,live_manifest='subset of frozen candidate_manifest, append-only activation with recorded reason'))
 print('Frozen',len(files),'files',flush=True)

def deploy(host):
 ssh(host,f"from pathlib import Path; r=Path({REMOTE_RUN!r}); [ (r/d).mkdir(parents=True,exist_ok=True) for d in ['code','assets','operations','analysis','schedulers','jobs','sources'] ]; (r/'operations/REMOTE_DEPLOYMENT.json').write_text({json.dumps(dict(source_project=str(PROJECT)))!r})")
 for folder in ['code','assets']:
  subprocess.run(['rsync','-a','--exclude=__pycache__',str(RUN/folder)+'/',host+':'+REMOTE_RUN+'/'+folder+'/'],check=True)
 files=list((RUN/'operations').glob('*.py'))+[RUN/'operations'/n for n in ['CHECKS.json','HOST_POLICY.json','compatibility_fixture.pt']]
 subprocess.run(['rsync','-a',*[str(p) for p in files],host+':'+REMOTE_RUN+'/operations/'],check=True)
 subprocess.run(['rsync','-a',*[str(RUN/n) for n in ['manifest.json','screen_manifest.json','candidate_manifest.json','PROTOCOL.json','SOURCE_FREEZE.json']],host+':'+REMOTE_RUN+'/'],check=True)
 print(ssh(host,f"from pathlib import Path; import sys,json; r=Path({REMOTE_RUN!r});sys.path.insert(0,str(r/'operations'));from relocate import source_path,activate;activate(r);from settings import sha,setup;setup(); f=json.loads((r/'SOURCE_FREEZE.json').read_text());bad=[p for p,h in f['files'].items() if sha(source_path(r,p))!=h];assert not bad,bad;print('Verified',len(f['files']),'files')"),flush=True)

def start(include_finalizer=True):
 require_protocol();setup();env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');remote={}
 for host in HOSTS:
  script=f'''import subprocess,os,json
from pathlib import Path
r=Path({REMOTE_RUN!r});(r/'operations/GLOBAL_COMPLETE.json').unlink(missing_ok=True)
env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'operations/bootstrap.log').open('a') as f:
 p=subprocess.Popen([{REMOTE_PROJECT+'/.runtime/chronos/bin/python3.11'!r},'-s',str(r/'operations/bootstrap_efficient_v3.py'),'--warmup-seconds','20','--allow-shared-compute'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
print(json.dumps(dict(pid=p.pid)))'''
  remote[host]=json.loads(ssh(host,script))
 (RUN/'operations/GLOBAL_COMPLETE.json').unlink(missing_ok=True)
 local={}
 names=[('remote_controller',[]),('bootstrap_efficient_v3',['--warmup-seconds','20'])]
 if include_finalizer:names.append(('finalize',[]))
 for name,args in names:
  with (RUN/'operations'/f'{name}.log').open('a') as f:
   p=subprocess.Popen([sys.executable,'-s',str(RUN/'operations'/f'{name}.py'),*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  local[name]=dict(pid=p.pid)
 record=dict(timestamp=now(),remote=remote,local=local);atomic_json(RUN/'operations'/f'LAUNCH_{len(list((RUN/"operations").glob("LAUNCH_*.json")))}.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 elif sys.argv[1]=='deploy':deploy(sys.argv[2])
 elif sys.argv[1]=='start':start()
