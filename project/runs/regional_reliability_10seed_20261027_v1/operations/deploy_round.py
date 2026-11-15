"""Deploy only this round's small code/derived assets; reuse existing data."""
from pathlib import Path
import subprocess,json,shlex,sys
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now
REMOTE_PROJECT='/home/liying/wind_physics_project/project';REMOTE_RUN=REMOTE_PROJECT+'/runs/'+RUN.name

def ssh(host,script):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)],check=True,text=True,capture_output=True,timeout=90).stdout

def deploy(host):
    print(ssh(host,f'''from pathlib import Path
import json
r=Path({REMOTE_RUN!r});r.mkdir(exist_ok=True)
for name in ['code','assets','operations','analysis','schedulers','jobs']:(r/name).mkdir(exist_ok=True)
if (r/'SOURCE_FREEZE.json').exists():
 import hashlib
 assert hashlib.sha256((r/'SOURCE_FREEZE.json').read_bytes()).hexdigest()=={__import__('hashlib').sha256((RUN/'SOURCE_FREEZE.json').read_bytes()).hexdigest()!r},'Different freeze already deployed'
(r/'operations/REMOTE_DEPLOYMENT.json').write_text(json.dumps(dict(source_project={str(RUN.parents[1])!r},source_storage={'/shared_data_ssd3/liying/'+RUN.name!r})))
print('remote directories ready')'''),flush=True)
    for folder in ['code','assets']:
        subprocess.run(['rsync','-a','--exclude=__pycache__',str(RUN/folder)+'/',host+':'+REMOTE_RUN+'/'+folder+'/'],check=True)
    paths=list((RUN/'operations').glob('*.py'))+[RUN/'operations/HOST_POLICY.json',RUN/'operations/CHECKS.json',RUN/'operations/REPORTING_CHECKS.json',RUN/'operations/compatibility_fixture.pt',RUN/'operations/README_CN.md']
    subprocess.run(['rsync','-a',*[str(p) for p in paths],host+':'+REMOTE_RUN+'/operations/'],check=True)
    root_files=['manifest.json','PROTOCOL.json','SOURCE_FREEZE.json','README_CN.md','SELECTION_RULES.json','candidate_extension_manifest.json','REUSED_SCREEN.json','ACTIVATION.json']
    subprocess.run(['rsync','-a',*[str(RUN/p) for p in root_files if (RUN/p).exists()],host+':'+REMOTE_RUN+'/'],check=True)
    subprocess.run(['rsync','-a',str(RUN/'analysis/train_only_bins.json'),host+':'+REMOTE_RUN+'/analysis/'],check=True)
    print(ssh(host,f'''from pathlib import Path
import sys,json,hashlib
r=Path({REMOTE_RUN!r});sys.path.insert(0,str(r/'operations'));from relocate import activate,source_path
activate(r)
from settings import setup,sha
setup()
freeze=json.loads((r/'SOURCE_FREEZE.json').read_text())
for path,h in freeze['files'].items():assert sha(source_path(r,path))==h,path
print(json.dumps(dict(host={host!r},verified_files=len(freeze['files']),data_reused=True)))'''),flush=True)
if __name__=='__main__':
    for host in ['v1006001','v1006002']:deploy(host)
    atomic_json(RUN/'operations/DEPLOYED.json',dict(timestamp=now(),hosts=['v1006001','v1006002'],verified=True))
