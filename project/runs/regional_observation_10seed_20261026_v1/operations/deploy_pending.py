"""Deploy a dormant queue and the exact new-seed initializers; no workers are launched."""
from pathlib import Path
import sys,subprocess,json,shlex,concurrent.futures
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE_PROJECT='/home/liying/wind_physics_project/project'
REMOTE_RUN=REMOTE_PROJECT+'/runs/'+RUN.name

def ssh(host,script):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)],check=True,capture_output=True,text=True,timeout=90).stdout

def deploy(host):
    ssh(host,f'''from pathlib import Path
import json
r=Path({REMOTE_RUN!r});r.mkdir(exist_ok=True)
for name in ['code','assets','operations','analysis','schedulers','jobs']:(r/name).mkdir(exist_ok=True)
(r/'operations/REMOTE_DEPLOYMENT.json').write_text(json.dumps(dict(source_project={str(PROJECT)!r},source_storage={str(STORAGE)!r})))
print('ready')''')
    for name in ['code','assets','operations']:
        subprocess.run(['rsync','-a','--bwlimit=60000','--exclude=__pycache__','--exclude=*.log','--exclude=*.lock',
                        str(RUN/name)+'/',host+':'+REMOTE_RUN+'/'+name+'/'],check=True)
    subprocess.run(['rsync','-a',*[str(RUN/n) for n in ['manifest.json','candidate_manifest.json','PROTOCOL.json','SOURCE_FREEZE.json','INITIALIZER_FREEZE.json','README_CN.md']],
                    host+':'+REMOTE_RUN+'/'],check=True)
    subprocess.run(['rsync','-a',str(RUN/'analysis/train_only_bins.json'),host+':'+REMOTE_RUN+'/analysis/'],check=True)
    initializers=json.loads((RUN/'INITIALIZER_FREEZE.json').read_text())['files']
    listing=RUN/'operations'/('initializer_paths_'+host+'.txt')
    listing.write_text(''.join(str(Path(p).relative_to(PROJECT))+'\n' for p in initializers))
    subprocess.run(['rsync','-a','--bwlimit=60000','--files-from='+str(listing),str(PROJECT)+'/',host+':'+REMOTE_PROJECT+'/'],check=True)
    result=ssh(host,f'''from pathlib import Path
import sys,json
r=Path({REMOTE_RUN!r});sys.path.insert(0,str(r/'operations'));from relocate import activate,source_path
activate(r)
from settings import sha,setup
setup();freeze=json.loads((r/'SOURCE_FREEZE.json').read_text())
for p,h in freeze['files'].items():assert sha(source_path(r,p))==h,p
print(json.dumps(dict(verified_files=len(freeze['files']),new_seed_initializers=448,pending_only=True)))''')
    return dict(host=host,**json.loads(result))

def main():
    with concurrent.futures.ThreadPoolExecutor(2) as pool:results=list(pool.map(deploy,['v1006001','v1006002']))
    atomic_json(RUN/'operations/DEPLOYED.json',dict(timestamp=now(),hosts=results,verified=True,pending_only=True))
    print(json.dumps(results))

if __name__=='__main__':main()
