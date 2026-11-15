"""Operational wrapper: original training math unchanged, cooperative release at checkpoints."""
from pathlib import Path
import sys,os,argparse,json,socket,time,traceback
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
from relocate import activate,source_path
DEPLOYMENT=activate(RUN)
import train_physics as training
from settings import JOBS,atomic_json,now

class ResourceYield(Exception):pass

def install_yield_hook(yield_path):
    def evaluation_check():
        if yield_path.exists():raise ResourceYield(str(yield_path))
    training.evaluation_yield_check=evaluation_check
    original=training.fit.save_torch
    def checkpoint(path,obj):
        original(path,obj)
        if path.name=='latest.pt' and yield_path.exists():
            # All optimizer/RNG/selected-weight references are durable at this boundary.
            keep={x['checkpoint'] for x in obj['selections'].values()}
            for extra in path.parent.glob('epoch_*.pt'):
                if extra.name not in keep:extra.unlink()
            raise ResourceYield(str(yield_path))
    training.fit.save_torch=checkpoint
    return original


def run(jid):
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==jid)
    out=JOBS/jid;out.mkdir(parents=True,exist_ok=True)
    control=out/'YIELD_REQUEST.json'
    atomic_json(out/'runtime.json',dict(host=socket.gethostname(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=now(),wrapper='scheduled_worker_v2',torch_version=training.torch.__version__,cuda_version=training.torch.version.cuda,gpu_name=training.torch.cuda.get_device_name(0),numpy_version=training.np.__version__))
    try:os.nice(10)
    except OSError:pass
    install_yield_hook(control)
    try:
        if control.exists():raise ResourceYield('Yield requested before data loading')
        training.run(spec,'cuda:0')
        (out/'YIELDED.json').unlink(missing_ok=True)
    except ResourceYield:
        atomic_json(out/'YIELDED.json',dict(timestamp=now(),host=socket.gethostname(),pid=os.getpid(),checkpoint_exists=(out/'latest.pt').exists(),reason='resource_headroom_reclaimed',exit_code=75))
        return 75
    except BaseException:
        atomic_json(out/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc(),host=socket.gethostname()))
        return 1
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);a=p.parse_args();sys.exit(run(a.job_id))
