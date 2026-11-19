"""Re-evaluate frozen R0/TF weights for the newly required diagnostics; no training."""
from settings import *
from train import resolve,evaluate
from data import SharedData
from model import SharedModel,configure
import argparse,torch,traceback

def main(jid):
    configure()
    requested={s['id']:s for s in json.loads((RUN/'diagnostic_manifest.json').read_text())}[jid]
    spec=resolve(requested);out=JOBS/jid;out.mkdir(parents=True,exist_ok=True)
    path=RUN/spec['checkpoint'];assert sha(path)==spec['checkpoint_sha256']
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=SharedModel(spec).cuda();model.load_state_dict(ck['model'])
    data=SharedData('val',spec);metrics=evaluate(model,data,1024,'cuda:0',diagnostics=True)
    group=spec['representation'];reference=json.loads((RUN/'reference_metrics'/group/str(spec['seed'])/'metrics.json').read_text())
    for key in ['mae','rmse','high_power_rmse']:assert abs(metrics[key]-reference['metrics'][key])<2e-7,(key,metrics[key],reference['metrics'][key])
    atomic_json(out/'metrics.json',dict(timestamp=now(),requested_spec=requested,spec=spec,smoke_only=False,
                evaluation_year=2023,checkpoint_epoch=ck['epoch'],checkpoint_sha256=sha(path),
                parameter_count=sum(p.numel() for p in model.parameters()),metrics=metrics,no_training=True))
    atomic_json(out/'SUCCESS.json',dict(timestamp=now(),passed=True,no_training=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);args=p.parse_args()
    try:main(args.job_id)
    except Exception:
        atomic_json(JOBS/args.job_id/'FAILURE.json',dict(timestamp=now(),error=traceback.format_exc()));raise
