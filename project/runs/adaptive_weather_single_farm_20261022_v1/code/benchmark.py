"""Measure actual full-step throughput at 1/2/4/8 workers before scheduling."""
from config import *
from models import Model
from data import Data
import argparse,subprocess,time,sys

def worker(directory,index):
 configure();seed_all(40+index);d=Data('train',6,7,'cuda:0',limit=2048);m=Model(spec('ST11',6,42,'benchmark',7)).cuda().train()
 opt=torch.optim.AdamW(m.groups(),weight_decay=1e-4,fused=True);args=d.batch(slice(0,256))
 def step():
  opt.zero_grad();loss=(m(*args)-d.y[:256]).abs().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 for _ in range(3):step()
 torch.cuda.synchronize();(directory/f'{index}.ready').touch()
 while not (directory/'go').exists():time.sleep(.05)
 start=time.perf_counter()
 for _ in range(30):step()
 torch.cuda.synchronize()
 atomic_json(directory/f'{index}.json',dict(seconds=time.perf_counter()-start,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,steps=30))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);ap.add_argument('--directory');ap.add_argument('--gpu',type=int,default=2);a=ap.parse_args()
 if a.worker is not None:return worker(Path(a.directory),a.worker)
 results={}
 for count in [1,2,4,8]:
  directory=RUN/'checks'/f'concurrency_{count}';directory.mkdir(exist_ok=True)
  for p in directory.glob('*'):p.unlink()
  ps=[]
  for i in range(count):
   log=open(directory/f'{i}.log','w');ps.append(subprocess.Popen([sys.executable,__file__,'--worker',str(i),'--directory',str(directory)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')));log.close()
  started=time.monotonic()
  while len(list(directory.glob('*.ready')))<count:
   if any(p.poll() is not None for p in ps) or time.monotonic()-started>240:
    for p in ps:
     if p.poll() is None:p.terminate()
    for p in ps:p.wait()
    raise RuntimeError('Benchmark worker failure/start timeout; inspect checks')
   time.sleep(.2)
  (directory/'go').touch()
  assert all(p.wait()==0 for p in ps)
  rows=[json.loads((directory/f'{i}.json').read_text()) for i in range(count)]
  results[count]=dict(aggregate_steps_per_second=30*count/max(x['seconds'] for x in rows),workers=rows);print(count,results[count]['aggregate_steps_per_second'],flush=True)
 best=max(results,key=lambda k:results[k]['aggregate_steps_per_second'])
 # Full train+val cycle caches require <400 MiB more than the subset benchmark.
 budget=max(x['peak_reserved_mib'] for r in results.values() for x in r['workers'])+1024
 atomic_json(RUN/'CONCURRENCY_BENCHMARK.json',dict(passed=True,results=results,recommended_workers=best,worker_budget_mib=budget*1.25,gpu=a.gpu,completed_at=now()))

if __name__=='__main__':main()
