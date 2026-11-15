from common import *
import argparse,subprocess,time
from models import baseline,correction

def worker(directory,index):
    configure_torch();seed_all(42+index);device='cuda:0';b=baseline(6,42)
    m=correction(6,42+index,128,'E2C_CBH',b.state_dict()).to(device).train();del b
    h=torch.rand(256,24,device=device);w=torch.rand(256,24,14,device=device);p=torch.ones(256,device=device,dtype=torch.long);base=torch.rand_like(h)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4,fused=True)
    def step():
        opt.zero_grad();m(h,w,p,base).square().mean().backward();opt.step()
    for _ in range(3):step()
    torch.cuda.synchronize();(directory/f'{index}.ready').touch()
    while not (directory/'go').exists():time.sleep(.05)
    t=time.perf_counter()
    for _ in range(100):step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    atomic_json(directory/f'{index}.json',dict(seconds=elapsed,steps_per_second=100/elapsed,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);ap.add_argument('--directory');a=ap.parse_args()
    if a.worker is not None:return worker(Path(a.directory),a.worker)
    results={}
    for count in [1,2,4,8]:
        directory=RUN/'concurrency_benchmark'/str(count);directory.mkdir(parents=True,exist_ok=True)
        for f in directory.glob('*'):f.unlink()
        ps=[];logs=[]
        for i in range(count):
            log=open(directory/f'{i}.log','w');logs.append(log)
            ps.append(subprocess.Popen([sys.executable,__file__,'--worker',str(i),'--directory',str(directory)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')))
        deadline=time.monotonic()+180
        while len(list(directory.glob('*.ready')))<count:
            if any(p.poll() is not None for p in ps):raise RuntimeError('Benchmark worker failed')
            if time.monotonic()>deadline:raise TimeoutError('Benchmark startup timeout')
            time.sleep(.2)
        (directory/'go').touch()
        for p in ps:
            assert p.wait()==0
        for log in logs:log.close()
        rows=[json.loads((directory/f'{i}.json').read_text()) for i in range(count)]
        throughput=100*count/max(r['seconds'] for r in rows)
        results[count]=dict(aggregate_steps_per_second=throughput,workers=rows)
        print(count,throughput,flush=True)
    best=max(results,key=lambda k:results[k]['aggregate_steps_per_second'])
    # Prefer smaller concurrency if within 5% of maximum observed throughput.
    threshold=results[best]['aggregate_steps_per_second']*.95
    recommended=min(k for k in results if results[k]['aggregate_steps_per_second']>=threshold)
    atomic_json(RUN/'CONCURRENCY_BENCHMARK.json',dict(results=results,recommended_workers=recommended,
        scope='GPU 0, width 128 CBH, batch 256, existing unrelated workloads retained'))

if __name__=='__main__':main()
