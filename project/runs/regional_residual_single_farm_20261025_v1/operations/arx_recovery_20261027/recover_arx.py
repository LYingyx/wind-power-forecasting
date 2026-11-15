"""Versioned ARX recovery, same data/features/objective/tolerance; resumable QR solve."""
from pathlib import Path
import sys, time, json, os, warnings, threading, socket, argparse
HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
sys.path.insert(0, str(RUN / 'operations'))
sys.path.insert(0, str(RUN / 'code'))
from relocate import activate
activate(RUN)
from settings import JOBS, setup, now, atomic_json, sha
from physical_data import FarmData
from sparse_arx import features
from train_physics import metrics
from diagnostics import stratified
from qr_solver import compress, model as make_model, certificate
import numpy as np
import torch

SOLVER = 'augmented_QR_multitask_elastic_net_v1'
ALPHAS = [.003, .001, .0003]


def save_npz(path, values):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(tmp, path)


def run(spec, device='cpu', smoke=False, output=None):
    assert device == 'cpu' and not smoke
    setup(); torch.set_num_threads(1)
    out = Path(output) if output else JOBS / spec['id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'SUCCESS.json').exists():
        return
    assert (HERE / 'QR_EQUIVALENCE.json').exists()
    assert json.loads((HERE / 'QR_EQUIVALENCE.json').read_text())['passed']
    atomic_json(out / 'spec.json', spec)
    start = time.monotonic()
    live = dict(id=spec['id'], solver=SOLVER, host=socket.gethostname(), pid=os.getpid(), phase='loading')
    stop = threading.Event()
    def heartbeat():
        while not stop.is_set():
            atomic_json(out / 'QR_HEARTBEAT.json', dict(live, timestamp=now(), elapsed_seconds=time.monotonic()-start))
            stop.wait(10)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    state_path = out / 'QR_PROGRESS.json'
    checkpoint_path = out / 'QR_CHECKPOINT.npz'
    state = dict(solver=SOLVER, curves={}, selections={}, details=[], completed_families=[],
                 current_family=None, alpha_index=0, alpha_iterations=0, max_iter=3000,
                 best=None, accumulated_seconds=0.)
    arrays = {}
    if checkpoint_path.exists():
        with np.load(checkpoint_path) as z:
            state = json.loads(z['_checkpoint_state'].tobytes().decode())
            assert state['solver'] == SOLVER
            arrays = {k:z[k].copy() for k in z.files if k != '_checkpoint_state'}
    previous_seconds = state['accumulated_seconds']
    def checkpoint():
        state['accumulated_seconds'] = previous_seconds + time.monotonic() - start
        state['timestamp'] = now()
        payload = dict(arrays, _checkpoint_state=np.frombuffer(json.dumps(state,allow_nan=False).encode(),dtype='u1'))
        save_npz(checkpoint_path, payload)
        state['checkpoint_sha256'] = sha(checkpoint_path)
        atomic_json(state_path, state)
    try:
        train = FarmData('train', spec['site'], 'N0')
        val = FarmData('val', spec['site'], 'N0', stats=train.stats)
        xtrain, xval = features(train), features(val)
        target = train.y.numpy() - train.h.numpy()[:, -1:]
        for name in xtrain:
            if name in state['completed_families']:
                continue
            x = xtrain[name].astype('f8'); mu = x.mean(0); sd = x.std(0)
            active = sd > 1e-6; sd = np.maximum(sd, 1e-6)
            x = np.asfortranarray(((x-mu)/sd)[:,active])
            xv = np.asfortranarray(((xval[name]-mu)/sd)[:,active])
            live.update(phase='QR', family=name)
            t = time.monotonic(); xc, yc, xm, ym = compress(x, target); qr_seconds = time.monotonic()-t
            # Check full sufficient statistics, including the target residual energy.
            original = np.concatenate([x-xm, target.astype('f8')-ym], 1)
            compact = np.concatenate([xc,yc],1)
            gram_error = float(np.max(np.abs(original.T@original/len(original)-compact.T@compact/len(compact))))
            del original, compact
            assert gram_error < 1e-10
            if state['current_family'] != name:
                state.update(current_family=name, alpha_index=0, alpha_iterations=0, max_iter=3000, best=None)
                state['curves'][name] = []
                for k in ['_current_coef','_best_coef','_best_intercept']:
                    arrays.pop(k,None)
            model = make_model(max_iter=state['max_iter'])
            if '_current_coef' in arrays:
                model.coef_ = np.asfortranarray(arrays['_current_coef'])
            for index in range(state['alpha_index'], len(ALPHAS)):
                alpha = ALPHAS[index]
                model.set_params(alpha=alpha, max_iter=state['max_iter'])
                while True:
                    live.update(phase='solve', family=name, alpha=alpha, iterations_completed=state['alpha_iterations'])
                    t = time.monotonic()
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter('always')
                        model.fit(xc,yc)
                    elapsed = time.monotonic()-t
                    state['alpha_iterations'] += int(model.n_iter_)
                    arrays['_current_coef'] = model.coef_.copy()
                    cert = certificate(xc,yc,model.coef_,alpha)
                    assert abs(cert['dual_gap']-float(model.dual_gap_)) < 1e-9
                    converged = bool(model.n_iter_ < model.max_iter and cert['dual_gap'] <= 1e-4 * cert['target_energy'] * (1+1e-8))
                    print(spec['id'],name,'alpha',alpha,'iterations',state['alpha_iterations'],
                          'gap',cert['dual_gap'],'limit',1e-4*cert['target_energy'],
                          'seconds',elapsed,'converged',converged,flush=True)
                    live.update(iterations_completed=state['alpha_iterations'],dual_gap=cert['dual_gap'],
                                gap_limit=1e-4*cert['target_energy'],last_chunk_seconds=elapsed)
                    if converged:
                        break
                    state['max_iter'] = 30000; model.set_params(max_iter=30000)
                    checkpoint()
                    assert state['alpha_iterations'] < 1000000, 'Unconverged after one million iterations; no tolerance relaxation'
                intercept = ym - xm @ model.coef_.T
                pred = xv @ model.coef_.T + intercept + val.h.numpy()[:,-1:]
                score = metrics(pred,val.y.numpy())
                state['curves'][name].append(dict(alpha=alpha,n_iter=int(model.n_iter_),
                    total_iterations=state['alpha_iterations'],mae=score['mae'],rmse=score['rmse'],
                    certificate=cert,compressed_rows=len(xc),original_rows=len(x),
                    gram_max_error=gram_error,qr_seconds=qr_seconds))
                if state['best'] is None or score['mae'] < state['best']['score']['mae']:
                    state['best'] = dict(alpha=alpha,score=score)
                    arrays['_best_coef'] = model.coef_.copy(); arrays['_best_intercept'] = intercept.copy()
                state.update(alpha_index=index+1,alpha_iterations=0)
                checkpoint()
            best = state['best']; coef = arrays['_best_coef']; intercept = arrays['_best_intercept']
            arrays.update({name+'_coef':coef.copy(),name+'_intercept':intercept.copy(),
                           name+'_mean':mu,name+'_std':sd,name+'_active':active})
            state['selections'][name] = dict(epoch=0,alpha=best['alpha'],metrics=best['score'],
                nonzero=int(np.count_nonzero(coef)),active_features=int(active.sum()))
            pred = xv @ coef.T + intercept + val.h.numpy()[:,-1:]
            state['details'].extend(dict(family=name,**row) for row in stratified(pred,val))
            state['completed_families'].append(name)
            state.update(current_family=None,alpha_index=0,alpha_iterations=0,best=None)
            for key in ['_current_coef','_best_coef','_best_intercept']:
                arrays.pop(key,None)
            checkpoint()
        live['phase']='finalizing'
        save_npz(out/'coefficients.npz',arrays)
        atomic_json(out/'diagnostics.json',dict(year=2023,rows=state['details']))
        atomic_json(out/'alpha_selection.json',state['curves'])
        atomic_json(out/'metrics.json',dict(**spec,finished=now(),seconds=previous_seconds+time.monotonic()-start,
            smoke_only=False,selections=state['selections'],artifact_sha256=sha(out/'coefficients.npz'),
            n_train=train.n,n_val=val.n,solver_revision=SOLVER,
            solver_validation_sha256=sha(HERE/'QR_EQUIVALENCE.json'),
            tolerance=1e-4,original_objective_preserved=True,iteration_budget_extended=True))
        atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=False,solver_revision=SOLVER))
        checkpoint_path.unlink(missing_ok=True)
        state['checkpoint_removed_after_success']=True;atomic_json(state_path,state)
        live['phase']='complete'
    finally:
        stop.set();thread.join(timeout=11)
        atomic_json(out/'QR_HEARTBEAT.json',dict(live,timestamp=now(),elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=int,required=True);parser.add_argument('--output')
    args=parser.parse_args();os.nice(15)
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==f'ARX_s{args.site:02d}')
    run(spec,output=args.output)
