"""Audit completed validation screening without changing frozen scientific sources."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,math,os,shlex,subprocess,sys
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now,sha


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=2e-12)


def check_metrics(m,n):
    assert m['n']==n and n>0
    assert all(math.isfinite(m[k]) for k in ['mae','rmse','bias','sq_sum'])
    assert m['mae']>=0 and m['rmse']>=0 and m['sq_sum']>=0
    assert close(m['rmse']**2*n,m['sq_sum'])
    assert m['mae']<=m['rmse']+1e-10 and abs(m['bias'])<=m['mae']+1e-10


def remote_inventory(host):
    rr='/home/liying/wind_physics_project/project/runs/'+RUN.name
    script=f'''from pathlib import Path
import json,subprocess
r=Path({rr!r});out={{'success':[],'archived':[],'leftover_weights':[]}}
for p in (r/'jobs').glob('*/SUCCESS.json'):out['success'].append(p.parent.name)
for p in (r/'jobs').glob('*/ARCHIVED.json'):
 out['archived'].append(p.parent.name)
 if list(p.parent.glob('epoch_*.pt')) or (p.parent/'latest.pt').exists():out['leftover_weights'].append(p.parent.name)
out['leased']=json.loads((r/'operations/LEASED_JOBS.json').read_text())['jobs']
out['processes']=subprocess.run(['pgrep','-af','[r]esource_scheduler.py|[s]cheduled_worker.py|[b]ootstrap_remote.py'],capture_output=True,text=True).stdout
out['compatibility']=json.loads((r/'operations/GPU_COMPATIBILITY.json').read_text())
print(json.dumps(out))'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
    return json.loads(result.stdout)


def main():
    manifest=json.loads((RUN/'manifest.json').read_text());ids={s['id'] for s in manifest}
    assert len(ids)==len(manifest)==3432
    frozen=json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files']
    assert all(sha(p)==h for p,h in frozen.items())
    success={p.parent.name for p in JOBS.glob('*/SUCCESS.json')};assert success==ids
    assert not list(JOBS.glob('*/FAILURE.json'))
    assert not list(JOBS.glob('*/REMOTE_RESERVATION.json'))
    results={};groups=collections.Counter();returned=collections.defaultdict(set);checked=set();missing_runtime=[];finished=[];trajectory_counts=collections.Counter()
    for spec in manifest:
        out=JOBS/spec['id'];m=json.loads((out/'metrics.json').read_text());s=json.loads((out/'SUCCESS.json').read_text())
        assert json.loads((out/'spec.json').read_text())==spec
        assert all(m[k]==v for k,v in spec.items())
        assert not m['smoke_only'] and not s['smoke_only'] and m['precision']=='float32_no_tf32'
        assert m['n_train']>0 and m['n_val']>0
        hist=json.loads((out/'history.json').read_text());assert len(hist)==m['epochs'] and 1<=len(hist)<=80
        assert [h['epoch'] for h in hist]==list(range(1,len(hist)+1))
        assert len(hist)==80 or hist[-1]['stale']>=10
        assert set(m['selections'])=={'mae','rmse'}
        selected=json.loads((out/'selection.json').read_text());assert set(selected)=={'mae','rmse'}
        for criterion,sel in m['selections'].items():
            best=min(hist,key=lambda h:h['val_'+criterion])
            assert sel['epoch']==best['epoch'] and close(sel['value'],best['val_'+criterion])
            assert all(selected[criterion][k]==sel[k] for k in ['epoch','checkpoint','value'])
            assert close(sel['metrics'][criterion],sel['value'])
            assert all(close(sel['metrics'][k],hist[sel['epoch']-1]['val_'+k]) for k in ['mae','rmse'])
            path=out/sel['checkpoint'];assert path.parent==out and path.name==f"epoch_{sel['epoch']:03d}.pt"
            if path not in checked:assert sha(path)==sel['sha256'];checked.add(path)
            else:assert sha(path)==sel['sha256']
            score=sel['metrics'];check_metrics(score,m['n_val']*24)
            for label,leads in [('0_1h',4),('1_3h',8),('3_6h',12)]:check_metrics(score['horizons'][label],m['n_val']*leads)
            assert close(sum(v['sq_sum'] for v in score['horizons'].values()),score['sq_sum'])
            for key in ['mae','bias']:assert close(sum(v[key]*v['n'] for v in score['horizons'].values())/score['n'],score[key])
            assert len(score['lead_mae'])==len(score['lead_rmse'])==24
            assert close(sum(score['lead_mae'])/24,score['mae'])
            assert close(sum(x*x for x in score['lead_rmse'])/24,score['rmse']**2)
        assert {f.name for f in out.glob('*.pt')}=={s['checkpoint'] for s in m['selections'].values()}
        assert not list(out.glob('*.tmp'))
        expected={'neighbor_identity','neighbor_reverse','neighbor_zero'} if spec['variant'].startswith('NEIGH_') else set()
        if spec['variant']=='NEIGH_ADVECT':expected.add('direction_rotate')
        assert set(m['interventions'])==expected
        for intervention in m['interventions'].values():
            check_metrics(intervention,m['n_val']*24)
            for label,leads in [('0_1h',4),('1_3h',8),('3_6h',12)]:check_metrics(intervention['horizons'][label],m['n_val']*leads)
        if s.get('verified_return'):returned[s['source_host']].add(spec['id'])
        if not (out/'runtime.json').exists():missing_runtime.append(spec['id'])
        results[spec['id']]=m;groups[spec['stage']]+=1;finished.append(m['finished']);trajectory_counts[m['epochs']]+=1
    # Independently validate every exported score and paired row against original JSON.
    table=RUN/'analysis/new_actual_per_farm_seed.csv';keys=set();actual_rows=0
    with table.open() as f:
        for row in csv.DictReader(f):
            key=(row['id'],row['selection'],row['horizon']);assert key not in keys;keys.add(key)
            m=results[row['id']];sel=m['selections'][row['selection']]
            value=sel['metrics'] if row['horizon']=='all' else sel['metrics']['horizons'][row['horizon']]
            for metric in ['n','mae','rmse','bias']:assert close(row[metric],value[metric])
            assert int(row['epoch'])==sel['epoch']
            assert int(row['site'])==m['site'] and int(row['seed'])==m['seed']
            assert Path(row['checkpoint']).resolve()==(JOBS/row['id']/sel['checkpoint']).resolve()
            actual_rows+=1
    assert actual_rows==len(manifest)*2*4
    pair_rows=0;pair_keys=set()
    with (RUN/'analysis/new_paired_per_farm_seed.csv').open() as f:
        for row in csv.DictReader(f):
            k=(row['candidate'],row['reference'],row['selection'],row['horizon']);assert k not in pair_keys;pair_keys.add(k)
            a=results[row['candidate']];b=results[row['reference']]
            assert a['site']==b['site']==int(row['site']) and a['seed']==b['seed']==int(row['seed'])
            vals=[]
            for m in [a,b]:
                v=m['selections'][row['selection']]['metrics'];vals.append(v if row['horizon']=='all' else v['horizons'][row['horizon']])
            for metric in ['mae','rmse']:
                assert close(row['candidate_'+metric],vals[0][metric]) and close(row['reference_'+metric],vals[1][metric])
                assert close(row[metric+'_difference'],vals[0][metric]-vals[1][metric])
                assert (row[metric+'_better']=='True')==(vals[0][metric]<vals[1][metric])
            pair_rows+=1
    remotes={h:remote_inventory(h) for h in ['v1006001','v1006002']}
    for host,remote in remotes.items():
        assert set(remote['success'])==set(remote['archived'])==returned[host]
        assert not remote['leftover_weights'] and not remote['leased'] and not remote['processes']
        assert remote['compatibility']['passed']
    assert not (returned['v1006001'] & returned['v1006002'])
    workers=subprocess.run(['pgrep','-af','[r]esource_scheduler.py|[s]cheduled_worker.py|[r]emote_controller.py'],capture_output=True,text=True).stdout
    assert not workers,workers
    audit=dict(timestamp=now(),passed=True,scope='2022 training / 2023 validation screening only',planned=len(manifest),completed=len(results),failed=0,active=0,pending=0,
        stage_counts=dict(groups),scientific_frozen_files_checked=len(frozen),selected_checkpoint_files_checked=len(checked),
        exact_manifest_and_spec=True,history_selection_verified=True,metric_horizon_and_lead_consistency=True,
        actual_csv_rows=actual_rows,paired_csv_rows=pair_rows,csv_values_verified=True,
        verified_remote_returns={h:len(v) for h,v in returned.items()},remote_duplicate_weights=0,live_training_or_scheduler_processes=0,
        initial_jobs_without_runtime_metadata=len(missing_runtime),initial_jobs_without_runtime_ids=missing_runtime,
        first_job_finished=min(finished),last_job_finished=max(finished),test_year_accessed=False,
        limitations=['Predictions were not newly recomputed; audit checks stored metrics, selections and checkpoint hashes.','Initial jobs predating operational wrapper have no runtime.json; do not infer exact host/GPU provenance.','2024 reevaluation and expansion of structural variants to all 56 farms are not part of this completion.'])
    atomic_json(RUN/'COMPLETION_AUDIT.json',audit)
    status=dict(timestamp=now(),planned=len(manifest),completed=len(results),failed=0,active=[],pending=0,failed_jobs=[],source_freeze=True,diagnostics_complete=True,test_year_accessed=False,enabled_hosts=['user-SYS-420GP-TNR','v1006001','v1006002'],completion_audit='COMPLETION_AUDIT.json')
    old=json.loads((RUN/'SCREENING_COMPLETE.json').read_text())
    audit['corrected_stale_completion_counts']=dict(previous_active=len(old.get('active',[])),previous_pending=old.get('pending'))
    atomic_json(RUN/'COMPLETION_AUDIT.json',audit)
    for name in ['STATUS.json','SCREENING_COMPLETE.json']:atomic_json(RUN/name,status)
    atomic_json(RUN/'FINISHED.json',dict(timestamp=now(),scope=audit['scope'],all_screening_jobs_complete=True,audit_passed=True,planned=len(manifest),completed=len(results),failed=0,last_job_finished=max(finished),test_year_accessed=False))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
