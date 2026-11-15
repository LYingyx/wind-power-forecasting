from __future__ import annotations
import os, subprocess, time, yaml, shutil
from pathlib import Path

PROJECT=Path('/storage/liying/new_try/wind_cluster_forced_evolution')
ROOT=PROJECT/'artifacts/trajectory_conditioned_correction_20260913'
BASE=PROJECT/'configs/incremental_correction_pad5.yaml'
SEEDS=(42,46,50)
# C0-C4 are the complete screening matrix for the trajectory-conditioned claim.
VARIANTS={
    'C0_weather_only': dict(correction_conditioning='weather_only', trajectory_conditioning='none', trajectory_interaction=False),
    'C1_level': dict(correction_conditioning='weather_only', trajectory_conditioning='level', trajectory_interaction=False),
    'C2_level_delta': dict(correction_conditioning='weather_only', trajectory_conditioning='level_delta', trajectory_interaction=False),
    'C3_explicit_interaction': dict(correction_conditioning='weather_only', trajectory_conditioning='level_delta', trajectory_interaction=True),
    'C4_hidden_plus_trajectory': dict(correction_conditioning='history_weather', trajectory_conditioning='level_delta', trajectory_interaction=False),
}

def config_for(name, seed):
    c=yaml.safe_load(BASE.read_text())
    c['seed']=seed
    c['data']['cache_dir']=str(PROJECT/'data/model_cache_pad5_f8')
    c['data']['source_h5']=str(PROJECT/'data/jilin_56farm_pad8.h5')
    m=c['model']
    m.update(VARIANTS[name])
    m['architecture']='incremental_correction'
    m['correction_fusion_mode']='blackbox'
    m['freeze_history_baseline']=True
    m['initialize_history_baseline_from_checkpoint']=True
    m['baseline_anchor_enabled']=False
    # Main objective is station CF MAE; cluster loss is disabled for this study.
    t=c['training']['deterministic']
    t['station_loss_weight']=1.0
    t['cluster_loss_weight']=0.0
    t['checkpoint_metric']='station_mae_cf'
    t['auxiliary_weather_loss_weight']=0.0
    c['training']['checkpoint_metric']='station_mae_cf'
    c['experiment']={'id':f'trajectory_conditioned_correction_20260913/{name}/seed{seed}'}
    return c

def checkpoint(seed):
    if seed == 42:
        p=ROOT/'history_prior/seed42/best_model.pt'
    else:
        p=PROJECT/f'artifacts/incremental_mechanism_v2/history_prior/seed{seed}/best_model.pt'
    if seed != 42 and not p.is_file(): raise FileNotFoundError(p)
    return p

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest={'created_unix':int(time.time()),'project':str(PROJECT),'variants':list(VARIANTS),'seeds':list(SEEDS),'protocol':'station_mae_cf; cluster_loss=0; 2023 validation selection','slots':{0:7,1:7,2:7,3:7,4:2,5:2,6:2,7:2}}
    (ROOT/'run_manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
    jobs=[]
    for name in VARIANTS:
      for seed in SEEDS:
        rd=ROOT/'runs'/name/f'seed{seed}'; rd.mkdir(parents=True,exist_ok=True)
        cfg=rd/'config.yaml'; cfg.write_text(yaml.safe_dump(config_for(name,seed),sort_keys=False))
        jobs.append((name,seed,rd,cfg,checkpoint(seed)))
    # One history-only checkpoint is absent from the reused archive. Train it
    # in parallel as a bootstrap job; correction jobs for seed42 are released
    # only after this checkpoint is complete.
    bootstrap_out=ROOT/'history_prior'/'seed42'
    bootstrap_ckpt=bootstrap_out/'best_model.pt'
    bootstrap=None
    if not bootstrap_ckpt.is_file():
        bootstrap_out.mkdir(parents=True,exist_ok=True)
        blog=(bootstrap_out/'launcher.log').open('a')
        benv=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(PROJECT/'src'))
        bcmd=['/data/liying/anaconda3/envs/chronos/bin/python','-B',str(PROJECT/'scripts/train_forced_evolution.py'),'--config',str(PROJECT/'configs/history_prior_for_trajectory_20260913.yaml'),'--output-dir',str(bootstrap_out),'--device','cuda']
        bootstrap=subprocess.Popen(bcmd,cwd=PROJECT,env=benv,stdout=blog,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'[bootstrap] history-only seed42 gpu7 pid={bootstrap.pid}',flush=True)
    # Launch bounded workers. Four 48 GB cards use 7 slots; four 24 GB cards
    # use 2 slots because the data/loader footprint makes 8-10 unsafe.
    slots=[]
    for gpu,count in manifest['slots'].items():
        # Reserve one 24-GiB slot while bootstrap occupies GPU 7.
        reserve=1 if bootstrap is not None and gpu==7 else 0
        slots += [gpu]*max(0,count-reserve)
    pending=[j for j in jobs if j[1]!=42 and not ((j[2]/'COMPLETE').is_file() and (j[2]/'best_model.pt').is_file())]
    delayed=[j for j in jobs if j[1]==42 and not ((j[2]/'COMPLETE').is_file() and (j[2]/'best_model.pt').is_file())]
    active=[]; next_slot=0
    while pending or delayed or active or bootstrap is not None:
      if bootstrap is not None and bootstrap.poll() is not None:
        print(f'[bootstrap] history-only seed42 exit={bootstrap.returncode}',flush=True)
        bootstrap=None
        if (ROOT/'history_prior/seed42/best_model.pt').is_file():
            pending.extend(delayed); delayed=[]
            slots.append(7)
        else:
            print('[bootstrap] checkpoint missing; seed42 correction jobs remain blocked',flush=True)
            delayed=[]
      # Fill free slots.
      while pending and len(active)<len(slots):
        name,seed,rd,cfg,hist=pending.pop(0)
        gpu=slots[next_slot % len(slots)]; next_slot += 1
        env=dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(PROJECT/'src'))
        log=(rd/'launcher.log').open('a')
        cmd=['/data/liying/anaconda3/envs/chronos/bin/python','-B',str(PROJECT/'scripts/train_forced_evolution.py'),'--config',str(cfg),'--history-checkpoint',str(hist),'--output-dir',str(rd),'--device','cuda']
        p=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (rd/'pid').write_text(str(p.pid)); (rd/'assigned_gpu').write_text(str(gpu))
        active.append((p,rd,log))
        print(f'[launch] {name} seed{seed} gpu{gpu} pid={p.pid}',flush=True)
      if active:
        time.sleep(5)
        still=[]
        for p,rd,log in active:
          code=p.poll()
          if code is None: still.append((p,rd,log))
          else:
            (rd/'exit_code').write_text(str(code)); log.close()
            if code==0: print(f'[done] {rd.relative_to(ROOT)}',flush=True)
            else: print(f'[failed] {rd.relative_to(ROOT)} exit={code}',flush=True)
        active=still
      elif bootstrap is not None:
        time.sleep(5)
    print('[launch] queue submitted; child processes continue independently',flush=True)
if __name__=='__main__': main()
