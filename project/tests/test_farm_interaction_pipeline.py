from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_farm_interaction_enhancement_v1 as launch
import analyze_farm_interaction_enhancement_v1 as analyze
import run_farm_interaction_enhancement_v1_job as job_runner
from farm_enhancement_scheduler import admission_free
from wind_forcing.experiment_guard import read_freeze, require_test_authorized, sha256


@pytest.mark.parametrize("change,eligible", [({},True), ({"wins":3},False), ({"mean":0.},False),
    ({"rmse":0.00001},False), ({"h36":0.00051},False), ({"h36":0.0005},True)])
def test_screen_gate_uses_all_registered_constraints(monkeypatch, change, eligible):
    def paired(sa,ga,ra,sb,gb,rb,seeds):
        if ga == "m0": return np.full(5,-0.001)
        mean = change.get("mean", -0.002)
        return np.array([mean] * change.get("wins",5) + [0.0001] * (5-change.get("wins",5)))
    monkeypatch.setattr(launch,"paired_values",paired)
    monkeypatch.setattr(launch,"metric",lambda *a,**kw: {"station_mae_cf":0.1})
    monkeypatch.setattr(launch,"station_diagnostics",lambda stage,group,*a,**kw:
        {"rmse":0.1+(change.get("rmse",-0.001) if group!="m0" else 0),
         "h01":0.1,"h13":0.1,"h36":0.1+(change.get("h36",0) if group!="m0" else 0)})
    assert launch.screen_record("screen","a1")["eligible"] == eligible


def test_selection_never_reads_test_metrics():
    with pytest.raises(ValueError,match="validation-only"):
        launch.metric("screen","a1","int",42,"test")
    with pytest.raises(ValueError,match="validation-only"):
        launch.station_diagnostics("screen","a1","int",42,"test")


def test_resume_promotes_exactly_once_without_replacing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    spec=launch.Spec()
    for role in launch.INTERACTIONS:
        for seed in launch.SCREEN:
            run=launch.Job("screen","a1",role,seed,spec).run_dir
            run.mkdir(parents=True)
            for name in ("COMPLETE","best_model.pt","history.json","metrics_val.json","predictions_val.npz"):
                (run/name).touch()
    launch.promote("screen","formal","a1",spec)
    launch.promote("screen","formal","a1",spec)
    assert launch.complete(launch.Job("formal","a1","int",42,spec))
    other=tmp_path/"other"
    other.mkdir()
    with pytest.raises(RuntimeError,match="replace"):
        job_runner.safe_link(other,launch.Job("formal","a1","int",42,spec).run_dir)
    assert job_runner.next_attempt(other,"train") != job_runner.next_attempt(other,"train")
    config=other/"config.yaml"
    job_runner.immutable_config(config,{"seed":42})
    job_runner.immutable_config(config,{"seed":42})
    with pytest.raises(RuntimeError,match="config"):
        job_runner.immutable_config(config,{"seed":43})
    attempt=job_runner.next_attempt(other,"train")
    for name in ("best_model.pt","history.json"):
        (attempt/name).write_text("successful")
    job_runner.safe_link(attempt/"best_model.pt",other/"best_model.pt")
    assert job_runner.recover_publication(other,"train",("best_model.pt","history.json"),"TRAIN_COMPLETE")
    assert job_runner.recover_publication(other,"train",("best_model.pt","history.json"),"TRAIN_COMPLETE")


def make_freeze(root):
    (root/"selection").mkdir()
    ckpt=root/"best_model.pt"
    ckpt.write_bytes(b"frozen checkpoint")
    (root/"code_snapshot.json").write_text(json.dumps({"sha256":"code"}))
    payload={"protocol_version":2,"test_results_read":False,"selected_spec":{"context":"none"},
             "code_sha256":"code","registered_runs":[{"checkpoint":"best_model.pt","checkpoint_sha256":sha256(ckpt)}],
             "train_q90_ramp_thresholds":{"1":{"station_cf":0.1,"cluster_mw":1.}},
             "formal_pairs":[],"screen_only_pairs":[]}
    (root/"selection/validation_freeze.json").write_text(json.dumps(payload))
    return payload


def test_test_before_freeze_is_blocked_and_checkpoint_is_verified(tmp_path):
    cfg={"experiment":{"id":"farm_interaction_enhancement_v1/x","protocol_root":str(tmp_path)}}
    with pytest.raises(RuntimeError,match="freeze is missing"):
        require_test_authorized(cfg,tmp_path/"best_model.pt")
    make_freeze(tmp_path)
    require_test_authorized(cfg,tmp_path/"best_model.pt")
    with pytest.raises(RuntimeError,match="not registered"):
        require_test_authorized(cfg,tmp_path/"other.pt")
    (tmp_path/"best_model.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError,match="changed"):
        require_test_authorized(cfg,tmp_path/"best_model.pt")


def test_analyzer_excludes_screen_seeds_promoted_to_formal(tmp_path,monkeypatch):
    p=make_freeze(tmp_path)
    p["formal_pairs"]=[{"stage":"formal","group":"a1"}]
    p["screen_only_pairs"]=[{"stage":"screen","group":"a1"},{"stage":"screen","group":"b1"}]
    (tmp_path/"selection/validation_freeze.json").write_text(json.dumps(p))
    monkeypatch.setattr(analyze,"ROOT",tmp_path)
    formal,screen=analyze.load_pairs()
    assert [x["group"] for x in formal+screen]==["a1","b1"]


def test_paired_bootstrap_retains_pairing_and_validates_shapes():
    a=np.arange(10.)
    result=analyze.paired_summary(a,a+2)
    assert result["mean"]==result["ci_low"]==result["ci_high"]==-2
    assert result["wins"]==10
    with pytest.raises(ValueError): analyze.paired_summary(a,a[:5])
    day=(np.arange(3),np.array([2.,4.,6.]),np.array([1,2,3]))
    other=(np.arange(3),np.array([3.,6.,9.]),np.array([1,2,3]))
    result=analyze.day_paired_summary([day],[other])
    assert result["day_ci_low"]==result["day_ci_high"]==-1.


def test_station_ramps_do_not_equal_physical_sum_and_lags_include_history():
    target=np.zeros((1,2,24))
    pred=np.zeros_like(target)
    pred[:,0]=np.arange(1,25)*0.01
    pred[:,1]=-pred[:,0]/2
    history=np.zeros((1,2,24))
    metrics,_=analyze.array_metrics(pred,target,history,np.array([10.,20.]),{lag:(0.1,1.) for lag in (1,4,12)})
    assert metrics["station_ramp_15m_mae_cf"]==pytest.approx(0.0075)
    for lag,label in analyze.LAGS.items():
        expected=np.mean(np.minimum(np.arange(1,25),lag))*.0075
        assert metrics[f"station_ramp_{label}_mae_cf"]==pytest.approx(expected)
        assert metrics[f"cluster_ramp_{label}_mae_mw"]==pytest.approx(0.,abs=1e-12)
        assert metrics[f"cluster_ramp_{label}_direction_accuracy"] is None


def test_global_sample_mapping_and_lifetime_memory_reservation():
    assert analyze.aligned_positions(np.array([100,102,104]),np.array([104,100])).tolist()==[2,0]
    for bad in (np.array([105]),np.array([100,100])):
        with pytest.raises(ValueError): analyze.aligned_positions(np.array([100,102,104]),bad)
    assert admission_free(24000,24000,18000)==6000
    assert admission_free(3000,24000,18000)==3000


def test_job_evaluation_retry_reuses_successful_training(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner,"physical_gpu_uuid",lambda _:"GPU-physical-zero")
    monkeypatch.setenv("WIND_FARM_INTERACTION_ROOT", str(tmp_path))
    monkeypatch.setattr(sys,"argv",["job","0","screen","a1_int","42","lead_summary","legacy","none","faithful_concat","train"])
    calls=[]
    def run(command,**kwargs):
        train="train_forced_evolution.py" in command[1]
        calls.append("train" if train else "val")
        if train:
            output=Path(command[command.index("--output-dir")+1])
            (output/"best_model.pt").write_text("checkpoint")
            (output/"history.json").write_text("{}")
        elif calls.count("val")==1:
            raise subprocess.CalledProcessError(1,command)
        else:
            Path(command[command.index("--output")+1]).write_text("{}")
            Path(command[command.index("--predictions-output")+1]).write_text("archive")
    monkeypatch.setattr(job_runner.subprocess,"run",run)
    with pytest.raises(subprocess.CalledProcessError): job_runner.main()
    job_runner.main()
    job_runner.main()
    assert calls==["train","val","val"]
    run_path=tmp_path/"screen/a1_int/seed42"
    assert len(list((run_path/"attempts").iterdir()))==2
    assert (run_path/"COMPLETE").is_file()


@pytest.mark.parametrize("protocol",[2,3])
def test_analyzer_full_report_on_synthetic_frozen_archives(tmp_path,monkeypatch,protocol):
    root=tmp_path/"artifacts"
    root.mkdir()
    freeze=make_freeze(root)
    baseline={"stage":"baseline","group":"m0","spec":{"context":"none"},"seeds":[42,44],"parent_group":"m0"}
    candidate={"stage":"screen","group":"a1","spec":{"context":"none"},"seeds":[42,44],"parent_group":"m0"}
    pairs=[baseline,candidate]
    freeze.update(formal_pairs=[baseline],screen_only_pairs=[candidate],selected_stage="baseline",selected_group="m0",
                  train_q90_ramp_thresholds={str(lag):{"station_cf":0.1,"cluster_mw":1.} for lag in (1,4,12)})
    if protocol==3:
        bounded={**baseline,"group":"m0_bounded","parent_group":"m0_bounded",
                 "spec":{"context":"none","output":"bounded_cf_mix"}}
        bounded_final={**bounded,"stage":"output_robustness","group":"bounded_finalist"}
        pairs += [bounded,bounded_final]
        freeze.update(protocol_version=3,formal_pairs=pairs,screen_only_pairs=[],selected_group="a1",selected_stage="screen",
                      selected_role="int",bounded_finalist_group="bounded_finalist",
                      component_ablations=[{"group":"m0","parent_group":"a1","removed":"history"}])
        monkeypatch.setattr(analyze,"FORMAL",(42,44))
        monkeypatch.setattr(analyze,"SCREEN",(42,44))
    (root/"selection/validation_freeze.json").write_text(json.dumps(freeze))
    cache=tmp_path/"cache"
    (cache/"static").mkdir(parents=True)
    capacity=np.array([10.,20.])
    np.save(cache/"static/station_capacity_mw.npy",capacity)
    for split in ("val","test"):
        (cache/split).mkdir()
        target=np.full((2,2,24),0.3)
        np.save(cache/split/"station_future.npy",target)
        np.save(cache/split/"station_history.npy",np.full((2,2,24),0.2))
        np.save(cache/split/"sample_index.npy",np.array([100,102]))
        for pair in pairs:
            for role in ("int","iadd"):
                for seed in pair["seeds"]:
                    run=root/pair["stage"]/f"{pair['group']}_{role}"/f"seed{seed}"
                    run.mkdir(parents=True,exist_ok=True)
                    np.savez(run/f"predictions_{split}.npz",sample_index=np.array([100,102]),
                        origin_unix=np.array([86400,172800]),station_target_mw=target*capacity[None,:,None],
                        station_prediction_mw=(target+(.01 if role=="int" else .02))*capacity[None,:,None],
                        history_response_attention_mean=np.full((2,24,12),1/12))
                    if protocol==3 and pair["group"] in ("m0","a1"):
                        for mode in ("reference","weather"):
                            np.savez(run/f"predictions_{split}_{mode}.npz",sample_index=np.array([100,102]),
                                origin_unix=np.array([86400,172800]),station_target_mw=target*capacity[None,:,None],
                                station_prediction_mw=(target+.04)*capacity[None,:,None])
    monkeypatch.setattr(analyze,"ROOT",root)
    monkeypatch.setattr(analyze,"CACHE",cache)
    monkeypatch.setattr(analyze,"DOC",tmp_path/"results.md")
    monkeypatch.setattr(analyze,"baseline_table",lambda thresholds:[{"split":split,"model":"P","seed":"deterministic",
        "station_mae_cf":.1,"station_rmse_cf":.1,"physical_sum_cluster_mae_mw":3.} for split in ("val","test")])
    analyze.main()
    for name in ("seed_metrics.csv","formal_comparisons.csv","history_attention.csv","complete_baselines.csv","summary.json"):
        assert (root/"analysis"/name).stat().st_size>0
    summary=json.loads((root/"analysis/summary.json").read_text())
    assert summary["unique_training_runs"]==len(pairs)*4
    if protocol==3:
        assert len(summary["strengthening"]["interventions"])==16
        assert all(r["mean"]>0 for r in summary["strengthening"]["interventions"])
        assert len(summary["strengthening"]["component_ablations"])==4
        assert len(summary["strengthening"]["output_link_robustness"])==8
    assert "a1" in (tmp_path/"results.md").read_text()


@pytest.mark.parametrize("directions",[(),("a",),("a","b")])
def test_launcher_full_flow_degrades_and_freeze_resume_is_idempotent(tmp_path,monkeypatch,directions):
    monkeypatch.setattr(launch.subprocess,"check_output",lambda *a,**k:"0, GPU-zero, 24000, 20000")
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    monkeypatch.setattr(launch,"code_snapshot",lambda:{"sha256":"code","files":{}})
    (tmp_path/"cuda_smoke.json").write_text(json.dumps({"passed":True,"code_sha256":"code"}))
    monkeypatch.setattr(sys,"argv",["launcher","--gpus","0","--slots","8"])
    trained=[]
    def jobs(label,items,slots):
        for job in items:
            assert job.split=="train"
            if launch.complete(job):
                continue
            trained.append(job.key)
            job.run_dir.mkdir(parents=True)
            for name in ("COMPLETE","best_model.pt","history.json","metrics_val.json","predictions_val.npz","config.yaml"):
                (job.run_dir/name).write_text("{}")
    monkeypatch.setattr(launch,"run_jobs",jobs)
    def choose(label,stage,groups,**kw):
        winner=None
        if label=="temporal_screen" and "a" in directions: winner=groups[0]
        if label=="forcing_screen" and "b" in directions: winner=groups[0]
        if label in ("site_screen","id_diagnostic_screen","ab_screen","context_on_ab_screen","id_on_ab_diagnostic"): winner=groups[0]
        return winner,{}
    monkeypatch.setattr(launch,"choose",choose)
    monkeypatch.setattr(launch,"choose_readout",lambda *a,**kw:(None,{"records":[{"eligible":False},{"eligible":False}]}))
    monkeypatch.setattr(launch,"station_diagnostics",lambda *a,**kw:{"rmse":.1,"h01":.1,"h13":.1,"h36":.1})
    monkeypatch.setattr(launch,"metric",lambda stage,g,*a,**kw:{"station_mae_cf":0.001 if "c3" in g else 0.1})
    monkeypatch.setattr(analyze,"train_thresholds",lambda:{lag:(0.1,1.) for lag in (1,4,12)})
    finished=[]
    monkeypatch.setattr(launch,"finish_frozen",lambda freeze,slots:finished.append(freeze))
    launch.main()
    count=len(trained)
    frozen=(tmp_path/"selection/validation_freeze.json").read_bytes()
    assert finished[0]["selected_spec"]["context"]!="id_film"
    assert len(trained)==len(set(trained))
    if not directions: assert not any("context_screen" in k for k in trained)
    launch.main()
    assert len(trained)==count
    assert (tmp_path/"selection/validation_freeze.json").read_bytes()==frozen


def test_readout_selection_rejects_shape_without_capacity_advantage(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    def record(stage,group,**kwargs):
        return {"group":group,"eligible":kwargs["parent_group"]!="d0","int_station_mae_mean":.08 if group=="d1" else .09}
    monkeypatch.setattr(launch,"screen_record",record)
    winner,payload=launch.choose_readout("heads","screen",("d0","d1"))
    assert winner=="d0" and payload["records"][1]["capacity_control_eligible"] is False


@pytest.mark.parametrize("shape_rmse,expected",[(.06,"d1"),(.2,"d0")])
def test_primary_full_seed_selection_checks_shape_capacity_and_excludes_id(tmp_path,monkeypatch,shape_rmse,expected):
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    catalog={}
    launch.register_pair(catalog,"baseline","m0",launch.Spec(),launch.FORMAL,"baseline","m0")
    launch.register_pair(catalog,"formal","d0",launch.Spec(readout="capacity_aux"),launch.FORMAL,"baseline","m0")
    launch.register_pair(catalog,"formal","d1",launch.Spec(readout="level_shape"),launch.FORMAL,"formal","d0")
    launch.register_pair(catalog,"formal","id",launch.Spec(context="id_film"),launch.FORMAL,"baseline","m0")
    score={"m0":.10,"d0":.09,"d1":.08,"id":.01}
    monkeypatch.setattr(launch,"metric",lambda stage,g,role,*a,**kw:{"station_mae_cf":score[g]+(.01 if role=="iadd" else 0)})
    monkeypatch.setattr(launch,"station_diagnostics",lambda stage,g,*a,**kw:
        {"rmse":shape_rmse if g=="d1" else score[g],"h01":score[g],"h13":score[g],"h36":score[g]})
    best=launch.primary_selection(catalog)
    assert best["group"]==expected and best["role"]=="int"


@pytest.mark.parametrize("mode",["test","val_reference","test_reference","val_weather","test_weather"])
def test_intervention_jobs_are_frozen_mapped_and_idempotent(tmp_path,monkeypatch,mode):
    monkeypatch.setenv("WIND_FARM_INTERACTION_ROOT",str(tmp_path))
    monkeypatch.setattr(sys,"argv",["job","7","formal","m0_int","42","pooled_state","legacy","none","faithful_concat",mode,"level_shape","bounded_cf_mix"])
    calls=[]
    monkeypatch.setattr(job_runner,"physical_gpu_uuid",lambda g:"GPU-physical-seven" if g=="7" else None)
    with pytest.raises(RuntimeError,match="freeze is missing"):
        job_runner.main()
    frozen=[]
    monkeypatch.setattr(job_runner,"require_test_authorized",lambda config,checkpoint:frozen.append(checkpoint))
    def run(command,**kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="GPU-physical-seven"
        assert command[command.index("--split")+1]==mode.split("_")[0]
        assert ("--response-state-intervention" in command)==mode.endswith("_reference")
        assert ("--weather-permutation-seed" in command)==mode.endswith("_weather")
        Path(command[command.index("--output")+1]).write_text("{}")
        Path(command[command.index("--predictions-output")+1]).write_text("archive")
    monkeypatch.setattr(job_runner.subprocess,"run",run)
    job_runner.main()
    job_runner.main()
    assert len(calls)==1 and len(frozen)==2
    path=tmp_path/"formal/m0_int/seed42"
    assert all((path/n).is_file() for n in job_runner.output_names(mode))


def test_physical_gpu_query_and_live_reservations(monkeypatch):
    from farm_enhancement_scheduler import child_gpu_memory,unallocated_reservation
    def query(command,**kw):
        assert "--id=7" in command
        return "GPU-physical-seven\n"
    monkeypatch.setattr(job_runner.subprocess,"check_output",query)
    assert job_runner.physical_gpu_uuid("7")=="GPU-physical-seven"
    assert unallocated_reservation(2048,700)==1348
    assert unallocated_reservation(2048,3000)==0
    import os
    assert child_gpu_memory(os.getppid(),{os.getpid():700})==700
    # New free memory is available after external work ends, minus only our
    # not-yet-allocated budget (allocated bytes already reduce live free).
    assert 8000-unallocated_reservation(2048,700)==6652


def test_mse_level_shape_identity_and_internal_ramps():
    target=np.zeros((2,2,24))
    pred=np.arange(24)[None,None,:]*np.ones_like(target)/100
    metrics,_=analyze.array_metrics(pred,target,target,np.array([10.,20.]),{l:(.1,1.) for l in (1,4,12)})
    assert metrics["station_mse_cf2"]==pytest.approx(metrics["trajectory_mean_error_mse_cf2"]+metrics["trajectory_shape_error_mse_cf2"])
    assert metrics["station_ramp_3h_forecast_internal_mae_cf"]==pytest.approx(.12)
    assert metrics["station_ramp_3h_history_boundary_mae_cf"]==pytest.approx(.055)


def test_resume_detects_live_worker_lock_and_releases_it(tmp_path,monkeypatch):
    import fcntl
    from farm_enhancement_scheduler import job_is_locked
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    job=launch.Job("screen","a1","int",42,launch.Spec())
    job.run_dir.mkdir(parents=True)
    with (job.run_dir/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert job_is_locked(job)
    assert not job_is_locked(job)


def test_freeze_rejects_changed_evaluation_config_and_validation_metrics(tmp_path):
    import yaml
    freeze=make_freeze(tmp_path)
    config={"experiment":{"id":"farm_interaction_enhancement_v1/test","protocol_root":str(tmp_path)}}
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path/"metrics_val.json").write_text("{}")
    freeze["registered_runs"][0].update(config_sha256=sha256(tmp_path/"config.yaml"),
                                       validation_metrics_sha256=sha256(tmp_path/"metrics_val.json"))
    (tmp_path/"selection/validation_freeze.json").write_text(json.dumps(freeze))
    require_test_authorized(config,tmp_path/"best_model.pt")
    with pytest.raises(RuntimeError,match="differs from frozen"):
        require_test_authorized(dict(config,seed=43),tmp_path/"best_model.pt")
    (tmp_path/"metrics_val.json").write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match="validation metrics changed"):
        require_test_authorized(config,tmp_path/"best_model.pt")


def test_launcher_all_directions_shape_finalist_and_component_dedup(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    monkeypatch.setattr(launch,"code_snapshot",lambda:{"sha256":"code","files":{}})
    monkeypatch.setattr(launch.subprocess,"check_output",lambda *a,**k:"0, GPU-zero, 24000, 20000")
    (tmp_path/"cuda_smoke.json").write_text(json.dumps({"passed":True,"code_sha256":"code"}))
    monkeypatch.setattr(sys,"argv",["launcher","--gpus","0","--slots","8"])
    specs={}
    trained=[]
    def jobs(label,items,slots):
        for job in items:
            assert job.split=="train"
            specs[job.base_group]=job.spec
            if launch.complete(job): continue
            trained.append((job.spec,job.role,job.seed))
            job.run_dir.mkdir(parents=True)
            for name in (*job_runner.output_names("train"),"config.yaml"):
                (job.run_dir/name).write_text("{}")
    monkeypatch.setattr(launch,"run_jobs",jobs)
    def score(g):
        s=specs[g]
        if s.context=="id_film": return .01
        return (.10-.002*(s.history!="pooled_state")-.002*(s.forcing!="legacy")-.002*(s.context!="none")
                -{"legacy":0,"capacity_aux":.0005,"level_shape":.001}[s.readout])
    monkeypatch.setattr(launch,"metric",lambda stage,g,role,*a,**kw:{"station_mae_cf":score(g)+(.01 if role=="iadd" else 0)})
    monkeypatch.setattr(launch,"station_diagnostics",lambda stage,g,*a,**kw:{k:score(g) for k in ("rmse","h01","h13","h36")})
    monkeypatch.setattr(analyze,"train_thresholds",lambda:{lag:(.1,1.) for lag in (1,4,12)})
    finished=[]
    monkeypatch.setattr(launch,"finish_frozen",lambda freeze,slots:finished.append(freeze))
    launch.main()
    freeze=finished[0]
    assert freeze["selected_role"]=="int" and freeze["selected_spec"]["readout"]=="level_shape"
    assert freeze["selected_spec"]["context"]=="metadata_residual"
    assert len(freeze["component_ablations"])==4
    assert len(trained)==len(set(trained))==350
    assert len(freeze["registered_runs"])==350
    assert freeze["bounded_finalist_group"]=="bounded_finalist"
    launch.main()
    assert len(trained)==350
