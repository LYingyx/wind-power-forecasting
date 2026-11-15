from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import subprocess
import time
from typing import Iterable, Sequence

import numpy as np
from wind_forcing.experiment_guard import read_freeze, sha256
from run_farm_interaction_enhancement_v1_job import safe_link, output_names

from launch_aggregate_cluster_experiments import (
    WorkerSlot,
    _looks_like_oom,
    atomic_json,
    build_worker_slots,
    execute_parallel,
    gpu_state,
    parse_slot_counts,
)


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "farm_interaction_enhancement_v1"
SCREEN = (42, 44, 46, 48, 50)
FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL if seed not in SCREEN)
INTERACTIONS = {
    "iadd": "independent_output_additive",
    "int": "faithful_concat",
}


@dataclass(frozen=True)
class Spec:
    history: str = "pooled_state"
    forcing: str = "legacy"
    context: str = "none"
    readout: str = "legacy"
    output: str = "cf_residual"


BASE_CANDIDATES = {
    "a1_lead_summary": Spec(history="lead_summary"),
    "a2_cross_attention": Spec(history="weather_cross_attention"),
    "b1_origin_delta_linear": Spec(forcing="origin_delta_linear"),
    "b2_origin_delta_tcn": Spec(forcing="origin_delta_tcn"),
    "c1_metadata_residual": Spec(context="metadata_residual"),
    "c2_metadata_film": Spec(context="metadata_film"),
    "c3_id_film": Spec(context="id_film"),
    "d0_capacity_aux": Spec(readout="capacity_aux"),
    "d1_level_shape": Spec(readout="level_shape"),
}


@dataclass(frozen=True)
class Job:
    stage: str
    base_group: str
    role: str
    seed: int
    spec: Spec
    split: str = "train"

    @property
    def key(self) -> str:
        return f"{self.stage}/{self.group}/seed{self.seed}/{self.split}"

    @property
    def group(self) -> str:
        return f"{self.base_group}_{self.role}"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"


def complete(job: Job) -> bool:
    names = output_names(job.split)
    return all((job.run_dir / name).is_file() for name in names)


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    from farm_enhancement_scheduler import execute
    materialized = list(jobs)
    if any(job.split != "train" for job in materialized):
        read_freeze(ROOT)
    smoke = json.loads((ROOT / "cuda_smoke.json").read_text())
    memory_mib = max(2048, int(max(row["peak_reserved_mib"] for row in smoke["models"]) * 1.5) + 1024)

    def command(job, slot):
        return ["bash", str(PROJECT / "shell/run_farm_interaction_enhancement_v1_job.sh"),
                str(slot.gpu), job.stage, job.group, str(job.seed), job.spec.history,
                job.spec.forcing, job.spec.context, INTERACTIONS[job.role], job.split,
                job.spec.readout, job.spec.output]

    execute(label, materialized, slots, ROOT, command, complete, memory_mib=memory_mib)


def link_baseline() -> None:
    sources = {
        "m0_iadd": PROJECT / "artifacts/local_state_cluster_transfer_v2/station/b_independent_add",
        "m0_int": PROJECT / "artifacts/local_state_cluster_transfer_v2/station/b_int",
    }
    target_root = ROOT / "reused"
    target_root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = target_root / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected baseline link: {target}")
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)


def run_path(stage: str, base_group: str, role: str, seed: int) -> Path:
    if base_group == "m0":
        return ROOT / "baseline" / f"m0_{role}" / f"seed{seed}"
    return ROOT / stage / f"{base_group}_{role}" / f"seed{seed}"


def metric(stage: str, base_group: str, role: str, seed: int, split: str = "val") -> dict:
    if split != "val":
        raise ValueError("Selection metrics are validation-only")
    values = json.loads(
        (run_path(stage, base_group, role, seed) / f"metrics_{split}.json").read_text()
    )
    values["station_mae_cf"] = station_diagnostics(stage, base_group, role, seed)["mae"]
    return values


def station_diagnostics(
    stage: str, base_group: str, role: str, seed: int, split: str = "val"
) -> dict[str, float]:
    if split != "val":
        raise ValueError("Selection diagnostics are validation-only")
    path = run_path(stage, base_group, role, seed)
    cached = path / "selection_metrics_float64.json"
    signature = {name: [int((path / name).stat().st_size), int((path / name).stat().st_mtime_ns)]
                 for name in (f"predictions_{split}.npz",)}
    if cached.exists():
        value = json.loads(cached.read_text())
        if value.get("signature") == signature:
            return value["metrics"]
    capacity = np.load(
        PROJECT / "data/model_cache_pad5_f8/static/station_capacity_mw.npy",
        allow_pickle=False,
    )
    with np.load(path / f"predictions_{split}.npz", allow_pickle=False) as archive:
        sample = archive["sample_index"]
        expected = np.load(PROJECT / "data/model_cache_pad5_f8/val/sample_index.npy")
        if len(sample) != len(expected) or not np.array_equal(np.sort(sample), expected):
            raise ValueError("Incomplete validation sample coverage")
        error = (archive["station_prediction_mw"].astype(np.float64) - archive["station_target_mw"]
                 ) / capacity[None, :, None]
    if not np.isfinite(error).all():
        raise ValueError("Non-finite validation predictions")
    lead = np.mean(np.abs(error), axis=(0, 1), dtype=np.float64)
    result = {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), axis=(0, 2), dtype=np.float64)).mean()),
        "h01": float(lead[:4].mean()),
        "h13": float(lead[4:12].mean()),
        "h36": float(lead[12:].mean()),
    }
    atomic_json(cached, {"signature": signature, "metrics": result})
    return result


def paired_values(
    stage_a: str,
    group_a: str,
    role_a: str,
    stage_b: str,
    group_b: str,
    role_b: str,
    seeds: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [
            metric(stage_a, group_a, role_a, seed)["station_mae_cf"]
            - metric(stage_b, group_b, role_b, seed)["station_mae_cf"]
            for seed in seeds
        ],
        dtype=np.float64,
    )


def screen_record(
    stage: str,
    base_group: str,
    *,
    parent_stage: str = "baseline",
    parent_group: str = "m0",
) -> dict:
    int_minus_add = paired_values(
        stage, base_group, "int", stage, base_group, "iadd", SCREEN
    )
    int_minus_parent = paired_values(
        stage, base_group, "int", parent_stage, parent_group, "int", SCREEN
    )
    did = int_minus_add - paired_values(
        parent_stage, parent_group, "int", parent_stage, parent_group, "iadd", SCREEN
    )
    candidate_diag = [
        station_diagnostics(stage, base_group, "int", seed) for seed in SCREEN
    ]
    parent_diag = [
        station_diagnostics(parent_stage, parent_group, "int", seed) for seed in SCREEN
    ]
    rmse_delta = np.asarray(
        [a["rmse"] - b["rmse"] for a, b in zip(candidate_diag, parent_diag, strict=True)]
    )
    band_delta = {
        key: float(np.mean([a[key] - b[key] for a, b in zip(candidate_diag, parent_diag, strict=True)]))
        for key in ("h01", "h13", "h36")
    }
    eligible = bool(
        int_minus_add.mean() < 0.0
        and np.sum(int_minus_add < 0.0) >= 4
        and int_minus_parent.mean() < 0.0
        and np.sum(int_minus_parent < 0.0) >= 4
        and rmse_delta.mean() <= 0.0
        and max(band_delta.values()) <= 5e-4 + 1e-12
    )
    return {
        "group": base_group,
        "stage": stage,
        "eligible": eligible,
        "int_minus_iadd": int_minus_add.tolist(),
        "int_minus_parent_int": int_minus_parent.tolist(),
        "interaction_did": did.tolist(),
        "rmse_delta": rmse_delta.tolist(),
        "horizon_delta": band_delta,
        "int_station_mae_mean": float(
            np.mean([metric(stage, base_group, "int", seed)["station_mae_cf"] for seed in SCREEN])
        ),
    }


def choose(
    label: str,
    stage: str,
    groups: Sequence[str],
    *,
    parent_stage: str = "baseline",
    parent_group: str = "m0",
) -> tuple[str | None, dict]:
    records = [
        screen_record(
            stage, group, parent_stage=parent_stage, parent_group=parent_group
        )
        for group in groups
    ]
    eligible = [record for record in records if record["eligible"]]
    winner = (
        min(eligible, key=lambda record: record["int_station_mae_mean"])["group"]
        if eligible
        else None
    )
    payload = {
        "label": label,
        "screen_seeds": list(SCREEN),
        "parent": {"stage": parent_stage, "group": parent_group},
        "winner": winner,
        "records": records,
        "test_results_read": False,
    }
    atomic_json(ROOT / "selection" / f"{label}.json", payload)
    return winner, payload


def formal_jobs(stage: str, group: str, spec: Spec, seeds: Sequence[int]) -> list[Job]:
    return [
        Job(stage, group, role, seed, spec)
        for role in INTERACTIONS
        for seed in seeds
    ]


def code_snapshot() -> dict:
    paths = sorted(set(PROJECT.glob("src/**/*.py")) | set(PROJECT.glob("scripts/*.py")) |
                   set(PROJECT.glob("shell/*.sh")) | set(PROJECT.glob("tests/test_farm_interaction*.py")) |
                   {PROJECT / "configs/farm_interaction_enhancement_v1.yaml",
                    PROJECT / "exp_docs/farm_interaction_enhancement_v1_strengthening_plan.md"})
    hashes = {str(p.relative_to(PROJECT)): sha256(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def promote(source_stage: str, target_stage: str, group: str, spec: Spec) -> None:
    for role in INTERACTIONS:
        for seed in SCREEN:
            source = Job(source_stage, group, role, seed, spec)
            if not complete(source):
                raise RuntimeError(f"Incomplete promotion source: {source.key}")
            target = Job(target_stage, group, role, seed, spec).run_dir
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_link(source.run_dir, target)


def gate_winner(records: Sequence[dict]) -> str | None:
    eligible = [r for r in records if r["eligible"] and r.get("context") != "id_film"]
    return min(eligible, key=lambda r: (r["int_station_mae_mean"], r["group"]))["group"] if eligible else None


def persist_once(path: Path, payload: dict) -> dict:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise RuntimeError(f"Immutable registration changed: {path}")
        return previous
    atomic_json(path, payload)
    return payload


def choose_readout(label, stage, groups, parent_stage="baseline", parent_group="m0"):
    records = [screen_record(stage, group, parent_stage=parent_stage, parent_group=parent_group)
               for group in groups]
    structural = screen_record(stage, groups[1], parent_stage=stage, parent_group=groups[0])
    records[1]["capacity_control_eligible"] = structural["eligible"]
    records[1]["eligible"] = records[1]["eligible"] and structural["eligible"]
    winner = gate_winner(records)
    payload = {"winner": winner, "records": records, "level_shape_vs_capacity": structural,
               "screen_seeds": list(SCREEN), "test_results_read": False}
    atomic_json(ROOT / "selection" / f"{label}.json", payload)
    return winner, payload


def register_pair(catalog, stage, group, spec, seeds, parent_stage, parent_group, **extra):
    for existing in catalog.values():
        if existing["spec"] == asdict(spec):
            return existing["group"]
    catalog[group] = {"stage": stage, "group": group, "spec": asdict(spec), "seeds": list(seeds),
                      "parent_stage": parent_stage, "parent_group": parent_group, **extra}
    return group


def promote_catalog(catalog, group, target_stage, slots):
    pair = catalog[group]
    if pair["seeds"] == list(FORMAL):
        return
    spec = Spec(**pair["spec"])
    run_jobs(f"formal-{group}", formal_jobs(target_stage, group, spec, REMAINING), slots)
    promote(pair["stage"], target_stage, group, spec)
    pair.update(stage=target_stage, seeds=list(FORMAL))


def primary_selection(catalog):
    records = []
    baseline = catalog["m0"]
    # A strict interaction advantage is allowed to fail. In that case M0-IAdd
    # remains an honest primary fallback rather than forcing an Int winner.
    best = {**baseline, "role": "iadd",
            "score": float(np.mean([metric(baseline["stage"], "m0", "iadd", s)["station_mae_cf"] for s in FORMAL]))}
    for pair in catalog.values():
        if (pair["seeds"] != list(FORMAL) or pair["spec"]["context"] == "id_film"
                or pair["spec"]["output"] != "cf_residual" or pair.get("primary_eligible") is False):
            continue
        values = np.array([metric(pair["stage"], pair["group"], "int", s)["station_mae_cf"] for s in FORMAL])
        additive = np.array([metric(pair["stage"], pair["group"], "iadd", s)["station_mae_cf"] for s in FORMAL])
        eligible = bool((values-additive).mean() < 0 and (values < additive).sum() >= 8)
        parent = pair["parent_group"]
        guard_groups = [(pair["parent_stage"], parent)]
        if pair["spec"]["readout"] == "level_shape":
            control = catalog[parent]
            guard_groups.append((control["parent_stage"], control["parent_group"]))
        guard_results = []
        for parent_stage, parent_group in dict.fromkeys(guard_groups):
            if parent_group == pair["group"]:
                continue
            pvalues = np.array([metric(parent_stage, parent_group, "int", s)["station_mae_cf"] for s in FORMAL])
            candidate_diag = [station_diagnostics(pair["stage"], pair["group"], "int", s) for s in FORMAL]
            parent_diag = [station_diagnostics(parent_stage, parent_group, "int", s) for s in FORMAL]
            deltas = {k: float(np.mean([a[k]-b[k] for a,b in zip(candidate_diag,parent_diag)]))
                      for k in ("rmse","h01","h13","h36")}
            passed = bool((values-pvalues).mean() < 0 and (values < pvalues).sum() >= 8
                          and deltas["rmse"] <= 0 and max(deltas[k] for k in ("h01","h13","h36")) <= 5e-4+1e-12)
            guard_results.append({"parent":parent_group,"eligible":passed,"deltas":deltas})
            eligible = eligible and passed
        score = float(values.mean())
        records.append({"group": pair["group"], "eligible": eligible, "mean": score,
                        "int_minus_iadd": (values-additive).tolist(), "parent_guards":guard_results})
        if eligible and score < best["score"]:
            best = {**pair, "role": "int", "score": score}
    payload = {"selected": best, "records": records, "test_results_read": False}
    persist_once(ROOT / "selection/primary_validation_selection.json", payload)
    return best


def strengthen_finalist(catalog, best, slots):
    spec = Spec(**best["spec"])
    ablations = []
    for field, default in (("history","pooled_state"),("forcing","legacy"),("context","none"),("readout","legacy")):
        if getattr(spec, field) == default:
            continue
        dropped = replace(spec, **{field: default})
        group = register_pair(catalog, "component_ablation", f"final_without_{field}", dropped,
                              SCREEN, best["stage"], best["group"], primary_eligible=False)
        pair = catalog[group]
        run_jobs(f"ablation-{field}", formal_jobs(pair["stage"], group, dropped, SCREEN), slots)
        ablations.append({"removed": field, "group": group, "stage": pair["stage"],
                          "parent_group": best["group"], "parent_stage": best["stage"]})
    bounded = replace(spec, output="bounded_cf_mix")
    group = register_pair(catalog, "output_robustness", "bounded_finalist", bounded, FORMAL,
                          "baseline", "m0_bounded", primary_eligible=False)
    pair = catalog[group]
    run_jobs("bounded-finalist-confirmation", formal_jobs(pair["stage"], group, bounded, FORMAL), slots)
    return ablations, group


def finish_frozen(freeze: dict, slots) -> None:
    pairs = freeze["formal_pairs"] + freeze["screen_only_pairs"]
    tests = [Job(p["stage"], p["group"], role, seed, Spec(**p["spec"]), "test")
             for p in pairs for role in INTERACTIONS for seed in p["seeds"]]
    run_jobs("frozen-2024-test", tests, slots)
    by_group = {p["group"]: p for p in pairs}
    interventions = [Job(by_group[g]["stage"], g, role, seed, Spec(**by_group[g]["spec"]), mode)
                     for g in dict.fromkeys(("m0", freeze["selected_group"]))
                     for role in INTERACTIONS for seed in FORMAL
                     for mode in ("val_reference", "test_reference", "val_weather", "test_weather")]
    run_jobs("frozen-mechanism-interventions", interventions, slots)
    subprocess.run([sys.executable, str(PROJECT / "scripts/analyze_farm_interaction_enhancement_v1.py")],
                   cwd=PROJECT, check=True)
    (ROOT / "COMPLETE").touch()
    atomic_json(ROOT / "status.json", {"state":"COMPLETE", "active":0, "queued":0,
                "training_runs":len(tests), "test_runs":len(tests), "intervention_runs":len(interventions),
                "launcher_pid":os.getpid(), "updated_unix":time.time()})
    print(f"[farm-enhancement] complete: {ROOT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run farm interaction enhancements")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="8")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ROOT / "launcher.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A farm-enhancement launcher is already active") from error
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()
    gpus = tuple(int(v) for v in args.gpus.split(","))
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 or g > 7 for g in gpus):
        raise ValueError("Unique GPU indices 0-7 required")
    counts = parse_slot_counts(gpus, args.slots)
    slots = tuple(sorted(build_worker_slots(gpus, counts), key=lambda x: (x.slot, x.gpu)))
    snapshot = code_snapshot()
    persist_once(ROOT / "code_snapshot.json", snapshot)
    for name in snapshot["files"]:
        target = ROOT / "code_snapshot" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT / name, target)
    smoke = json.loads((ROOT / "cuda_smoke.json").read_text())
    if smoke["code_sha256"] != snapshot["sha256"] or not smoke["passed"]:
        raise RuntimeError("Successful CUDA smoke for current code is required")
    manifest_path = ROOT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "created_unix": int(time.time()), "screen_seeds": list(SCREEN), "formal_seeds": list(FORMAL),
        "candidates": {name: asdict(spec) for name, spec in BASE_CANDIDATES.items()},
        "primary_metric": "2023 station_mae_cf", "test_before_validation_freeze": False,
        "output_mode": "cf_residual", "m0_retrained": True,
        "training_runs_min": 130, "training_runs_max": 360,
        "protocol_version": 3, "physical_gpu_identification": "nvidia-smi index -> UUID",
        "note": "Adaptive total; one frozen test inference per unique training run; legacy baselines reported separately"}
    gpu_inventory = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.total,memory.free",
                                             "--format=csv,noheader,nounits"], text=True)
    manifest["gpu_inventory_at_start"] = gpu_inventory
    manifest["completed_results_review"] = "2023-only trajectory/ramp follow-up; mean-error share of MSE reduction 88.45%"
    manifest.update(gpus=list(gpus), slot_count_by_gpu=dict(zip(map(str, gpus), counts)),
                    launcher_pid=os.getpid(), last_started_unix=int(time.time()))
    atomic_json(manifest_path, manifest)
    freeze_path = ROOT / "selection/validation_freeze.json"
    if freeze_path.exists():
        finish_frozen(read_freeze(ROOT), slots)
        return
    # Selection always uses an M0 with the identical final-output link and schedule.
    run_jobs("m0-two-links-and-nine-pair-screen",
             formal_jobs("baseline", "m0", Spec(), SCREEN) +
             formal_jobs("baseline", "m0_bounded", Spec(output="bounded_cf_mix"), SCREEN) +
             [Job("screen", g, r, s, spec) for g, spec in BASE_CANDIDATES.items()
              for r in INTERACTIONS for s in SCREEN], slots)
    a, _ = choose("temporal_screen", "screen", ("a1_lead_summary", "a2_cross_attention"))
    b, _ = choose("forcing_screen", "screen", ("b1_origin_delta_linear", "b2_origin_delta_tcn"))
    c, _ = choose("site_screen", "screen", ("c1_metadata_residual", "c2_metadata_film"))
    diagnostic, _ = choose("id_diagnostic_screen", "screen", ("c3_id_film",))
    d, d_payload = choose_readout("readout_screen", "screen", ("d0_capacity_aux", "d1_level_shape"))
    winners = [g for g in (a, b, c, diagnostic, d) if g]
    if d == "d1_level_shape" and "d0_capacity_aux" not in winners:
        winners.append("d0_capacity_aux")
    # Each catalog entry is one method/seed identity; promotion replaces its view.
    catalog = {"m0": {"stage": "baseline", "group": "m0", "spec": asdict(Spec()), "seeds": list(FORMAL),
                       "parent_stage": "baseline", "parent_group": "m0"}}
    catalog["m0_bounded"] = {"stage":"baseline", "group":"m0_bounded", "spec":asdict(Spec(output="bounded_cf_mix")),
                             "seeds":list(FORMAL), "parent_stage":"baseline", "parent_group":"m0_bounded",
                             "primary_eligible":False}
    for g, spec in BASE_CANDIDATES.items():
        catalog[g] = {"stage": "screen", "group": g, "spec": asdict(spec), "seeds": list(SCREEN),
                      "parent_stage": "baseline", "parent_group": "m0"}
    catalog["d0_capacity_aux"]["primary_eligible"] = bool(d_payload["records"][0]["eligible"])
    catalog["d1_level_shape"].update(parent_stage="formal" if d == "d1_level_shape" else "screen", parent_group="d0_capacity_aux")
    run_jobs("m0-and-direction-formal",
             formal_jobs("baseline", "m0", Spec(), REMAINING) +
             formal_jobs("baseline", "m0_bounded", Spec(output="bounded_cf_mix"), REMAINING) +
             [job for g in winners for job in formal_jobs("formal", g, BASE_CANDIDATES[g], REMAINING)], slots)
    for g in winners:
        promote("screen", "formal", g, BASE_CANDIDATES[g])
        catalog[g].update(stage="formal", seeds=list(FORMAL))
    parent_group = "m0"
    parent_stage = "baseline"
    ab_spec = Spec()
    if a and b:
        ab_spec = Spec(history=BASE_CANDIDATES[a].history, forcing=BASE_CANDIDATES[b].forcing)
        ab = f"ab_{a}_{b}"
        # Compare the combination to the better eligible direction on screening.
        parent_group = min((a, b), key=lambda g: np.mean([metric("formal", g, "int", s)["station_mae_cf"] for s in SCREEN]))
        parent_stage = "formal"
        run_jobs("ab-screen", formal_jobs("combination_screen", ab, ab_spec, SCREEN), slots)
        ab_winner, _ = choose("ab_screen", "combination_screen", (ab,), parent_stage=parent_stage, parent_group=parent_group)
        catalog[ab] = {"stage": "combination_screen", "group": ab, "spec": asdict(ab_spec), "seeds": list(SCREEN),
                       "parent_stage": parent_stage, "parent_group": parent_group}
        if ab_winner:
            run_jobs("ab-formal", formal_jobs("combination", ab, ab_spec, REMAINING), slots)
            promote("combination_screen", "combination", ab, ab_spec)
            catalog[ab].update(stage="combination", seeds=list(FORMAL))
            parent_stage, parent_group = "combination", ab
        else:
            ab_spec = BASE_CANDIDATES[parent_group]
    elif a or b:
        parent_group, parent_stage = a or b, "formal"
        ab_spec = BASE_CANDIDATES[parent_group]
    # No A/B improvement means C has already been screened on M0: reuse exactly.
    if parent_group != "m0":
        contexts = {f"{parent_group}_{g}": Spec(ab_spec.history, ab_spec.forcing, spec.context)
                    for g, spec in BASE_CANDIDATES.items() if g.startswith("c")}
        run_jobs("context-on-ab-screen", [job for g, spec in contexts.items()
                 for job in formal_jobs("context_screen", g, spec, SCREEN)], slots)
        physical = tuple(g for g, spec in contexts.items() if spec.context != "id_film")
        cg, _ = choose("context_on_ab_screen", "context_screen", physical, parent_stage=parent_stage, parent_group=parent_group)
        id_groups = tuple(g for g, spec in contexts.items() if spec.context == "id_film")
        dg, _ = choose("id_on_ab_diagnostic", "context_screen", id_groups, parent_stage=parent_stage, parent_group=parent_group)
        for g, spec in contexts.items():
            catalog[g] = {"stage": "context_screen", "group": g, "spec": asdict(spec), "seeds": list(SCREEN),
                          "parent_stage": parent_stage, "parent_group": parent_group}
        for g in (cg, dg):
            if g:
                run_jobs(f"context-formal-{g}", formal_jobs("context_formal", g, contexts[g], REMAINING), slots)
                promote("context_screen", "context_formal", g, contexts[g])
                catalog[g].update(stage="context_formal", seeds=list(FORMAL))
        if cg:
            parent_stage, parent_group = catalog[cg]["stage"], cg

    if parent_group == "m0" and c:
        parent_stage, parent_group = catalog[c]["stage"], c

    # Test the level/shape idea again on the selected representation, with a
    # parameter-matched ordinary head as the structural control.
    if parent_group != "m0":
        parent_spec = Spec(**catalog[parent_group]["spec"])
        head_groups = []
        for mode in ("capacity_aux","level_shape"):
            group = register_pair(catalog, "head_screen", f"{parent_group}_head_{mode}",
                                  replace(parent_spec, readout=mode), SCREEN, parent_stage, parent_group)
            head_groups.append(group)
        run_jobs("head-on-combination-screen",
                 [job for g in head_groups for job in formal_jobs(catalog[g]["stage"], g, Spec(**catalog[g]["spec"]), SCREEN)], slots)
        head_winner, head_payload = choose_readout("head_on_combination", "head_screen", tuple(head_groups),
                                                  parent_stage, parent_group)
        catalog[head_groups[0]]["primary_eligible"] = bool(head_payload["records"][0]["eligible"])
        if head_winner:
            # A winning level/shape head carries its equal-parameter capacity
            # control into formal evaluation even when that control fails.
            carry = head_groups if head_winner == head_groups[1] else [head_winner]
            for g in carry:
                promote_catalog(catalog, g, "head_formal", slots)
        catalog[head_groups[1]].update(parent_stage=catalog[head_groups[0]]["stage"], parent_group=head_groups[0])
    best = primary_selection(catalog)
    ablations, bounded_group = strengthen_finalist(catalog, best, slots)
    formal = [p for p in catalog.values() if p["seeds"] == list(FORMAL)]
    from analyze_farm_interaction_enhancement_v1 import train_thresholds
    thresholds = {str(lag): {"station_cf": values[0], "cluster_mw": values[1]}
                  for lag, values in train_thresholds().items()}
    registered = []
    for p in catalog.values():
        for role in INTERACTIONS:
            for seed in p["seeds"]:
                job = Job(p["stage"], p["group"], role, seed, Spec(**p["spec"]))
                if not complete(job):
                    raise RuntimeError(f"Missing registered training result: {job.key}")
                checkpoint = job.run_dir / "best_model.pt"
                registered.append({"checkpoint": str(checkpoint.relative_to(ROOT)),
                                   "checkpoint_sha256": sha256(checkpoint),
                                   "config_sha256": sha256(job.run_dir / "config.yaml"),
                                   "validation_metrics_sha256": sha256(job.run_dir / "metrics_val.json")})
    freeze = {"protocol_version": 3, "selected_stage": best["stage"], "selected_group": best["group"],
              "selected_role": best["role"], "component_ablations": ablations, "bounded_finalist_group": bounded_group,
              "selected_spec": best["spec"], "formal_pairs": formal,
              "screen_only_pairs": [p for p in catalog.values() if p not in formal],
              "registered_runs": registered, "train_q90_ramp_thresholds": thresholds,
              "code_sha256": snapshot["sha256"], "test_results_read": False, "created_unix": int(time.time())}
    atomic_json(freeze_path, freeze)
    finish_frozen(freeze, slots)


if __name__ == "__main__":
    main()
