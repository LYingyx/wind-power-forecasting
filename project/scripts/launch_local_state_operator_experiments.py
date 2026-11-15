from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Iterable, Sequence

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
ROOT = PROJECT / "artifacts" / "local_state_operator_v1"
PRIOR = PROJECT / "artifacts" / "local_to_global_raw_bilinear_v1"
CACHE = PROJECT / "data" / "local_bilinear_56farm_v1"
SCREEN = (42, 46, 50)
FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL if seed not in SCREEN)
NEW_GROUPS = ("g_int", "l_int", "g_cat", "l_cat")
AGGREGATE_GROUPS = {"g_rmul", "g_int", "g_cat"}
LOCAL_GROUPS = {"l_rmul", "l_int", "l_cat"}
HORIZON_KEYS = (
    "horizon_0_1h_mae_mw",
    "horizon_1_3h_mae_mw",
    "horizon_3_6h_mae_mw",
)
CONTROL_PATHS = {
    "l1": ("formal", "l1_mft"),
    "g_rmul": ("formal", "l2_agg"),
    "l_rmul": ("formal", "l3_local"),
}


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    initial: str = ""
    epochs: int = 20
    split: str = "train"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"


def required_files(job: Job) -> tuple[str, ...]:
    if job.split == "train":
        return (
            "COMPLETE", "config.yaml", "best_model.pt", "metrics_val.json",
            "predictions_val.npz",
        )
    return ("TEST_COMPLETE", "best_model.pt", "metrics_test.json", "predictions_test.npz")


def complete(job: Job) -> bool:
    return all((job.run_dir / name).is_file() for name in required_files(job))


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[state-operator] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        environment = dict(os.environ)
        environment.update(
            WIND_LOCAL_GLOBAL_ROOT=str(ROOT),
            WIND_LOCAL_GLOBAL_EXPERIMENT_ID="local_state_operator_v1",
            WIND_LOCAL_CYCLES_PER_BATCH="12" if total_memory >= 40000 else "6",
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_local_to_global_job.sh"),
                str(slot.gpu), job.stage, job.group, str(job.seed),
                job.initial, str(job.epochs), job.split,
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        label,
        pending,
        slots,
        is_heavy=lambda _job: False,
        runner=runner,
        item_label=lambda job: f"{job.stage}/{job.group}/seed{job.seed}/{job.split}",
        oom_detector=lambda job, code: _looks_like_oom(job.run_dir / "run.log", code),
        unit="run",
    )


def control_run(group: str, seed: int) -> Path:
    stage, prior_group = CONTROL_PATHS[group]
    path = PRIOR / stage / prior_group / f"seed{seed}"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def run_path(stage: str, group: str, seed: int) -> Path:
    return control_run(group, seed) if group in CONTROL_PATHS else ROOT / stage / group / f"seed{seed}"


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict[str, Any]:
    path = run_path(stage, group, seed) / f"metrics_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def intervention_path(stage: str, group: str, name: str, seed: int, split: str) -> Path:
    if group == "l_rmul":
        return PRIOR / "interventions" / "l3_local" / name / f"seed{seed}" / f"metrics_{split}.json"
    return ROOT / "interventions" / stage / group / name / f"seed{seed}" / f"metrics_{split}.json"


def intervention_names(group: str) -> tuple[str, ...]:
    if group in AGGREGATE_GROUPS:
        return ("batch_roll", "reference")
    if group in LOCAL_GROUPS:
        return ("station_roll", "batch_roll", "reference")
    return ()


def evaluate_interventions(
    stage: str,
    groups: Sequence[str],
    seeds: Sequence[int],
    split: str,
    slots: Sequence[WorkerSlot],
) -> None:
    items = [
        (group, seed, name)
        for group in groups
        for seed in seeds
        for name in intervention_names(group)
        if not intervention_path(stage, group, name, seed, split).is_file()
    ]
    if not items:
        print(f"[state-operator] interventions-{stage}-{split} already complete", flush=True)
        return

    def runner(item: tuple[str, int, str], slot: WorkerSlot, _total: int) -> int:
        group, seed, name = item
        source = run_path(stage, group, seed)
        target = intervention_path(stage, group, name, seed, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        return subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "evaluate_forced_evolution.py"),
                "--config", str(source / "config.yaml"),
                "--checkpoint", str(source / "best_model.pt"),
                "--split", split,
                "--output", str(target),
                "--local-state-intervention", name,
                "--device", "cuda",
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        f"interventions-{stage}-{split}",
        items,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/seed{item[1]}/{item[2]}/{split}",
        unit="eval",
    )


def mean_difference(
    stage: str, candidate: str, reference: str, key: str, seeds: Sequence[int]
) -> tuple[float, list[float]]:
    values = [
        float(metric(stage, candidate, seed)[key]) - float(metric(stage, reference, seed)[key])
        for seed in seeds
    ]
    return statistics.mean(values), values


def intervention_summary(
    stage: str, group: str, name: str, seeds: Sequence[int], split: str = "val"
) -> dict[str, Any]:
    differences = []
    for seed in seeds:
        base = float(metric(stage, group, seed, split)["cluster_mae_mw"])
        changed = json.loads(
            intervention_path(stage, group, name, seed, split).read_text(encoding="utf-8")
        )
        differences.append(float(changed["cluster_mae_mw"]) - base)
    return {
        "mean_degradation_mw": statistics.mean(differences),
        "positive_seeds": sum(value > 0.0 for value in differences),
        "values_mw": differences,
    }


def screen_candidate(group: str) -> dict[str, Any]:
    reference = "g_rmul" if group.startswith("g_") else "l1"
    mae_delta, mae_values = mean_difference(
        "screen", group, reference, "cluster_mae_mw", SCREEN
    )
    rmse_delta, _ = mean_difference(
        "screen", group, reference, "cluster_rmse_mw", SCREEN
    )
    horizon_delta = {
        key: mean_difference("screen", group, reference, key, SCREEN)[0]
        for key in HORIZON_KEYS
    }
    interventions = {
        name: intervention_summary("screen", group, name, SCREEN)
        for name in intervention_names(group)
    }
    sample_state_used = any(
        interventions[name]["mean_degradation_mw"] >= 0.5
        and interventions[name]["positive_seeds"] >= 2
        for name in ("batch_roll", "reference")
    )
    local_correspondence = bool(
        group.startswith("l_")
        and interventions["station_roll"]["mean_degradation_mw"] >= 0.25
        and interventions["station_roll"]["positive_seeds"] >= 2
    )
    safety_passed = rmse_delta <= 3.0 and max(horizon_delta.values()) <= 5.0
    passed = (
        mae_delta <= -0.5
        and sum(value < 0.0 for value in mae_values) >= 2
        and sample_state_used
        and safety_passed
    )
    return {
        "group": group,
        "reference": reference,
        "mean_mae_delta_mw": mae_delta,
        "mae_wins": sum(value < 0.0 for value in mae_values),
        "mae_differences_mw": mae_values,
        "mean_rmse_delta_mw": rmse_delta,
        "horizon_delta_mw": horizon_delta,
        "interventions": interventions,
        "sample_state_used": sample_state_used,
        "local_correspondence": local_correspondence,
        "safety_passed": safety_passed,
        "promoted": passed,
    }


def code_snapshot() -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[str] = []
    for directory in ("src", "scripts", "shell", "configs", "tests"):
        for path in sorted((PROJECT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
                relative = str(path.relative_to(PROJECT))
                files.append(relative)
                digest.update(relative.encode())
                digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": files, "created_unix": int(time.time())}


def validate_prior_inputs() -> None:
    if not (CACHE / "COMPLETE").is_file():
        raise FileNotFoundError(f"Missing completed bilinear cache: {CACHE}")
    for group in CONTROL_PATHS:
        for seed in FORMAL:
            run = control_run(group, seed)
            for name in ("COMPLETE", "best_model.pt", "metrics_val.json", "metrics_test.json"):
                if not (run / name).is_file():
                    raise FileNotFoundError(run / name)
    for seed in FORMAL:
        stage = "screen_base" if seed in SCREEN else "formal_base"
        run = PRIOR / stage / "l1_m" / f"seed{seed}"
        if not (run / "best_model.pt").is_file():
            raise FileNotFoundError(run / "best_model.pt")


def base_checkpoint(seed: int) -> str:
    stage = "screen_base" if seed in SCREEN else "formal_base"
    return str(PRIOR / stage / "l1_m" / f"seed{seed}" / "best_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local state x interaction operator experiments")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slots = build_worker_slots(gpus, parse_slot_counts(gpus, args.slots, totals))
    slots = tuple(sorted(slots, key=lambda slot: (slot.slot, slot.gpu)))

    validate_prior_inputs()
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        prior_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if prior_snapshot["sha256"] != snapshot["sha256"]:
            raise RuntimeError("Scientific code changed after state-operator runs began")
    else:
        atomic_json(snapshot_path, snapshot)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "screen_seeds": list(SCREEN),
            "formal_seeds": list(FORMAL),
            "new_groups": list(NEW_GROUPS),
            "reused_controls": CONTROL_PATHS,
            "gpus": list(gpus),
            "slots": args.slots,
            "promotion": {
                "mae_delta_max_mw": -0.5,
                "mae_min_wins": 2,
                "intervention_degradation_min_mw": 0.5,
                "intervention_min_positive_seeds": 2,
                "rmse_delta_max_mw": 3.0,
                "horizon_delta_max_mw": 5.0,
                "local_correspondence_degradation_min_mw": 0.25,
            },
            "candidate_test_used_for_selection": False,
            "station_specific_models_launched": False,
        },
    )

    run_jobs(
        "operator-screen",
        [Job("screen", group, seed, base_checkpoint(seed)) for group in NEW_GROUPS for seed in SCREEN],
        slots,
    )
    evaluate_interventions(
        "screen", ("g_rmul", *NEW_GROUPS), SCREEN, "val", slots
    )
    screening = {group: screen_candidate(group) for group in NEW_GROUPS}
    promoted = tuple(group for group in NEW_GROUPS if screening[group]["promoted"])
    atomic_json(
        ROOT / "selection" / "screening.json",
        {
            "candidates": screening,
            "promoted": list(promoted),
            "station_identity_eligible": bool(screening["l_int"]["promoted"]),
            "test_seen": False,
        },
    )

    run_jobs(
        "operator-formal",
        [
            Job("formal", group, seed, base_checkpoint(seed))
            for group in promoted for seed in REMAINING
        ],
        slots,
    )
    for group in promoted:
        for seed in SCREEN:
            target = ROOT / "formal" / group / f"seed{seed}"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(ROOT / "screen" / group / f"seed{seed}", target_is_directory=True)
    evaluate_interventions(
        "formal", ("g_rmul", *promoted), FORMAL, "val", slots
    )

    formal_groups = ("l1", "g_rmul", "l_rmul", *promoted)
    formal_means = {
        group: statistics.mean(
            float(metric("formal", group, seed)["cluster_mae_mw"]) for seed in FORMAL
        )
        for group in formal_groups
    }
    zero_invariant_groups = tuple(
        group for group in formal_groups if group not in {"g_cat", "l_cat"}
    )
    selected = min(zero_invariant_groups, key=formal_means.__getitem__)
    cat_groups = tuple(group for group in promoted if group.endswith("_cat"))
    selected_cat = min(cat_groups, key=formal_means.__getitem__) if cat_groups else None
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {
            "formal_means": formal_means,
            "selected_zero_invariant": selected,
            "selected_cat_diagnostic": selected_cat,
            "seeds": list(FORMAL),
            "candidate_test_seen": False,
        },
    )

    run_jobs(
        "frozen-test",
        [Job("formal", group, seed, split="test") for group in promoted for seed in FORMAL],
        slots,
    )
    evaluate_interventions(
        "formal", ("g_rmul", *promoted), FORMAL, "test", slots
    )
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_local_state_operator.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[state-operator] complete selected={selected} promoted={promoted}", flush=True)


if __name__ == "__main__":
    main()
