from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable, Sequence

import numpy as np

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
ROOT = PROJECT / "artifacts" / "farm_level_history_weather_v1"
SEEDS = tuple(range(42, 52))
SCREEN = (42, 46, 50)
REMAINING = tuple(seed for seed in SEEDS if seed not in SCREEN)


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    history_steps: int
    train_indices: tuple[int, ...] | None = None
    metric_indices: tuple[int, ...] | None = None
    split: str = "train"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"


def complete(job: Job) -> bool:
    if job.split == "train":
        required = (
            job.run_dir / "COMPLETE",
            job.run_dir / "best_model.pt",
            job.run_dir / "metrics_val.json",
            job.run_dir / "predictions_val.npz",
        )
    else:
        required = (
            job.run_dir / "TEST_COMPLETE",
            job.run_dir / "metrics_test.json",
            job.run_dir / "predictions_test.npz",
        )
    return all(path.is_file() for path in required)


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[farm-level-v1] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, _total_memory: int) -> int:
        environment = dict(os.environ)
        environment.update(
            WIND_FARM_LEVEL_V1_ROOT=str(ROOT),
            WIND_FARM_LEVEL_V1_CYCLES_PER_BATCH="3",
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        train = "null" if job.train_indices is None else json.dumps(job.train_indices)
        metric = "null" if job.metric_indices is None else json.dumps(job.metric_indices)
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_farm_level_history_weather_v1_job.sh"),
                str(slot.gpu),
                job.stage,
                job.group,
                str(job.seed),
                str(job.history_steps),
                train,
                metric,
                job.split,
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


def link_reused_runs() -> None:
    sources = {
        "hw_none": PROJECT / "artifacts/local_state_spatial_interaction_v1/formal/bilinear_none",
        "hw_sadd": PROJECT / "artifacts/local_state_cluster_transfer_v2/station/b_shared_add",
        "hw_iadd": PROJECT / "artifacts/local_state_cluster_transfer_v2/station/b_independent_add",
        "hw_int": PROJECT / "artifacts/local_state_cluster_transfer_v2/station/b_int",
    }
    target_root = ROOT / "reused"
    target_root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = target_root / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected reused-run link: {target}")
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)


def build_folds() -> list[list[int]]:
    capacities = np.load(ROOT.parent.parent / "data/model_cache_pad5_f8/static/station_capacity_mw.npy")
    station_ids = np.load(ROOT.parent.parent / "data/model_cache_pad5_f8/static/station_id.npy")
    rng = np.random.default_rng(20260907)
    tie = rng.random(len(capacities))
    order = sorted(range(len(capacities)), key=lambda i: (-float(capacities[i]), tie[i], str(station_ids[i])))
    folds: list[list[int]] = [[] for _ in range(8)]
    totals = [0.0] * 8
    for station in order:
        candidates = [index for index, fold in enumerate(folds) if len(fold) < 7]
        chosen = min(candidates, key=lambda index: (totals[index], index))
        folds[chosen].append(station)
        totals[chosen] += float(capacities[station])
    for fold in folds:
        fold.sort()
    if sorted(value for fold in folds for value in fold) != list(range(56)):
        raise RuntimeError("Held-out folds are not a partition of the 56 farms")
    atomic_json(
        ROOT / "selection/fold_manifest.json",
        {
            "seed": 20260907,
            "inputs": ["station_id", "station_capacity_mw"],
            "algorithm": "capacity-descending greedy assignment with seeded tie order",
            "folds": folds,
            "capacity_mw": [[float(capacities[i]) for i in fold] for fold in folds],
            "station_id": [[str(station_ids[i]) for i in fold] for fold in folds],
        },
    )
    return folds


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict:
    return json.loads(
        (ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json").read_text()
    )


def freeze_history_selection() -> int:
    path = ROOT / "selection/history_validation_freeze.json"
    candidates = {4: "h_strong_4", 12: "h_matched", 24: "h_strong_24"}
    scores = {
        str(steps): float(np.mean([metric("baseline", group, seed)["station_mae_cf"] for seed in SCREEN]))
        for steps, group in candidates.items()
    }
    winner = int(min(scores, key=scores.get))
    payload = {
        "selected_history_visible_steps": winner,
        "screen_seeds": list(SCREEN),
        "validation_metric": "station_mae_cf",
        "scores": scores,
        "test_results_read": False,
        "created_unix": int(time.time()),
    }
    if path.is_file():
        prior = json.loads(path.read_text())
        if prior["selected_history_visible_steps"] != winner:
            raise RuntimeError("Frozen history selection changed")
    else:
        atomic_json(path, payload)
    return winner


def freeze_heldout_protocol(folds: list[list[int]]) -> None:
    path = ROOT / "selection/heldout_validation_freeze.json"
    payload = {
        "folds": folds,
        "models": ["independent_output_additive", "faithful_concat"],
        "seeds": list(SCREEN),
        "checkpoint_selection": "training-farm 2023 station_mae_cf only",
        "heldout_targets_used_for_gradient_or_selection": False,
        "test_results_read": False,
        "created_unix": int(time.time()),
    }
    if path.is_file():
        prior = json.loads(path.read_text())
        if prior["folds"] != folds:
            raise RuntimeError("Frozen held-out protocol changed")
    else:
        atomic_json(path, payload)


def code_snapshot() -> dict:
    paths = (
        PROJECT / "configs/farm_level_history_weather_v1.yaml",
        PROJECT / "shell/run_farm_level_history_weather_v1_job.sh",
        PROJECT / "scripts/launch_farm_level_history_weather_v1.py",
        PROJECT / "scripts/analyze_farm_level_history_weather_v1.py",
        PROJECT / "src/wind_forcing/models/local_state_identifiability.py",
        PROJECT / "src/wind_forcing/training.py",
        PROJECT / "scripts/train_forced_evolution.py",
        PROJECT / "scripts/evaluate_forced_evolution.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT)).encode())
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": [str(p.relative_to(PROJECT)) for p in paths]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete farm-level source-role study")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="8")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ROOT / "launcher.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A farm-level launcher is already running") from error
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()

    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    counts = parse_slot_counts(gpus, args.slots, totals)
    slots = tuple(sorted(build_worker_slots(gpus, counts), key=lambda slot: (slot.slot, slot.gpu)))
    link_reused_runs()
    folds = build_folds()
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file() and json.loads(snapshot_path.read_text())["sha256"] != snapshot["sha256"]:
        raise RuntimeError("Scientific code changed after the experiment began")
    if not snapshot_path.is_file():
        atomic_json(snapshot_path, snapshot)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "gpus": list(gpus),
            "slots_per_gpu": list(counts),
            "formal_seeds": list(SEEDS),
            "heldout_seeds": list(SCREEN),
            "data_order": "2023 validation frozen before new 2024 evaluation",
            "primary_contrast": "HW-Int minus equal-parameter HW-IAdd",
            "launcher_pid": os.getpid(),
        },
    )
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts/analyze_farm_level_history_weather_v1.py"),
            "--freeze-protocol",
        ],
        cwd=PROJECT,
        check=True,
    )

    run_jobs(
        "source-role-baselines",
        [Job("baseline", "h_matched", seed, 12) for seed in SEEDS]
        + [Job("baseline", "w_anchor", seed, 12) for seed in SEEDS]
        + [Job("baseline", "h_strong_4", seed, 4) for seed in SCREEN]
        + [Job("baseline", "h_strong_24", seed, 24) for seed in SCREEN],
        slots,
    )
    winner = freeze_history_selection()
    winner_group = {4: "h_strong_4", 12: "h_matched", 24: "h_strong_24"}[winner]
    if winner != 12:
        run_jobs(
            "strong-history-formal",
            [Job("baseline", winner_group, seed, winner) for seed in REMAINING],
            slots,
        )
    test_groups = [("h_matched", 12), ("w_anchor", 12)]
    if winner_group != "h_matched":
        test_groups.append((winner_group, winner))
    run_jobs(
        "baseline-frozen-test",
        [
            Job("baseline", group, seed, steps, split="test")
            for group, steps in test_groups
            for seed in SEEDS
        ],
        slots,
    )

    heldout_jobs: list[Job] = []
    for fold_index, heldout in enumerate(folds):
        train = tuple(index for index in range(56) if index not in heldout)
        for suffix in ("iadd", "int"):
            for seed in SCREEN:
                heldout_jobs.append(
                    Job(
                        "heldout", f"fold{fold_index}_{suffix}", seed, 12,
                        train_indices=train, metric_indices=train,
                    )
                )
    run_jobs("heldout-farm-training", heldout_jobs, slots)
    freeze_heldout_protocol(folds)
    run_jobs(
        "heldout-farm-frozen-test",
        [
            Job(
                job.stage, job.group, job.seed, job.history_steps,
                train_indices=job.train_indices,
                metric_indices=tuple(folds[int(job.group.split("_")[0][4:])]),
                split="test",
            )
            for job in heldout_jobs
        ],
        slots,
    )
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts/analyze_farm_level_history_weather_v1.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[farm-level-v1] complete: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
