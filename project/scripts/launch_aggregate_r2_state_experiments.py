from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from launch_aggregate_cluster_experiments import (
    HIGH_MEMORY_MIB,
    WorkerSlot,
    atomic_json,
    build_worker_slots,
    execute_parallel,
    gpu_state,
    parse_slot_counts,
)


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_cluster_r2_state_v1"
SOURCE_ROOT = PROJECT / "artifacts" / "aggregate_cluster_v1"
SEEDS = tuple(range(42, 52))
GROUPS = ("r2_w", "r2_add")


@dataclass(frozen=True)
class TrainingJob:
    group: str
    seed: int

    @property
    def run_dir(self) -> Path:
        return ROOT / "mechanism" / self.group / f"seed{self.seed}"

    @property
    def history_checkpoint(self) -> Path:
        return SOURCE_ROOT / "history" / "hb_tcn_3h" / f"seed{self.seed}" / "best_model.pt"


def _code_snapshot() -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[str] = []
    for directory in ("src", "scripts", "shell", "configs", "tests"):
        for path in sorted((PROJECT / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh", ".yaml"}:
                continue
            relative = str(path.relative_to(PROJECT))
            files.append(relative)
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": files, "created_unix": int(time.time())}


def _audited(run_dir: Path, split: str) -> bool:
    marker = run_dir / ("COMPLETE" if split == "val" else "TEST_COMPLETE")
    if not marker.is_file():
        return False
    return subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "audit_aggregate_run.py"),
            "--run-dir", str(run_dir), "--split", split,
        ],
        cwd=PROJECT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _intervention_valid(split: str, seed: int) -> bool:
    directory = ROOT / "interventions" / "r2_full_state_shift" / f"seed{seed}"
    marker = directory / f"{split.upper()}_COMPLETE"
    metrics_path = directory / f"metrics_{split}.json"
    predictions_path = directory / f"predictions_{split}.npz"
    if not all(path.is_file() for path in (marker, metrics_path, predictions_path)):
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
        expected = 32550 if split == "val" else 28782
        with np.load(predictions_path, allow_pickle=False) as archive:
            return (
                metrics.get("correction_state_intervention") == "cyclic_shift"
                and int(metrics["samples"]) == expected
                and archive["cluster_prediction_mw"].shape == (expected, 24)
                and np.isfinite(archive["cluster_prediction_mw"]).all()
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def run_training(slots: tuple[WorkerSlot, ...]) -> None:
    pending = [
        TrainingJob(group, seed)
        for group in GROUPS
        for seed in SEEDS
        if not _audited(ROOT / "mechanism" / group / f"seed{seed}", "val")
    ]

    def runner(job: TrainingJob, slot: WorkerSlot, total_memory: int) -> int:
        cycles, accumulation = ((12, 1) if total_memory >= HIGH_MEMORY_MIB else (6, 2))
        environment = dict(os.environ)
        environment.update(
            WIND_AGG_ROOT=str(ROOT),
            WIND_AGG_EXPERIMENT_NAMESPACE="aggregate_cluster_r2_state_v1",
            WIND_AGG_HISTORY_VISIBLE="12",
            WIND_AGG_STATE_MODE="current",
            WIND_AGG_CYCLES_PER_BATCH=str(cycles),
            WIND_AGG_GRAD_ACCUMULATION=str(accumulation),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.group}:s{job.seed}]",
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_aggregate_cluster_job.sh"),
                str(slot.gpu), "mechanism", job.group, str(job.seed),
                str(job.history_checkpoint),
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "r2-state-training",
        pending,
        slots,
        is_heavy=lambda _job: False,
        runner=runner,
        item_label=lambda job: f"mechanism/{job.group}/seed{job.seed}",
        unit="train",
    )


def run_test(slots: tuple[WorkerSlot, ...]) -> None:
    pending = [
        (group, seed)
        for group in GROUPS
        for seed in SEEDS
        if not _audited(ROOT / "mechanism" / group / f"seed{seed}", "test")
    ]

    def runner(item: tuple[str, int], slot: WorkerSlot, _total_memory: int) -> int:
        group, seed = item
        run_dir = ROOT / "mechanism" / group / f"seed{seed}"
        environment = dict(os.environ)
        environment.update(
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:test:{group}:s{seed}]",
        )
        return subprocess.run(
            [str(PROJECT / "shell" / "run_aggregate_cluster_test.sh"), str(slot.gpu), str(run_dir)],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "r2-state-frozen-test",
        pending,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"mechanism/{item[0]}/seed{item[1]}/test",
        unit="eval",
    )


def run_intervention(slots: tuple[WorkerSlot, ...]) -> None:
    pending = [
        (split, seed)
        for split in ("val", "test")
        for seed in SEEDS
        if not _intervention_valid(split, seed)
    ]

    def runner(item: tuple[str, int], slot: WorkerSlot, _total_memory: int) -> int:
        split, seed = item
        source = SOURCE_ROOT / "weather" / "r2_latent_interpolation" / f"seed{seed}"
        output = ROOT / "interventions" / "r2_full_state_shift" / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        metrics = output / f"metrics_{split}.json"
        predictions = output / f"predictions_{split}.npz"
        marker = output / f"{split.upper()}_COMPLETE"
        marker.unlink(missing_ok=True)
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot.gpu),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:state-shift:{split}:s{seed}]",
        )
        command = [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "evaluate_aggregate_cluster.py"),
            "--config", str(source / "config.yaml"),
            "--checkpoint", str(source / "best_model.pt"),
            "--split", split,
            "--output", str(metrics),
            "--predictions-output", str(predictions),
            "--device", "cuda",
            "--correction-state-intervention", "cyclic_shift",
        ]
        log_path = output / f"{split}.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=log, stderr=log)
        if result.returncode == 0 and _intervention_valid_without_marker(split, seed):
            marker.touch()
        return result.returncode if result.returncode else (0 if marker.is_file() else 4)

    execute_parallel(
        "r2-full-state-shift",
        pending,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"r2-full/seed{item[1]}/{item[0]}-state-shift",
        unit="eval",
    )


def _intervention_valid_without_marker(split: str, seed: int) -> bool:
    directory = ROOT / "interventions" / "r2_full_state_shift" / f"seed{seed}"
    marker = directory / f"{split.upper()}_COMPLETE"
    existed = marker.is_file()
    if not existed:
        marker.touch()
    valid = _intervention_valid(split, seed)
    if not existed:
        marker.unlink(missing_ok=True)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R2 correction-state control matrix")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must list distinct physical GPU indices")
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slot_counts = parse_slot_counts(gpus, args.slots, totals)
    slots = build_worker_slots(gpus, slot_counts)
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = _code_snapshot()
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "objective": "R2 correction-state controls",
            "groups": list(GROUPS),
            "seeds": list(SEEDS),
            "source_full_runs": str(SOURCE_ROOT / "weather" / "r2_latent_interpolation"),
            "history_checkpoints": str(SOURCE_ROOT / "history" / "hb_tcn_3h"),
            "gpus": list(gpus),
            "slot_counts": list(slot_counts),
            "code_sha256": snapshot["sha256"],
        },
    )
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        previous = json.loads(snapshot_path.read_text())
        if previous["sha256"] != snapshot["sha256"] and any(ROOT.glob("**/COMPLETE")):
            raise RuntimeError("Scientific code changed after R2-state formal runs began")
    atomic_json(snapshot_path, snapshot)
    print(
        f"[r2-state] gpus={list(gpus)} slots={list(slot_counts)} workers={len(slots)}",
        flush=True,
    )
    run_training(slots)
    atomic_json(
        ROOT / "validation_freeze.json",
        {
            "created_unix": int(time.time()),
            "selection_rule": "lowest 10-seed validation mean MAE; no significance threshold",
            "candidate_groups": ["r2_full", *GROUPS],
        },
    )
    run_test(slots)
    run_intervention(slots)
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_aggregate_r2_state.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[r2-state] complete root={ROOT}", flush=True)


if __name__ == "__main__":
    main()
