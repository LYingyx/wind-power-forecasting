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

import numpy as np

from launch_aggregate_cluster_experiments import (
    HIGH_MEMORY_MIB,
    WorkerSlot,
    _looks_like_oom,
    atomic_json,
    build_worker_slots,
    execute_parallel,
    gpu_state,
    parse_slot_counts,
)


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
R2_ROOT = PROJECT / "artifacts" / "aggregate_cluster_r2_state_v1" / "mechanism" / "r2_w"
SCREEN_SEEDS = (42, 46, 50)
FORMAL_SEEDS = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL_SEEDS if seed not in SCREEN_SEEDS)
CANDIDATES = ("b1", "b2", "c1", "c2", "d1", "d2", "e1", "f1")


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    training_stage: str
    initial_checkpoint: str = ""

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"

    @property
    def base_checkpoint(self) -> Path:
        return R2_ROOT / f"seed{self.seed}" / "best_model.pt"


def code_snapshot() -> dict[str, Any]:
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


def audited(run_dir: Path, split: str = "val") -> bool:
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


def read_metric(run_dir: Path, key: str = "cluster_mae_mw", split: str = "val") -> float:
    with (run_dir / f"metrics_{split}.json").open(encoding="utf-8") as stream:
        return float(json.load(stream)[key])


def base_metric(seed: int, key: str = "cluster_mae_mw", split: str = "val") -> float:
    return read_metric(R2_ROOT / f"seed{seed}", key, split)


def run_metric(stage: str, group: str, seed: int, key: str = "cluster_mae_mw") -> float:
    return read_metric(ROOT / stage / group / f"seed{seed}", key)


def combined_metric(group: str, seed: int, key: str = "cluster_mae_mw") -> float:
    stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
    return run_metric(stage, group, seed, key)


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def run_jobs(label: str, jobs: Sequence[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not audited(job.run_dir)]
    if not pending:
        print(f"[strict-scheduler] stage={label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        cycles, accumulation = ((12, 1) if total_memory >= HIGH_MEMORY_MIB else (6, 2))
        environment = dict(os.environ)
        environment.update(
            WIND_STRICT_ROOT=str(ROOT),
            WIND_AGG_CYCLES_PER_BATCH=str(cycles),
            WIND_AGG_GRAD_ACCUMULATION=str(accumulation),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        command = [
            str(PROJECT / "shell" / "run_aggregate_strict_modulation_job.sh"),
            str(slot.gpu), job.stage, job.group, str(job.seed),
            str(job.base_checkpoint), job.training_stage,
        ]
        if job.initial_checkpoint:
            command.append(job.initial_checkpoint)
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        label,
        pending,
        slots,
        is_heavy=lambda _job: False,
        runner=runner,
        item_label=lambda job: f"{job.stage}/{job.group}/seed{job.seed}",
        oom_detector=lambda job, returncode: _looks_like_oom(job.run_dir / "run.log", returncode),
        unit="train",
    )


def run_tests(label: str, items: Sequence[tuple[str, str, int]], slots: Sequence[WorkerSlot]) -> None:
    pending = [item for item in items if not audited(ROOT / item[0] / item[1] / f"seed{item[2]}", "test")]
    if not pending:
        return

    def runner(item: tuple[str, str, int], slot: WorkerSlot, _total: int) -> int:
        stage, group, seed = item
        run_dir = ROOT / stage / group / f"seed{seed}"
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
        label,
        pending,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/{item[1]}/seed{item[2]}/test",
        unit="eval",
    )


def intervention_valid(group: str, intervention: str, split: str, seed: int) -> bool:
    directory = ROOT / "interventions" / group / intervention / f"seed{seed}"
    marker = directory / f"{split.upper()}_COMPLETE"
    metrics = directory / f"metrics_{split}.json"
    predictions = directory / f"predictions_{split}.npz"
    if not all(path.is_file() for path in (marker, metrics, predictions)):
        return False
    try:
        with metrics.open(encoding="utf-8") as stream:
            values = json.load(stream)
        expected = 32550 if split == "val" else 28782
        with np.load(predictions, allow_pickle=False) as archive:
            return (
                values.get("correction_state_intervention") == intervention
                and int(values["samples"]) == expected
                and archive["cluster_prediction_mw"].shape == (expected, 24)
                and np.isfinite(archive["cluster_prediction_mw"]).all()
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_interventions(group: str, slots: Sequence[WorkerSlot]) -> None:
    items = [
        (intervention, split, seed)
        for intervention in ("global_permutation", "reference")
        for split in ("val", "test")
        for seed in FORMAL_SEEDS
        if not intervention_valid(group, intervention, split, seed)
    ]
    if not items:
        return

    def runner(item: tuple[str, str, int], slot: WorkerSlot, _total: int) -> int:
        intervention, split, seed = item
        stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
        run_dir = ROOT / stage / group / f"seed{seed}"
        output = ROOT / "interventions" / group / intervention / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        marker = output / f"{split.upper()}_COMPLETE"
        marker.unlink(missing_ok=True)
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot.gpu),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{intervention}:{split}:s{seed}]",
        )
        log_path = output / f"{split}.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                    str(PROJECT / "scripts" / "evaluate_aggregate_cluster.py"),
                    "--config", str(run_dir / "config.yaml"),
                    "--checkpoint", str(run_dir / "best_model.pt"),
                    "--split", split,
                    "--output", str(output / f"metrics_{split}.json"),
                    "--predictions-output", str(output / f"predictions_{split}.npz"),
                    "--device", "cuda",
                    "--correction-state-intervention", intervention,
                ],
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=log,
            )
        if result.returncode == 0:
            marker.touch()
        return result.returncode if result.returncode else (0 if intervention_valid(group, intervention, split, seed) else 4)

    execute_parallel(
        "strict-state-interventions",
        items,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{group}/{item[0]}/{item[1]}/seed{item[2]}",
        unit="eval",
    )


def run_invariance(group: str, slots: Sequence[WorkerSlot]) -> None:
    items = [
        seed for seed in FORMAL_SEEDS
        if not (ROOT / "diagnostics" / group / f"seed{seed}" / "zero_weather_invariance.json").is_file()
    ]
    if not items:
        return

    def runner(seed: int, slot: WorkerSlot, _total: int) -> int:
        stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
        run_dir = ROOT / stage / group / f"seed{seed}"
        output = ROOT / "diagnostics" / group / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot.gpu),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:invariance:{group}:s{seed}]",
        )
        return subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "diagnose_aggregate_strict_invariance.py"),
                "--config", str(run_dir / "config.yaml"),
                "--checkpoint", str(run_dir / "best_model.pt"),
                "--output", str(output / "zero_weather_invariance.json"),
                "--device", "cuda",
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "strict-zero-weather-invariance",
        items,
        slots,
        is_heavy=lambda _seed: False,
        runner=runner,
        item_label=lambda seed: f"{group}/zero-weather/seed{seed}",
        unit="audit",
    )


def initial(stage: str, group: str, seed: int) -> str:
    path = ROOT / stage / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def select_screen_candidates() -> list[str]:
    rows = []
    for group in CANDIDATES:
        values = [run_metric("screen_adapter", group, seed) for seed in SCREEN_SEEDS]
        rows.append({"group": group, "mean_mae_mw": mean(values), "values_mw": values})
    rows.sort(key=lambda row: row["mean_mae_mw"])
    promoted = [str(row["group"]) for row in rows[:3]]
    atomic_json(
        ROOT / "selection" / "stage1_adapter_screen.json",
        {
            "selection_split": "2023_validation",
            "rule": "promote the three lowest 3-seed mean-MAE strict candidates",
            "base_r2_w_mean_mae_mw": mean(base_metric(seed) for seed in SCREEN_SEEDS),
            "ranking": rows,
            "promoted": promoted,
        },
    )
    return promoted


def select_formal_candidates(promoted: Sequence[str]) -> list[str]:
    rows = []
    for group in promoted:
        values = [run_metric("screen_joint", group, seed) for seed in SCREEN_SEEDS]
        rows.append({"group": group, "mean_mae_mw": mean(values), "values_mw": values})
    rows.sort(key=lambda row: row["mean_mae_mw"])
    base = mean(base_metric(seed) for seed in SCREEN_SEEDS)
    control = mean(run_metric("screen_joint", "a0_ft", seed) for seed in SCREEN_SEEDS)
    selected = [str(rows[0]["group"])]
    if len(rows) > 1 and (
        float(rows[1]["mean_mae_mw"]) < min(base, control)
        or float(rows[1]["mean_mae_mw"]) <= float(rows[0]["mean_mae_mw"]) + 1.0
    ):
        selected.append(str(rows[1]["group"]))
    atomic_json(
        ROOT / "selection" / "stage2_joint_screen.json",
        {
            "selection_split": "2023_validation",
            "rule": (
                "always formalize the best strict candidate; retain a second only if it beats "
                "both R2-W and A0-FT by mean, or is within 1 MW of the best"
            ),
            "base_r2_w_mean_mae_mw": base,
            "a0_ft_mean_mae_mw": control,
            "ranking": rows,
            "formal_candidates": selected,
        },
    )
    return selected


def final_selection(formal: Sequence[str]) -> tuple[str, str]:
    rows = [
        {
            "group": group,
            "mean_mae_mw": mean(combined_metric(group, seed) for seed in FORMAL_SEEDS),
            "values_mw": [combined_metric(group, seed) for seed in FORMAL_SEEDS],
        }
        for group in formal
    ]
    rows.sort(key=lambda row: row["mean_mae_mw"])
    strict_best = str(rows[0]["group"])
    r2_mean = mean(base_metric(seed) for seed in FORMAL_SEEDS)
    selected = strict_best if float(rows[0]["mean_mae_mw"]) < r2_mean else "r2_w"
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {
            "created_unix": int(time.time()),
            "selection_split": "2023_validation",
            "rule": "lower 10-checkpoint mean MAE wins; exact tie retains R2-W",
            "r2_w_mean_mae_mw": r2_mean,
            "formal_strict_ranking": rows,
            "best_strict_candidate": strict_best,
            "selected_final_model": selected,
            "test_results_seen": False,
        },
    )
    return selected, strict_best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict state-conditioned R2 weather modulation")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must list distinct physical GPU indices")
    missing_base = [seed for seed in FORMAL_SEEDS if not audited(R2_ROOT / f"seed{seed}")]
    if missing_base:
        raise RuntimeError(f"Missing audited same-seed R2-W checkpoints: {missing_base}")
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slots = build_worker_slots(gpus, parse_slot_counts(gpus, args.slots, totals))
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        previous = json.loads(snapshot_path.read_text())
        if previous["sha256"] != snapshot["sha256"] and any(ROOT.glob("**/COMPLETE")):
            raise RuntimeError("Scientific code changed after strict-modulation runs began")
    atomic_json(snapshot_path, snapshot)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "objective": "strict state-conditioned weather-response mapping",
            "candidates": list(CANDIDATES),
            "screen_seeds": list(SCREEN_SEEDS),
            "formal_seeds": list(FORMAL_SEEDS),
            "base_r2_w_root": str(R2_ROOT),
            "gpus": list(gpus),
            "slots": [slot.label for slot in slots],
            "code_sha256": snapshot["sha256"],
        },
    )
    print(f"[strict] gpus={list(gpus)} workers={len(slots)}", flush=True)

    run_jobs(
        "stage1-adapter-screen",
        [Job("screen_adapter", group, seed, "adapter") for group in CANDIDATES for seed in SCREEN_SEEDS],
        slots,
    )
    promoted = select_screen_candidates()
    joint_jobs = [
        Job(
            "screen_joint", group, seed, "joint",
            initial("screen_adapter", group, seed),
        )
        for group in promoted for seed in SCREEN_SEEDS
    ]
    joint_jobs += [
        Job(
            "screen_joint", f"{group}_const", seed, "joint",
            initial("screen_adapter", group, seed),
        )
        for group in promoted for seed in SCREEN_SEEDS
    ]
    joint_jobs += [Job("screen_joint", "a0_ft", seed, "joint") for seed in SCREEN_SEEDS]
    run_jobs("stage2-joint-screen-controls", joint_jobs, slots)
    formal = select_formal_candidates(promoted)

    run_jobs(
        "stage3-formal-adapters",
        [Job("formal_adapter", group, seed, "adapter") for group in formal for seed in REMAINING],
        slots,
    )
    run_jobs(
        "stage3-formal-joint",
        [
            Job(
                "formal_joint", group, seed, "joint",
                initial("formal_adapter", group, seed),
            )
            for group in formal for seed in REMAINING
        ]
        + [Job("formal_joint", "a0_ft", seed, "joint") for seed in REMAINING],
        slots,
    )
    selected, strict_best = final_selection(formal)

    run_jobs(
        "stage3-formal-constant-control",
        [
            Job(
                "formal_joint", f"{strict_best}_const", seed, "joint",
                initial("formal_adapter", strict_best, seed),
            )
            for seed in REMAINING
        ],
        slots,
    )
    test_items: list[tuple[str, str, int]] = []
    for group in (*formal, "a0_ft", f"{strict_best}_const"):
        for seed in FORMAL_SEEDS:
            stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
            test_items.append((stage, group, seed))
    atomic_json(
        ROOT / "selection" / "test_evaluation_manifest.json",
        {
            "created_unix": int(time.time()),
            "validation_selected": selected,
            "best_strict_candidate": strict_best,
            "evaluated_groups": list(dict.fromkeys(item[1] for item in test_items)),
            "seeds": list(FORMAL_SEEDS),
        },
    )
    run_tests("stage4-frozen-test", test_items, slots)
    run_interventions(strict_best, slots)
    run_invariance(strict_best, slots)
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_aggregate_strict_modulation.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[strict] complete selected={selected} root={ROOT}", flush=True)


if __name__ == "__main__":
    main()
