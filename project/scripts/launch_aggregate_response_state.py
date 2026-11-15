from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from launch_aggregate_strict_modulation import code_snapshot


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_response_state_v1"
R2_ROOT = PROJECT / "artifacts" / "aggregate_cluster_r2_state_v1" / "mechanism" / "r2_w"
STRICT_ROOT = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
SCREEN_SEEDS = (42, 46, 50)
FORMAL_SEEDS = tuple(range(42, 52))
REMAINING_SEEDS = tuple(seed for seed in FORMAL_SEEDS if seed not in SCREEN_SEEDS)
NEW_GROUPS = ("c2", "c3", "r1", "r2", "r3")
PAIRED_GROUPS = ("r1", "r2", "r3")
SINGLE_SOURCE_GROUPS = ("c2", "c3")
REFERENCE_GROUPS = ("c0_a0_ft", "c1_d2")


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


def response_run_dir(group: str, seed: int) -> Path:
    stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
    return ROOT / stage / group / f"seed{seed}"


def strict_reference_dir(group: str, seed: int) -> Path:
    strict_group = {"c0_a0_ft": "a0_ft", "c1_d2": "d2"}[group]
    stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
    return STRICT_ROOT / stage / strict_group / f"seed{seed}"


def run_dir(group: str, seed: int) -> Path:
    return strict_reference_dir(group, seed) if group in REFERENCE_GROUPS else response_run_dir(group, seed)


def read_metric(group: str, seed: int, split: str = "val", key: str = "cluster_mae_mw") -> float:
    with (run_dir(group, seed) / f"metrics_{split}.json").open(encoding="utf-8") as stream:
        return float(json.load(stream)[key])


def mean_metric(group: str, seeds: Sequence[int], split: str = "val") -> float:
    return statistics.mean(read_metric(group, seed, split) for seed in seeds)


def cache_response_latents(
    splits: Sequence[str], gpus: Sequence[int], slots: Sequence[WorkerSlot]
) -> None:
    items = [(seed, split) for seed in FORMAL_SEEDS for split in splits]
    cache_slots = tuple(slot for slot in slots if slot.slot == 0)
    if not cache_slots:
        cache_slots = tuple(
            WorkerSlot(gpu=gpu, slot=0, position=index + 1) for index, gpu in enumerate(gpus)
        )

    def runner(item: tuple[int, str], slot: WorkerSlot, _total: int) -> int:
        seed, split = item
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot.gpu),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:latent:s{seed}:{split}]",
        )
        return subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "cache_aggregate_response_latents.py"),
                "--aggregate-cache", str(PROJECT / "data" / "aggregate_cluster_v1"),
                "--response-cache", str(PROJECT / "data" / "aggregate_response_state_v1"),
                "--checkpoint", str(R2_ROOT / f"seed{seed}" / "best_model.pt"),
                "--split", split,
                "--output-root", str(ROOT / "latent_cache"),
                "--device", "cuda",
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        f"response-latent-{'-'.join(splits)}",
        items,
        cache_slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"latent/seed{item[0]}/{item[1]}",
        unit="cache",
    )


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not audited(job.run_dir)]
    if not pending:
        print(f"[response-scheduler] stage={label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        cycles, accumulation = ((6, 2) if total_memory >= HIGH_MEMORY_MIB else (3, 4))
        environment = dict(os.environ)
        environment.update(
            WIND_RESPONSE_ROOT=str(ROOT),
            WIND_AGG_CYCLES_PER_BATCH=str(cycles),
            WIND_AGG_GRAD_ACCUMULATION=str(accumulation),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        command = [
            str(PROJECT / "shell" / "run_aggregate_response_state_job.sh"),
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


def initial(stage: str, group: str, seed: int) -> str:
    path = ROOT / stage / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def select_screen_candidates() -> list[str]:
    rows = [
        {
            "group": group,
            "mean_mae_mw": mean_metric(group, SCREEN_SEEDS),
            "seed_mae_mw": [read_metric(group, seed) for seed in SCREEN_SEEDS],
        }
        for group in NEW_GROUPS
    ]
    rows.sort(key=lambda row: float(row["mean_mae_mw"]))
    best_paired = min(PAIRED_GROUPS, key=lambda group: mean_metric(group, SCREEN_SEEDS))
    best_single = min(SINGLE_SOURCE_GROUPS, key=lambda group: mean_metric(group, SCREEN_SEEDS))
    selected = [best_paired, best_single]
    if best_paired == "r3" and "r2" not in selected:
        selected.append("r2")
    atomic_json(
        ROOT / "selection" / "screening_selection.json",
        {
            "created_unix": int(time.time()),
            "selection_split": "2023_validation",
            "rule": (
                "retain the lowest-mean paired state, the lowest-mean single-source state, "
                "and transition-pair R2 as the identification-loss control when R3 wins"
            ),
            "ranking": rows,
            "a0_ft_mean_mae_mw": mean_metric("c0_a0_ft", SCREEN_SEEDS),
            "d2_mean_mae_mw": mean_metric("c1_d2", SCREEN_SEEDS),
            "best_paired": best_paired,
            "best_single_source": best_single,
            "formal_candidates": selected,
            "test_results_seen": False,
        },
    )
    return selected


def freeze_validation(formal: Sequence[str]) -> tuple[str, str]:
    compared = [*REFERENCE_GROUPS, *formal]
    rows = [
        {
            "group": group,
            "mean_mae_mw": mean_metric(group, FORMAL_SEEDS),
            "sd_mae_mw": statistics.stdev(
                read_metric(group, seed) for seed in FORMAL_SEEDS
            ),
            "seed_mae_mw": [read_metric(group, seed) for seed in FORMAL_SEEDS],
        }
        for group in compared
    ]
    rows.sort(key=lambda row: float(row["mean_mae_mw"]))
    best_response = min(formal, key=lambda group: mean_metric(group, FORMAL_SEEDS))
    selected = str(rows[0]["group"])
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {
            "created_unix": int(time.time()),
            "selection_split": "2023_validation",
            "rule": "lower 10-checkpoint mean MAE wins; exact tie retains the simpler existing model",
            "ranking": rows,
            "selected_final_model": selected,
            "best_response_candidate": best_response,
            "test_results_seen": False,
        },
    )
    return selected, best_response


def run_tests(groups: Sequence[str], slots: Sequence[WorkerSlot]) -> None:
    items = [
        (group, seed)
        for group in groups
        for seed in FORMAL_SEEDS
        if not audited(response_run_dir(group, seed), "test")
    ]
    if not items:
        return

    def runner(item: tuple[str, int], slot: WorkerSlot, _total: int) -> int:
        group, seed = item
        environment = dict(os.environ)
        environment.update(
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:test:{group}:s{seed}]",
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_aggregate_cluster_test.sh"),
                str(slot.gpu), str(response_run_dir(group, seed)),
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "response-frozen-test",
        items,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/seed{item[1]}/test",
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
                values.get("response_state_intervention") == intervention
                and int(values["samples"]) == expected
                and archive["cluster_prediction_mw"].shape == (expected, 24)
                and np.isfinite(archive["cluster_prediction_mw"]).all()
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_interventions(group: str, slots: Sequence[WorkerSlot]) -> None:
    interventions = ("power_shuffle", "weather_shuffle", "pair_shift", "response_mean", "support_only")
    items = [
        (intervention, split, seed)
        for intervention in interventions
        for split in ("val", "test")
        for seed in FORMAL_SEEDS
        if not intervention_valid(group, intervention, split, seed)
    ]
    if not items:
        return

    def runner(item: tuple[str, str, int], slot: WorkerSlot, _total: int) -> int:
        intervention, split, seed = item
        run = response_run_dir(group, seed)
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
        with (output / f"{split}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                    str(PROJECT / "scripts" / "evaluate_aggregate_cluster.py"),
                    "--config", str(run / "config.yaml"),
                    "--checkpoint", str(run / "best_model.pt"),
                    "--split", split,
                    "--output", str(output / f"metrics_{split}.json"),
                    "--predictions-output", str(output / f"predictions_{split}.npz"),
                    "--device", "cuda",
                    "--response-state-intervention", intervention,
                ],
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=log,
            )
        if result.returncode == 0:
            marker.touch()
        if result.returncode:
            return result.returncode
        return 0 if intervention_valid(group, intervention, split, seed) else 4

    execute_parallel(
        "response-interventions",
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
        run = response_run_dir(group, seed)
        output = ROOT / "diagnostics" / group / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot.gpu),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:zero-weather:{group}:s{seed}]",
        )
        return subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "diagnose_aggregate_response_invariance.py"),
                "--config", str(run / "config.yaml"),
                "--checkpoint", str(run / "best_model.pt"),
                "--output", str(output / "zero_weather_invariance.json"),
                "--device", "cuda",
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "response-zero-weather-invariance",
        items,
        slots,
        is_heavy=lambda _seed: False,
        runner=runner,
        item_label=lambda seed: f"{group}/zero-weather/seed{seed}",
        unit="audit",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forcing-realization response-state experiments")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must list distinct physical GPU indices")
    missing_base = [seed for seed in FORMAL_SEEDS if not audited(R2_ROOT / f"seed{seed}")]
    missing_reference = [
        (group, seed) for group in REFERENCE_GROUPS for seed in FORMAL_SEEDS
        if not audited(strict_reference_dir(group, seed))
    ]
    if missing_base:
        raise RuntimeError(f"Missing audited same-seed R2-W checkpoints: {missing_base}")
    if missing_reference:
        raise RuntimeError(f"Missing audited A0-FT/D2 references: {missing_reference}")
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slots = build_worker_slots(gpus, parse_slot_counts(gpus, args.slots, totals))
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous["sha256"] != snapshot["sha256"] and any(ROOT.glob("**/COMPLETE")):
            raise RuntimeError("Scientific code changed after response-state runs began")
    atomic_json(snapshot_path, snapshot)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "objective": "identify weather-power response state from causal forcing-realization pairs",
            "aggregate_only": True,
            "station_history_used": False,
            "new_groups": list(NEW_GROUPS),
            "reference_groups": list(REFERENCE_GROUPS),
            "screen_seeds": list(SCREEN_SEEDS),
            "formal_seeds": list(FORMAL_SEEDS),
            "base_r2_w_root": str(R2_ROOT),
            "gpus": list(gpus),
            "slots": [slot.label for slot in slots],
            "code_sha256": snapshot["sha256"],
        },
    )
    print(f"[response] gpus={list(gpus)} workers={len(slots)}", flush=True)

    cache_response_latents(("train", "val"), gpus, slots)
    run_jobs(
        "response-adapter-screen",
        [Job("screen_adapter", group, seed, "adapter") for group in NEW_GROUPS for seed in SCREEN_SEEDS],
        slots,
    )
    run_jobs(
        "response-joint-screen",
        [
            Job(
                "screen_joint", group, seed, "joint",
                initial("screen_adapter", group, seed),
            )
            for group in NEW_GROUPS for seed in SCREEN_SEEDS
        ],
        slots,
    )
    formal = select_screen_candidates()
    run_jobs(
        "response-formal-adapter",
        [Job("formal_adapter", group, seed, "adapter") for group in formal for seed in REMAINING_SEEDS],
        slots,
    )
    run_jobs(
        "response-formal-joint",
        [
            Job(
                "formal_joint", group, seed, "joint",
                initial("formal_adapter", group, seed),
            )
            for group in formal for seed in REMAINING_SEEDS
        ],
        slots,
    )
    selected, best_response = freeze_validation(formal)
    atomic_json(
        ROOT / "selection" / "test_evaluation_manifest.json",
        {
            "created_unix": int(time.time()),
            "validation_selected": selected,
            "best_response_candidate": best_response,
            "response_groups": list(formal),
            "reference_groups": list(REFERENCE_GROUPS),
            "seeds": list(FORMAL_SEEDS),
        },
    )
    cache_response_latents(("test",), gpus, slots)
    run_tests(formal, slots)
    run_interventions(best_response, slots)
    run_invariance(best_response, slots)
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_aggregate_response_state.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[response] complete selected={selected} best_response={best_response} root={ROOT}", flush=True)


if __name__ == "__main__":
    main()
