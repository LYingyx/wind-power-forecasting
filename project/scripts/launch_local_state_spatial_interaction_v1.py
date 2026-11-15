from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
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
ROOT = PROJECT / "artifacts" / "local_state_spatial_interaction_v1"
SCREEN = (42, 46, 50)
FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL if seed not in SCREEN)
SPATIALS = ("bilinear", "fixed4", "learned4")
OPERATORS = ("none", "add", "int")
GROUPS = tuple(f"{spatial}_{operator}" for spatial in SPATIALS for operator in OPERATORS)
STATE_GROUPS = tuple(group for group in GROUPS if not group.endswith("_none"))
INTERVENTIONS = ("strong_sample", "station_permutation", "reference")


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
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
            job.run_dir / "best_model.pt",
            job.run_dir / "metrics_test.json",
            job.run_dir / "predictions_test.npz",
        )
    return all(path.is_file() for path in required)


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict[str, Any]:
    return json.loads(
        (ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json").read_text(
            encoding="utf-8"
        )
    )


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[spatial-interaction] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, _total_memory: int) -> int:
        environment = dict(os.environ)
        environment.update(
            WIND_SPATIAL_INTERACTION_ROOT=str(ROOT),
            WIND_SPATIAL_INTERACTION_CYCLES_PER_BATCH="3",
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=(
                f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}:{job.split}]"
            ),
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_local_state_spatial_interaction_job.sh"),
                str(slot.gpu),
                job.stage,
                job.group,
                str(job.seed),
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


def link_screen_runs() -> None:
    for group in GROUPS:
        for seed in SCREEN:
            source = ROOT / "screen" / group / f"seed{seed}"
            target = ROOT / "formal" / group / f"seed{seed}"
            if target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve(), target_is_directory=True)


def intervention_path(
    group: str, intervention: str, seed: int, split: str
) -> Path:
    return (
        ROOT
        / "interventions"
        / group
        / intervention
        / f"seed{seed}"
        / f"metrics_{split}.json"
    )


def evaluate_interventions(
    split: str, groups: Sequence[str], seeds: Sequence[int], slots: Sequence[WorkerSlot]
) -> None:
    items = [
        (group, seed, intervention)
        for group in groups
        for seed in seeds
        for intervention in INTERVENTIONS
        if not intervention_path(group, intervention, seed, split).is_file()
    ]
    if not items:
        print(f"[spatial-interaction] interventions/{split} already complete", flush=True)
        return

    def runner(item: tuple[str, int, str], slot: WorkerSlot, _total: int) -> int:
        group, seed, intervention = item
        run = ROOT / "formal" / group / f"seed{seed}"
        output = intervention_path(group, intervention, seed, split)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions = output.parent / f"predictions_{split}.npz"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        environment["PYTHONPATH"] = (
            f"{PROJECT / 'src'}:{PROJECT}"
            + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else "")
        )
        command = [
            os.environ.get(
                "CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"
            ),
            str(PROJECT / "scripts" / "evaluate_forced_evolution.py"),
            "--config",
            str(run / "config.yaml"),
            "--checkpoint",
            str(run / "best_model.pt"),
            "--split",
            split,
            "--output",
            str(output),
            "--predictions-output",
            str(predictions),
            "--export-station-predictions",
            "--response-state-intervention",
            intervention,
            "--device",
            "cuda",
        ]
        log = output.parent / f"eval_{split}.log"
        with log.open("w", encoding="utf-8") as stream:
            return subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            ).returncode

    execute_parallel(
        f"spatial-interaction-interventions-{split}",
        items,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/s{item[1]}/{item[2]}/{split}",
        unit="eval",
    )


def paired_delta(candidate: str, control: str, split: str = "val") -> list[float]:
    return [
        float(metric("formal", candidate, seed, split)["station_mae_cf"])
        - float(metric("formal", control, seed, split)["station_mae_cf"])
        for seed in FORMAL
    ]


def validation_freeze() -> dict[str, Any]:
    means = {
        group: statistics.mean(
            float(metric("formal", group, seed)["station_mae_cf"])
            for seed in FORMAL
        )
        for group in GROUPS
    }
    interaction_minus_additive = {
        spatial: paired_delta(f"{spatial}_int", f"{spatial}_add")
        for spatial in SPATIALS
    }
    bilinear = interaction_minus_additive["bilinear"]
    did = {
        spatial: [
            interaction_minus_additive[spatial][index] - bilinear[index]
            for index in range(len(FORMAL))
        ]
        for spatial in ("fixed4", "learned4")
    }
    record = {
        "formal_station_mae_cf_means": means,
        "primary_contrasts": {
            "interaction_minus_additive_by_spatial_support": interaction_minus_additive,
            "interaction_support_difference_in_differences_vs_bilinear": did,
        },
        "primary_interpretation_rule": (
            "A weather-state modulation claim requires Int-Add < 0 with a paired 95% "
            "CI below zero. A surrounding-weather-specific claim additionally requires "
            "the 4x4-vs-bilinear interaction DID < 0 with its paired 95% CI below zero."
        ),
        "all_test_groups_frozen": list(GROUPS),
        "seeds": list(FORMAL),
        "test_seen": False,
    }
    atomic_json(ROOT / "selection" / "validation_freeze.json", record)
    return record


def code_snapshot() -> dict[str, Any]:
    paths = (
        PROJECT / "src" / "wind_forcing" / "models" / "local_state_identifiability.py",
        PROJECT / "src" / "wind_forcing" / "models" / "components.py",
        PROJECT / "configs" / "local_state_spatial_interaction_v1.yaml",
        PROJECT / "shell" / "run_local_state_spatial_interaction_job.sh",
        PROJECT / "scripts" / "launch_local_state_spatial_interaction_v1.py",
        PROJECT / "scripts" / "analyze_local_state_spatial_interaction_v1.py",
        PROJECT / "scripts" / "prepare_spatial_interaction_state_sidecar.py",
    )
    digest = hashlib.sha256()
    files = []
    for path in paths:
        relative = str(path.relative_to(PROJECT))
        files.append(relative)
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": files, "created_unix": int(time.time())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run spatial weather support x local-history mechanism experiment"
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="3")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ROOT / "launcher.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A spatial-interaction launcher is already running") from error
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()

    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    counts = parse_slot_counts(gpus, args.slots, totals)
    slots = tuple(sorted(build_worker_slots(gpus, counts), key=lambda x: (x.slot, x.gpu)))
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        prior = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if prior["sha256"] != snapshot["sha256"]:
            raise RuntimeError("Scientific code changed after this experiment began")
    else:
        atomic_json(snapshot_path, snapshot)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "screen_seeds": list(SCREEN),
            "formal_seeds": list(FORMAL),
            "spatial_supports": list(SPATIALS),
            "history_operators": list(OPERATORS),
            "gpus": list(gpus),
            "slot_count_by_gpu": dict(zip(map(str, gpus), counts)),
            "primary_metric": "station_mae_cf",
            "primary_contrast": "interaction_minus_additive",
            "test_read_before_validation_freeze": False,
            "launcher_pid": os.getpid(),
        },
    )

    subprocess.run(
        [
            os.environ.get(
                "CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"
            ),
            str(PROJECT / "scripts" / "prepare_spatial_interaction_state_sidecar.py"),
        ],
        cwd=PROJECT,
        check=True,
    )

    run_jobs(
        "spatial-interaction-screen",
        [Job("screen", group, seed) for group in GROUPS for seed in SCREEN],
        slots,
    )
    run_jobs(
        "spatial-interaction-formal",
        [Job("formal", group, seed) for group in GROUPS for seed in REMAINING],
        slots,
    )
    link_screen_runs()
    evaluate_interventions("val", STATE_GROUPS, FORMAL, slots)
    validation_freeze()
    run_jobs(
        "spatial-interaction-frozen-test",
        [Job("formal", group, seed, split="test") for group in GROUPS for seed in FORMAL],
        slots,
    )
    evaluate_interventions("test", STATE_GROUPS, FORMAL, slots)
    subprocess.run(
        [
            os.environ.get(
                "CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"
            ),
            str(PROJECT / "scripts" / "analyze_local_state_spatial_interaction_v1.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[spatial-interaction] complete: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
