from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
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
ROOT = PROJECT / "artifacts" / "local_state_cluster_transfer_v2"
SOURCE = PROJECT / "artifacts" / "local_state_spatial_interaction_v1"
SEEDS = tuple(range(42, 52))
GROUPS = (
    "b_shared_add",
    "b_independent_add",
    "b_int",
    "f_shared_add",
    "f_independent_add",
    "f_int",
)
INDEPENDENT = ("b_independent_add", "f_independent_add")
INTERVENTIONS = ("strong_sample", "station_permutation", "reference")
SOURCE_GROUP = {
    "b_shared_add": "bilinear_add",
    "b_int": "bilinear_int",
    "f_shared_add": "fixed4_add",
    "f_int": "fixed4_int",
}


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    initial_checkpoint: str = ""
    epochs: int = 20
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


def values(stage: str, group: str, key: str, split: str = "val") -> list[float]:
    return [float(metric(stage, group, seed, split)[key]) for seed in SEEDS]


def paired(stage: str, candidate: str, control: str, key: str) -> dict[str, Any]:
    delta = [
        candidate_value - control_value
        for candidate_value, control_value in zip(
            values(stage, candidate, key), values(stage, control, key), strict=True
        )
    ]
    mean = statistics.mean(delta)
    half = 2.262 * statistics.stdev(delta) / math.sqrt(len(delta))
    return {
        "candidate": candidate,
        "control": control,
        "metric": key,
        "mean_delta": mean,
        "ci95": [mean - half, mean + half],
        "wins": sum(value < 0.0 for value in delta),
        "seed_delta": delta,
    }


def spatial_did(additive: str, key: str) -> dict[str, Any]:
    fixed = [
        candidate - control
        for candidate, control in zip(
            values("transfer", "f_int", key),
            values("transfer", f"f_{additive}", key),
            strict=True,
        )
    ]
    bilinear = [
        candidate - control
        for candidate, control in zip(
            values("transfer", "b_int", key),
            values("transfer", f"b_{additive}", key),
            strict=True,
        )
    ]
    delta = [f - b for f, b in zip(fixed, bilinear, strict=True)]
    mean = statistics.mean(delta)
    half = 2.262 * statistics.stdev(delta) / math.sqrt(len(delta))
    return {
        "additive": additive,
        "metric": key,
        "mean_delta": mean,
        "ci95": [mean - half, mean + half],
        "wins": sum(value < 0.0 for value in delta),
        "seed_delta": delta,
    }


def link_existing_station_runs() -> None:
    station = ROOT / "station"
    intervention = ROOT / "interventions" / "station"
    station.mkdir(parents=True, exist_ok=True)
    intervention.mkdir(parents=True, exist_ok=True)
    for target_group, source_group in SOURCE_GROUP.items():
        pairs = (
            (station / target_group, SOURCE / "formal" / source_group),
            (intervention / target_group, SOURCE / "interventions" / source_group),
        )
        for target, source in pairs:
            if not source.is_dir():
                raise FileNotFoundError(source)
            if target.exists() or target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeError(f"Unexpected existing source link: {target}")
                continue
            target.symlink_to(source.resolve(), target_is_directory=True)


def station_checkpoint(group: str, seed: int) -> str:
    path = ROOT / "station" / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[cluster-transfer-v2] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, _total_memory: int) -> int:
        environment = dict(os.environ)
        environment.update(
            WIND_CLUSTER_TRANSFER_V2_ROOT=str(ROOT),
            WIND_CLUSTER_TRANSFER_V2_CYCLES_PER_BATCH="3",
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=(
                f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}:{job.split}]"
            ),
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_local_state_cluster_transfer_v2_job.sh"),
                str(slot.gpu),
                job.stage,
                job.group,
                str(job.seed),
                job.initial_checkpoint,
                str(job.epochs),
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


def intervention_path(
    view: str, group: str, intervention: str, seed: int, split: str
) -> Path:
    return (
        ROOT
        / "interventions"
        / view
        / group
        / intervention
        / f"seed{seed}"
        / f"metrics_{split}.json"
    )


def evaluate_interventions(
    view: str,
    stage: str,
    groups: Sequence[str],
    split: str,
    slots: Sequence[WorkerSlot],
) -> None:
    items = [
        (group, seed, intervention)
        for group in groups
        for seed in SEEDS
        for intervention in INTERVENTIONS
        if not intervention_path(view, group, intervention, seed, split).is_file()
    ]
    if not items:
        print(f"[cluster-transfer-v2] interventions/{view}/{split} complete", flush=True)
        return

    def runner(item: tuple[str, int, str], slot: WorkerSlot, _total: int) -> int:
        group, seed, intervention = item
        run = ROOT / stage / group / f"seed{seed}"
        output = intervention_path(view, group, intervention, seed, split)
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
        with (output.parent / f"eval_{split}.log").open("w", encoding="utf-8") as stream:
            return subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            ).returncode

    execute_parallel(
        f"cluster-transfer-v2-{view}-interventions-{split}",
        items,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/s{item[1]}/{item[2]}/{split}",
        unit="eval",
    )


def validation_freeze() -> dict[str, Any]:
    station_means = {
        group: statistics.mean(values("station", group, "station_mae_cf"))
        for group in GROUPS
    }
    transfer_means = {
        group: statistics.mean(values("transfer", group, "cluster_mae_mw"))
        for group in GROUPS
    }
    contrasts = {}
    for prefix in ("b", "f"):
        for additive in ("shared_add", "independent_add"):
            contrasts[f"station/{prefix}_int-minus-{prefix}_{additive}"] = paired(
                "station", f"{prefix}_int", f"{prefix}_{additive}", "station_mae_cf"
            )
            contrasts[f"transfer/{prefix}_int-minus-{prefix}_{additive}"] = paired(
                "transfer", f"{prefix}_int", f"{prefix}_{additive}", "cluster_mae_mw"
            )
    did = {
        additive: spatial_did(additive, "cluster_mae_mw")
        for additive in ("shared_add", "independent_add")
    }
    selected = min(transfer_means, key=transfer_means.__getitem__)
    record = {
        "station_mae_cf_means": station_means,
        "transfer_cluster_mae_mw_means": transfer_means,
        "primary_contrasts": contrasts,
        "support_did": did,
        "selected_by_2023_cluster_mae": selected,
        "all_test_groups_frozen": list(GROUPS),
        "seeds": list(SEEDS),
        "rule": (
            "Cluster interaction survives only when Int beats both shared-output and "
            "independent-branch Add controls with paired CI<0 and >=7/10 wins on 2023, "
            "then repeats with CI<0 and >=7/10 wins on frozen 2024. Fixed4 is "
            "interaction-specific only if both additive-control support DIDs pass."
        ),
        "test_seen": False,
    }
    atomic_json(ROOT / "selection" / "validation_freeze.json", record)
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_audit() -> None:
    expected = {
        "b_shared_add": ("raw_bilinear", "output_additive"),
        "b_int": ("raw_bilinear", "faithful_concat"),
        "f_shared_add": ("fixed_gaussian_4x4", "output_additive"),
        "f_int": ("fixed_gaussian_4x4", "faithful_concat"),
    }
    rows = []
    for group, (spatial, interaction) in expected.items():
        for seed in SEEDS:
            run = ROOT / "station" / group / f"seed{seed}"
            config = json.loads(
                subprocess.check_output(
                    [
                        os.environ.get(
                            "CHRONOS_PYTHON",
                            "/data/liying/anaconda3/envs/chronos/bin/python",
                        ),
                        "-c",
                        (
                            "import json,sys,yaml; "
                            "print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"
                        ),
                        str(run / "config.yaml"),
                    ],
                    text=True,
                )
            )
            model = config["model"]
            if model.get("spatial_weather_mode") != spatial:
                raise RuntimeError(f"Spatial source mismatch: {run}")
            if model.get("response_interaction_mode") != interaction:
                raise RuntimeError(f"Interaction source mismatch: {run}")
            checkpoint = run / "best_model.pt"
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "sha256": sha256(checkpoint),
                    "spatial_weather_mode": spatial,
                    "response_interaction_mode": interaction,
                }
            )
    atomic_json(ROOT / "source_checkpoint_audit.json", rows)


def code_snapshot() -> dict[str, Any]:
    paths = (
        PROJECT / "src" / "wind_forcing" / "models" / "local_state_identifiability.py",
        PROJECT / "configs" / "local_state_cluster_transfer_v2.yaml",
        PROJECT / "shell" / "run_local_state_cluster_transfer_v2_job.sh",
        PROJECT / "scripts" / "launch_local_state_cluster_transfer_v2.py",
        PROJECT / "scripts" / "analyze_local_state_cluster_transfer_v2.py",
        PROJECT / "tests" / "test_local_state_identifiability.py",
        PROJECT / "tests" / "test_local_state_cluster_transfer_analysis.py",
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
    parser = argparse.ArgumentParser(description="Run the registered 2x3 cluster transfer")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="8")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ROOT / "launcher.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A cluster-transfer-v2 launcher is already running") from error
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()

    link_existing_station_runs()
    source_audit()
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
            "groups": list(GROUPS),
            "seeds": list(SEEDS),
            "gpus": list(gpus),
            "slot_count_by_gpu": dict(zip(map(str, gpus), counts, strict=True)),
            "station_epochs": 40,
            "transfer_epochs": 20,
            "station_loss_weight": 1.0,
            "cluster_loss_weight": 2.0,
            "primary_metric": "cluster_mae_mw",
            "test_read_before_validation_freeze": False,
            "launcher_pid": os.getpid(),
        },
    )

    run_jobs(
        "independent-add-station-pretrain",
        [Job("station", group, seed, epochs=40) for group in INDEPENDENT for seed in SEEDS],
        slots,
    )
    evaluate_interventions("station", "station", GROUPS, "val", slots)
    run_jobs(
        "matched-cluster-transfer",
        [
            Job(
                "transfer",
                group,
                seed,
                initial_checkpoint=station_checkpoint(group, seed),
                epochs=20,
            )
            for group in GROUPS
            for seed in SEEDS
        ],
        slots,
    )
    evaluate_interventions("transfer", "transfer", GROUPS, "val", slots)

    if not (ROOT / "selection" / "validation_freeze.json").is_file():
        unexpected = list((ROOT / "transfer").glob("*/*/metrics_test.json"))
        unexpected += list((ROOT / "station").glob("*_independent_add/*/metrics_test.json"))
        if unexpected:
            raise RuntimeError(f"Test artifacts exist before validation freeze: {unexpected[:3]}")
        validation_freeze()

    run_jobs(
        "independent-add-frozen-station-test",
        [
            Job("station", group, seed, epochs=40, split="test")
            for group in INDEPENDENT
            for seed in SEEDS
        ],
        slots,
    )
    run_jobs(
        "frozen-cluster-transfer-test",
        [
            Job("transfer", group, seed, epochs=20, split="test")
            for group in GROUPS
            for seed in SEEDS
        ],
        slots,
    )
    evaluate_interventions("station", "station", GROUPS, "test", slots)
    evaluate_interventions("transfer", "transfer", GROUPS, "test", slots)
    subprocess.run(
        [
            os.environ.get(
                "CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"
            ),
            str(PROJECT / "scripts" / "analyze_local_state_cluster_transfer_v2.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[cluster-transfer-v2] complete: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
