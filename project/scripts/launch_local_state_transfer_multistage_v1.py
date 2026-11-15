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
ROOT = PROJECT / "artifacts" / "local_state_transfer_multistage_v1"
SCREEN = (42, 46, 50)
FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL if seed not in SCREEN)
STATION_GROUPS = ("s0", "s_add", "s_int")
E2E_GROUPS = ("e0", "e_add", "e_int")
TRANSFER_GROUPS = ("p0_free", "p_add_free", "p_int_free", "p_int_preserve")
INTERACTION_GROUPS = ("s_int", "e_int", "p_int_free", "p_int_preserve")
INTERVENTIONS = ("strong_sample", "station_permutation", "reference")


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    initial_checkpoint: str = ""
    epochs: int = 40
    split: str = "train"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"


def complete(job: Job) -> bool:
    suffix = "val" if job.split == "train" else "test"
    marker = "COMPLETE" if job.split == "train" else "TEST_COMPLETE"
    return all(
        path.is_file()
        for path in (
            job.run_dir / marker,
            job.run_dir / "best_model.pt",
            job.run_dir / f"metrics_{suffix}.json",
            job.run_dir / f"predictions_{suffix}.npz",
        )
    )


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict[str, Any]:
    return json.loads(
        (ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json").read_text(
            encoding="utf-8"
        )
    )


def values(
    stage: str, group: str, seeds: Sequence[int], key: str, split: str = "val"
) -> list[float]:
    return [float(metric(stage, group, seed, split)[key]) for seed in seeds]


def mean(stage: str, group: str, seeds: Sequence[int], key: str) -> float:
    return statistics.mean(values(stage, group, seeds, key))


def paired_summary(
    stage: str, candidate: str, control: str, seeds: Sequence[int], key: str
) -> dict[str, Any]:
    delta = [
        float(metric(stage, candidate, seed)[key])
        - float(metric(stage, control, seed)[key])
        for seed in seeds
    ]
    return {
        "candidate": candidate,
        "control": control,
        "metric": key,
        "delta_candidate_minus_control": delta,
        "mean_delta": statistics.mean(delta),
        "wins": sum(value < 0.0 for value in delta),
    }


def checkpoint(stage: str, group: str, seed: int) -> str:
    path = ROOT / stage / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def station_source(stage: str, group: str, seed: int) -> str:
    source = {
        "p0_free": "s0",
        "p_add_free": "s_add",
        "p_int_free": "s_int",
        "p_int_preserve": "s_int",
    }[group]
    return checkpoint(stage, source, seed)


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[state-transfer] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        # Small batches are intentional: the user requested several processes
        # per GPU and some devices are concurrently occupied by unrelated work.
        cycles = 5 if total_memory >= 40000 else 3
        environment = dict(os.environ)
        environment.update(
            WIND_STATE_TRANSFER_ROOT=str(ROOT),
            WIND_STATE_TRANSFER_CYCLES_PER_BATCH=str(cycles),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=(
                f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]"
            ),
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_local_state_transfer_job.sh"),
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


def intervention_metric_path(
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
    seeds: Sequence[int],
    split: str,
    slots: Sequence[WorkerSlot],
) -> None:
    items = [
        (group, seed, intervention)
        for group in groups
        for seed in seeds
        for intervention in INTERVENTIONS
        if not intervention_metric_path(view, group, intervention, seed, split).is_file()
    ]
    if not items:
        return

    def runner(item: tuple[str, int, str], slot: WorkerSlot, _total: int) -> int:
        group, seed, intervention = item
        run = ROOT / stage / group / f"seed{seed}"
        output = intervention_metric_path(view, group, intervention, seed, split)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions = output.parent / f"predictions_{split}.npz"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        python_path = f"{PROJECT / 'src'}:{PROJECT}"
        if environment.get("PYTHONPATH"):
            python_path = f"{python_path}:{environment['PYTHONPATH']}"
        environment["PYTHONPATH"] = python_path
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
                command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT
            ).returncode

    execute_parallel(
        f"{view}-interventions-{split}",
        items,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/s{item[1]}/{item[2]}/{split}",
        unit="eval",
    )


def intervention_degradation(
    view: str,
    stage: str,
    group: str,
    seeds: Sequence[int],
    intervention: str,
    key: str,
    split: str = "val",
) -> list[float]:
    result = []
    for seed in seeds:
        perturbed = json.loads(
            intervention_metric_path(view, group, intervention, seed, split).read_text(
                encoding="utf-8"
            )
        )
        result.append(float(perturbed[key]) - float(metric(stage, group, seed, split)[key]))
    return result


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


def link_screen_runs(stage: str, source_stage: str, groups: Sequence[str]) -> None:
    for group in groups:
        for seed in SCREEN:
            source = ROOT / source_stage / group / f"seed{seed}"
            target = ROOT / stage / group / f"seed{seed}"
            if target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve(), target_is_directory=True)


def write_jobs_manifest() -> None:
    rows = ["phase\tgroup\tseeds\tepochs\tobjective\tcheckpoint_metric"]
    rows.extend(
        [
            "station_pretrain\ts0,s_add,s_int\t42-51\t40\tstation_cf\tstation_mae_cf",
            "end_to_end\te0,e_int\t42-51\t60\tcluster+station\tcluster_mae_mw",
            "end_to_end_diagnostic\te_add\t42,46,50 (+formal if needed)\t60\tcluster+station\tcluster_mae_mw",
            "transfer\tp0_free,p_int_free,p_int_preserve\t42-51\t20\tcluster+station\tcluster_mae_mw",
            "transfer_diagnostic\tp_add_free\t42,46,50 (+formal if needed)\t20\tcluster+station\tcluster_mae_mw",
            "interventions\tstrong_sample,station_permutation,reference\tall state runs\t0\tevaluation\tn/a",
        ]
    )
    (ROOT / "jobs.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the decisive local-state pretrain/transfer experiment"
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ROOT / "launcher.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A local-state transfer launcher is already running") from error
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()

    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    counts = parse_slot_counts(gpus, args.slots, totals)
    slots = tuple(
        sorted(build_worker_slots(gpus, counts), key=lambda slot: (slot.slot, slot.gpu))
    )
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
            "gpus": list(gpus),
            "slot_count_by_gpu": dict(zip(map(str, gpus), counts)),
            "stage1_checkpoint_metric": "station_mae_cf",
            "stage2_checkpoint_metric": "cluster_mae_mw",
            "test_read_before_validation_freeze": False,
            "launcher_pid": os.getpid(),
        },
    )
    write_jobs_manifest()

    run_jobs(
        "screen-station",
        [Job("screen_station", group, seed, epochs=40) for group in STATION_GROUPS for seed in SCREEN],
        slots,
    )
    evaluate_interventions(
        "screen_station", "screen_station", ("s_add", "s_int"), SCREEN, "val", slots
    )
    run_jobs(
        "screen-joint",
        [Job("screen_joint", group, seed, epochs=60) for group in E2E_GROUPS for seed in SCREEN],
        slots,
    )
    run_jobs(
        "screen-transfer",
        [
            Job(
                "screen_transfer",
                group,
                seed,
                station_source("screen_station", group, seed),
                epochs=20,
            )
            for group in TRANSFER_GROUPS
            for seed in SCREEN
        ],
        slots,
    )
    evaluate_interventions(
        "screen_joint", "screen_joint", ("e_add", "e_int"), SCREEN, "val", slots
    )
    evaluate_interventions(
        "screen_transfer",
        "screen_transfer",
        ("p_add_free", "p_int_free", "p_int_preserve"),
        SCREEN,
        "val",
        slots,
    )

    screen_comparisons = {
        "station_interaction_vs_none": paired_summary(
            "screen_station", "s_int", "s0", SCREEN, "station_mae_cf"
        ),
        "station_interaction_vs_additive": paired_summary(
            "screen_station", "s_int", "s_add", SCREEN, "station_mae_cf"
        ),
        "e2e_interaction_vs_none": paired_summary(
            "screen_joint", "e_int", "e0", SCREEN, "cluster_mae_mw"
        ),
        "transfer_interaction_vs_none": paired_summary(
            "screen_transfer", "p_int_free", "p0_free", SCREEN, "cluster_mae_mw"
        ),
        "preserve_vs_free": paired_summary(
            "screen_transfer", "p_int_preserve", "p_int_free", SCREEN, "cluster_mae_mw"
        ),
    }
    station_donor = intervention_degradation(
        "screen_station", "screen_station", "s_int", SCREEN,
        "strong_sample", "station_mae_cf",
    )
    state_signal = bool(
        screen_comparisons["station_interaction_vs_none"]["wins"] >= 2
        or screen_comparisons["e2e_interaction_vs_none"]["wins"] >= 2
        or screen_comparisons["transfer_interaction_vs_none"]["wins"] >= 2
        or (statistics.mean(station_donor) > 0.0 and sum(x > 0.0 for x in station_donor) >= 2)
    )
    screen_record = {
        "comparisons": screen_comparisons,
        "station_strong_donor_degradation_cf": station_donor,
        "state_signal": state_signal,
        "formal_core_always_runs": True,
        "additive_formal_expansion": state_signal,
        "test_seen": False,
    }
    atomic_json(ROOT / "selection" / "screening.json", screen_record)

    formal_station_groups = ("s0", "s_int") + (("s_add",) if state_signal else ())
    formal_e2e_groups = ("e0", "e_int") + (("e_add",) if state_signal else ())
    formal_transfer_groups = (
        "p0_free", "p_int_free", "p_int_preserve"
    ) + (("p_add_free",) if state_signal else ())
    run_jobs(
        "formal-station",
        [
            Job("formal_station", group, seed, epochs=40)
            for group in formal_station_groups
            for seed in REMAINING
        ],
        slots,
    )
    run_jobs(
        "formal-joint",
        [
            Job("formal_joint", group, seed, epochs=60)
            for group in formal_e2e_groups
            for seed in REMAINING
        ],
        slots,
    )
    run_jobs(
        "formal-transfer",
        [
            Job(
                "formal_transfer",
                group,
                seed,
                station_source("formal_station", group, seed),
                epochs=20,
            )
            for group in formal_transfer_groups
            for seed in REMAINING
        ],
        slots,
    )
    link_screen_runs("formal_station", "screen_station", formal_station_groups)
    link_screen_runs("formal_joint", "screen_joint", formal_e2e_groups)
    link_screen_runs("formal_transfer", "screen_transfer", formal_transfer_groups)

    evaluate_interventions(
        "formal_station", "formal_station", ("s_int",), FORMAL, "val", slots
    )
    evaluate_interventions(
        "formal_joint", "formal_joint", ("e_int",), FORMAL, "val", slots
    )
    evaluate_interventions(
        "formal_transfer",
        "formal_transfer",
        ("p_int_free", "p_int_preserve"),
        FORMAL,
        "val",
        slots,
    )

    final_cells = [
        ("formal_joint", group) for group in formal_e2e_groups
    ] + [
        ("formal_transfer", group) for group in formal_transfer_groups
    ]
    final_means = {
        f"{stage}/{group}": mean(stage, group, FORMAL, "cluster_mae_mw")
        for stage, group in final_cells
    }
    selected = min(final_means, key=final_means.__getitem__)
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {
            "formal_cluster_mae_means": final_means,
            "selected": selected,
            "seeds": list(FORMAL),
            "test_seen": False,
        },
    )

    run_jobs(
        "frozen-test",
        [Job(stage, group, seed, split="test") for stage, group in final_cells for seed in FORMAL],
        slots,
    )
    evaluate_interventions(
        "formal_station", "formal_station", ("s_int",), FORMAL, "test", slots
    )
    evaluate_interventions(
        "formal_joint", "formal_joint", ("e_int",), FORMAL, "test", slots
    )
    evaluate_interventions(
        "formal_transfer",
        "formal_transfer",
        ("p_int_free", "p_int_preserve"),
        FORMAL,
        "test",
        slots,
    )
    subprocess.run(
        [
            os.environ.get(
                "CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"
            ),
            str(PROJECT / "scripts" / "analyze_local_state_transfer_multistage_v1.py"),
        ],
        cwd=PROJECT,
        check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[state-transfer] COMPLETE selected={selected}", flush=True)


if __name__ == "__main__":
    main()
