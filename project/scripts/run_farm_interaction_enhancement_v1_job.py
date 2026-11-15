from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from wind_forcing.experiment_guard import require_test_authorized

PROJECT = Path(__file__).resolve().parent.parent


def output_names(mode: str):
    if mode == "train":
        return ("COMPLETE", "best_model.pt", "history.json", "metrics_val.json", "predictions_val.npz")
    marker = "TEST_COMPLETE" if mode == "test" else f"{mode.upper()}_COMPLETE"
    return (marker, f"metrics_{mode}.json", f"predictions_{mode}.npz")


def physical_gpu_uuid(index: str) -> str:
    value = subprocess.check_output(["nvidia-smi", f"--id={int(index)}", "--query-gpu=uuid",
                                     "--format=csv,noheader"], text=True).strip()
    if not value.startswith("GPU-") or "\n" in value:
        raise RuntimeError(f"Invalid UUID for physical GPU {index}: {value}")
    return value


def immutable_config(path: Path, expected: dict) -> None:
    if path.exists():
        if yaml.safe_load(path.read_text()) != expected:
            raise RuntimeError(f"Existing config does not match invocation: {path}")
    else:
        path.write_text(yaml.safe_dump(expected, sort_keys=False))


def safe_link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace existing artifact: {target}")
    else:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def next_attempt(run: Path, split: str) -> Path:
    root = run / "attempts"
    root.mkdir(exist_ok=True)
    for number in range(1, 10000):
        path = root / f"{split}_{number:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many attempts")


def recover_publication(run: Path, kind: str, names: tuple[str, ...], marker: str) -> bool:
    """Recover a kill between a successful subprocess and publishing symlinks."""
    if (run / marker).is_file():
        if not all((run / name).is_file() for name in names):
            raise RuntimeError(f"Completion marker has missing artifacts: {run}/{marker}")
        return True
    for attempt in sorted((run / "attempts").glob(f"{kind}_*"), reverse=True):
        if all((attempt / name).is_file() for name in names):
            for name in names:
                safe_link(attempt / name, run / name)
            (run / marker).touch()
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("gpu", "stage", "group", "seed", "history", "forcing", "context", "interaction"):
        parser.add_argument(name)
    parser.add_argument("split", choices=("train", "test", "val_reference", "test_reference", "val_weather", "test_weather"))
    parser.add_argument("readout", nargs="?", default="legacy")
    parser.add_argument("output_mode", nargs="?", default="cf_residual")
    args = parser.parse_args()
    root = Path(os.environ.get("WIND_FARM_INTERACTION_ROOT", PROJECT / "artifacts/farm_interaction_enhancement_v1"))
    run = root / args.stage / args.group / f"seed{int(args.seed)}"
    config = yaml.safe_load((PROJECT / "configs/farm_interaction_enhancement_v1.yaml").read_text())
    config["seed"] = int(args.seed)
    config["model"].update(response_history_mode=args.history, forcing_transition_mode=args.forcing,
                           station_context_mode=args.context, response_interaction_mode=args.interaction,
                           response_readout_mode=args.readout, response_output_mode=args.output_mode)
    config["experiment"] = {"id": f"farm_interaction_enhancement_v1/{args.group}/seed{args.seed}",
                            "protocol_root": str(root.resolve())}
    if args.split != "train":
        require_test_authorized(config, run / "best_model.pt")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_config(run / "config.yaml", config)
        names = output_names(args.split)
        if all((run / name).is_file() for name in names):
            return
        if args.split == "train":
            recover_publication(run, "train", ("best_model.pt", "history.json"), "TRAIN_COMPLETE")
        split = "val" if args.split == "train" else args.split
        marker = names[0]
        if recover_publication(run, args.split, (f"metrics_{split}.json", f"predictions_{split}.npz"), marker):
            return
        attempt = next_attempt(run, args.split)
        status = {"pid": os.getpid(), "attempt": str(attempt), "split": args.split, "started_unix": time.time()}
        (run / "worker.json").write_text(json.dumps(status))
        gpu_uuid = physical_gpu_uuid(args.gpu)
        status.update(physical_gpu=int(args.gpu), gpu_uuid=gpu_uuid)
        (run / "worker.json").write_text(json.dumps(status))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        python = os.environ.get("CHRONOS_PYTHON", sys.executable)
        with (attempt / "run.log").open("w") as log:
            if args.split == "train" and not (run / "TRAIN_COMPLETE").is_file():
                subprocess.run([python, str(PROJECT / "scripts/train_forced_evolution.py"),
                                "--config", str(run / "config.yaml"), "--output-dir", str(attempt),
                                "--device", "cuda", "--epochs", "40"], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                for name in ("best_model.pt", "history.json"):
                    safe_link(attempt / name, run / name)
                (run / "TRAIN_COMPLETE").touch()
            split = "val" if args.split == "train" else args.split
            actual_split = split.split("_", 1)[0]
            intervention = (["--response-state-intervention", "reference"] if split.endswith("_reference") else
                            ["--weather-permutation-seed", "7301"] if split.endswith("_weather") else [])
            subprocess.run([python, str(PROJECT / "scripts/evaluate_forced_evolution.py"),
                            "--config", str(run / "config.yaml"), "--checkpoint", str(run / "best_model.pt"),
                            "--split", actual_split, "--output", str(attempt / f"metrics_{split}.json"),
                            "--predictions-output", str(attempt / f"predictions_{split}.npz"),
                            "--export-station-predictions", "--device", "cuda"] + intervention, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            for name in (f"metrics_{split}.json", f"predictions_{split}.npz"):
                safe_link(attempt / name, run / name)
        (run / marker).touch()


if __name__ == "__main__":
    main()
