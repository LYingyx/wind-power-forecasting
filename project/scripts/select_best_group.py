from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
from typing import Any


REQUIRED = ("config.yaml", "best_model.pt", "metrics_val.json", "history.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select an experiment family by mean validation metric across seeds"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="cluster_mae_mw")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value]
    if len(seeds) < 2:
        raise ValueError("Group selection requires at least two seeds")
    records: list[dict[str, Any]] = []
    for raw_candidate in args.candidates:
        candidate = Path(raw_candidate).resolve()
        seed_records = []
        for seed in seeds:
            directory = candidate / f"seed{seed}"
            missing = [
                str(directory / name)
                for name in (*REQUIRED, "COMPLETE")
                if not (directory / name).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Candidate {candidate} seed {seed} is incomplete: {missing}"
                )
            metrics = load_json(directory / "metrics_val.json")
            seed_records.append(
                {
                    "seed": seed,
                    "directory": str(directory),
                    "metric_value": float(metrics[args.metric]),
                    "metrics": metrics,
                }
            )
        values = [record["metric_value"] for record in seed_records]
        records.append(
            {
                "experiment_id": candidate.name,
                "directory": str(candidate),
                "metric": args.metric,
                "metric_mean": statistics.mean(values),
                "metric_sample_std": statistics.stdev(values),
                "seeds": seed_records,
            }
        )
    records.sort(key=lambda item: (item["metric_mean"], item["experiment_id"]))
    winner = records[0]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for seed_record in winner["seeds"]:
        source = Path(seed_record["directory"])
        destination = output / f"seed{seed_record['seed']}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED:
            atomic_copy(source / name, destination / name)
        (destination / "COMPLETE").touch()
    selection = {
        "selection_split": "val",
        "selection_rule": f"minimum_seed_mean_{args.metric}",
        "seeds": seeds,
        "winner": winner,
        "ranking": records,
    }
    temporary = output / f".selection.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=True, indent=2)
    os.replace(temporary, output / "selection.json")
    (output / "COMPLETE").touch()
    print(
        f"[select-group] winner={winner['experiment_id']} "
        f"mean_{args.metric}={winner['metric_mean']:.6f} "
        f"std={winner['metric_sample_std']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
