from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return values


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a validation winner")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="cluster_mae_mw")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for raw_candidate in args.candidates:
        candidate = Path(raw_candidate).resolve()
        required = {
            "config": candidate / "config.yaml",
            "checkpoint": candidate / "best_model.pt",
            "metrics": candidate / "metrics_val.json",
            "history": candidate / "history.json",
            "complete": candidate / "COMPLETE",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Candidate {candidate.name} is incomplete: {missing}"
            )
        metrics = load_json(required["metrics"])
        if args.metric not in metrics:
            raise KeyError(f"Metric {args.metric} is absent from {required['metrics']}")
        records.append(
            {
                "experiment_id": candidate.name,
                "directory": str(candidate),
                "metric": args.metric,
                "metric_value": float(metrics[args.metric]),
                "metrics": metrics,
            }
        )
    records.sort(key=lambda item: (item["metric_value"], item["experiment_id"]))
    winner = records[0]
    source = Path(winner["directory"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "best_model.pt", "metrics_val.json", "history.json"):
        atomic_copy(source / name, output / name)
    selection = {
        "selection_split": "val",
        "selection_rule": f"minimum_{args.metric}",
        "winner": winner,
        "ranking": records,
    }
    temporary = output / f".selection.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=True, indent=2)
    os.replace(temporary, output / "selection.json")
    (output / "COMPLETE").touch()
    print(
        f"[select] winner={winner['experiment_id']}, "
        f"{args.metric}={winner['metric_value']:.6f}"
    )


if __name__ == "__main__":
    main()
