from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wind_forcing.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one final Normal/KD checkpoint")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--group", choices=("normal", "kd"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    directory = Path(args.run_dir).resolve()
    temporal = json.loads(
        (project / "artifacts/cross_resolution_temporal_v2/temporal_selection.json").read_text(encoding="utf-8")
    )
    spatial = json.loads(
        (project / "artifacts/cross_resolution_temporal_v2/spatial/analysis_formal/decision.json").read_text(encoding="utf-8")
    )
    required = ("COMPLETE", "config.yaml", "best_model.pt", "metrics_val.json", "predictions_val.npz")
    missing = [name for name in required if not (directory / name).is_file()]
    missing.extend(
        name
        for name in required
        if name != "COMPLETE" and (directory / name).is_file() and (directory / name).stat().st_size == 0
    )
    if missing:
        raise SystemExit(f"missing or empty artifacts: {missing}")
    config = load_config(directory / "config.yaml")
    model = config["model"]
    settings = config["training"]["deterministic"]
    expected_kd = 0.5 if args.group == "kd" else 0.0
    checks = {
        "seed": int(config["seed"]) == args.seed,
        "time_query_mode": model["time_query_mode"] == temporal["selected_time_query_mode"],
        "bracket_time_feature_mode": model.get("bracket_time_feature_mode", "relative_distances")
        == temporal["selected_bracket_time_feature_mode"],
        "bracket_mapper_activation": model.get("bracket_mapper_activation", "gelu")
        == temporal["selected_bracket_mapper_activation"],
        "forcing_difference_input": model.get("forcing_difference_input", "actual")
        == temporal["selected_forcing_difference_input"],
        "station_query_mode": model["station_query_mode"] == spatial["selected_station_query_mode"],
        "station_neighborhood_cells": int(model["station_neighborhood_cells"])
        == int(spatial["selected_station_neighborhood_cells"]),
        "include_wind_geometry": bool(model["include_wind_geometry"])
        == bool(spatial["selected_include_wind_geometry"]),
        "anchor_schedule": settings["anchor_schedule_mode"] == "horizon_taper",
        "distillation_weight": float(settings["distillation_weight"]) == expected_kd,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"final configuration mismatch: {failed}")
    metrics = json.loads((directory / "metrics_val.json").read_text(encoding="utf-8"))
    if not np.isfinite(float(metrics["cluster_mae_mw"])) or int(metrics["parameter_count"]) <= 0:
        raise SystemExit("invalid validation metrics")
    with np.load(directory / "predictions_val.npz") as archive:
        prediction = archive["cluster_prediction_mw"]
        target = archive["cluster_target_mw"]
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 24:
            raise SystemExit("invalid validation prediction shape")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise SystemExit("non-finite validation prediction")
    if args.group == "kd":
        import torch

        checkpoint = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
        teacher = checkpoint.get("distillation_teacher")
        if not teacher or teacher.get("kind") != "ensemble_mean" or teacher.get("seeds") != list(range(42, 52)):
            raise SystemExit(f"invalid KD teacher provenance: {teacher}")
    print(f"[final-audit] valid group={args.group} seed={args.seed} dir={directory}")


if __name__ == "__main__":
    main()
