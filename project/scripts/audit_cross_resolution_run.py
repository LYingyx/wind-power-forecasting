from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wind_forcing.config import load_config


GROUP_SETTINGS = {
    "t0_raw_linear": ("raw_linear", "relative_distances", "gelu", "actual", "learned_cells", 3, True),
    "t1_latent_linear": ("latent_linear", "relative_distances", "gelu", "actual", "learned_cells", 3, True),
    "t2_bracket_nonlinear": ("bracketing_mlp", "relative_distances", "gelu", "actual", "learned_cells", 3, True),
    "t3_nearest": ("nearest", "relative_distances", "gelu", "actual", "learned_cells", 3, True),
    "t4_native_query": ("full_attention", "relative_distances", "gelu", "actual", "learned_cells", 3, True),
    "t2_phase_off": ("bracketing_mlp", "zero", "gelu", "actual", "learned_cells", 3, True),
    "t2_linear": ("bracketing_mlp", "relative_distances", "identity", "actual", "learned_cells", 3, True),
    "t2_no_diff": ("bracketing_mlp", "relative_distances", "gelu", "zero", "learned_cells", 3, True),
    "s0_raw_bilinear": (None, "relative_distances", "gelu", "actual", "raw_bilinear", 1, False),
    "s1_latent_bilinear": (None, "relative_distances", "gelu", "actual", "bilinear", 1, False),
    "s2_learned_2x2": (None, "relative_distances", "gelu", "actual", "learned_cells", 1, False),
    "s3_fixed_gaussian_4x4": (None, "relative_distances", "gelu", "actual", "fixed_gaussian", 3, False),
    "s4_learned_4x4_nogeom": (None, "relative_distances", "gelu", "actual", "learned_cells", 3, False),
    "s5_learned_4x4_wind": (None, "relative_distances", "gelu", "actual", "learned_cells", 3, True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one resumable cross-resolution run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--group", required=True, choices=sorted(GROUP_SETTINGS))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    required = ("COMPLETE", "config.yaml", "best_model.pt", "metrics_val.json", "predictions_val.npz")
    missing = [name for name in required if not (directory / name).is_file()]
    missing.extend(
        name
        for name in required
        if name != "COMPLETE"
        and (directory / name).is_file()
        and (directory / name).stat().st_size == 0
    )
    if missing:
        raise SystemExit(f"missing or empty artifacts: {missing}")
    config = load_config(directory / "config.yaml")
    model = config["model"]
    expected = GROUP_SETTINGS[args.group]
    actual = (
        model["time_query_mode"],
        model.get("bracket_time_feature_mode", "relative_distances"),
        model.get("bracket_mapper_activation", "gelu"),
        model.get("forcing_difference_input", "actual"),
        model["station_query_mode"],
        int(model["station_neighborhood_cells"]),
        bool(model["include_wind_geometry"]),
    )
    if expected[0] is None:
        # Spatial controls inherit the already-frozen temporal mechanism;
        # their audit constrains only the spatial support/mapping fields.
        expected = (*actual[:4], *expected[4:])
    if int(config["seed"]) != args.seed or actual != expected:
        raise SystemExit(f"configuration mismatch: actual={actual}, expected={expected}")
    settings = config["training"]["deterministic"]
    if settings["anchor_schedule_mode"] != "horizon_taper" or float(settings["distillation_weight"]) != 0:
        raise SystemExit("training-protocol mismatch")
    metrics = json.loads((directory / "metrics_val.json").read_text(encoding="utf-8"))
    if not np.isfinite(float(metrics["cluster_mae_mw"])) or int(metrics["parameter_count"]) <= 0:
        raise SystemExit("invalid metrics")
    with np.load(directory / "predictions_val.npz") as archive:
        prediction = archive["cluster_prediction_mw"]
        target = archive["cluster_target_mw"]
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 24:
            raise SystemExit("invalid prediction shape")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise SystemExit("non-finite predictions")
    print(f"[run-audit] valid group={args.group} seed={args.seed} dir={directory}")


if __name__ == "__main__":
    main()
