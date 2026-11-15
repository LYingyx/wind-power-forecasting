from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wind_forcing.config import load_config


SEEDS = tuple(range(42, 52))
GROUPS = {
    "T0_raw_linear": "t0_raw_linear",
    "T1_latent_linear": "t1_latent_linear",
    "T2_bracket_nonlinear": "t2_bracket_nonlinear",
    "T4_native_query": "l1_anchor_taper",
}
EXPECTED_QUERY = {
    "T0_raw_linear": "raw_linear",
    "T1_latent_linear": "latent_linear",
    "T2_bracket_nonlinear": "bracketing_mlp",
    "T4_native_query": "full_attention",
}
REUSED_SEEDS = {42, 46, 50}
REQUIRED_FILES = (
    "COMPLETE",
    "config.yaml",
    "best_model.pt",
    "metrics_val.json",
    "predictions_val.npz",
)


def run_directory(project: Path, group: str, seed: int) -> tuple[Path, str]:
    if group == "T4_native_query":
        return (
            project
            / "artifacts/horizon_specialist_v1/groups/l1_anchor_taper"
            / f"seed{seed}",
            "reused_horizon_specialist_v1",
        )
    directory = GROUPS[group]
    if seed in REUSED_SEEDS:
        return (
            project
            / "artifacts/cross_resolution_temporal_v1/groups"
            / directory
            / f"seed{seed}",
            "reused_cross_resolution_temporal_v1",
        )
    return (
        project
        / "artifacts/cross_resolution_temporal_v2/formal/groups"
        / directory
        / f"seed{seed}",
        "new_cross_resolution_temporal_v2",
    )


def audit_run(directory: Path, group: str, seed: int) -> tuple[list[str], dict[str, Any]]:
    reasons = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    metadata: dict[str, Any] = {}
    if reasons:
        return reasons, metadata
    try:
        config = load_config(directory / "config.yaml")
        model = config["model"]
        deterministic = config["training"]["deterministic"]
        checks = {
            "seed": int(config["seed"]) == seed,
            "architecture": model["architecture"] == "incremental_correction",
            "time_query_mode": model["time_query_mode"] == EXPECTED_QUERY[group],
            "station_query_mode": model["station_query_mode"] == "learned_cells",
            "station_neighborhood_cells": int(model["station_neighborhood_cells"]) == 3,
            "include_wind_geometry": bool(model["include_wind_geometry"]),
            "forcing_path_mode": model["forcing_path_mode"] == "difference",
            "correction_history_mode": model["correction_history_mode"] == "cluster_recent_1h",
            "correction_fusion_mode": model["correction_fusion_mode"] == "blackbox",
            "matched_conditioning_backbone": bool(model["matched_conditioning_backbone"]),
            "baseline_anchor_enabled": bool(model["baseline_anchor_enabled"]),
            "freeze_history_baseline": not bool(model["freeze_history_baseline"]),
            "anchor_schedule_mode": deterministic["anchor_schedule_mode"] == "horizon_taper",
            "distillation_weight": float(deterministic["distillation_weight"]) == 0.0,
        }
        reasons.extend(name for name, passed in checks.items() if not passed)
        metrics = json.loads((directory / "metrics_val.json").read_text(encoding="utf-8"))
        parameter_count = int(metrics["parameter_count"])
        cluster_mae = float(metrics["cluster_mae_mw"])
        if not np.isfinite(cluster_mae):
            reasons.append("non_finite_metric")
        with np.load(directory / "predictions_val.npz") as archive:
            required_arrays = {
                "origin_unix",
                "sample_index",
                "cluster_prediction_mw",
                "cluster_target_mw",
            }
            if not required_arrays.issubset(archive.files):
                reasons.append("prediction_keys")
                sample_count = 0
            else:
                prediction = archive["cluster_prediction_mw"]
                target = archive["cluster_target_mw"]
                origin = archive["origin_unix"]
                sample = archive["sample_index"]
                sample_count = len(origin)
                if prediction.shape != target.shape or prediction.shape != (sample_count, 24):
                    reasons.append("prediction_shape")
                if sample.shape != origin.shape or len(np.unique(origin)) != sample_count:
                    reasons.append("prediction_identity")
                if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                    reasons.append("non_finite_prediction")
        metadata = {
            "parameter_count": parameter_count,
            "cluster_mae_mw": cluster_mae,
            "validation_samples": sample_count,
            "checkpoint_bytes": (directory / "best_model.pt").stat().st_size,
        }
    except Exception as error:  # preserve all 40 audit entries in one pass
        reasons.append(f"audit_exception:{type(error).__name__}:{error}")
    return sorted(set(reasons)), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit the 40-run temporal manifest")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    formal = project / "artifacts/cross_resolution_temporal_v2/formal"
    formal.mkdir(parents=True, exist_ok=True)
    records = []
    for group in GROUPS:
        for seed in SEEDS:
            directory, source = run_directory(project, group, seed)
            reasons, metadata = audit_run(directory, group, seed)
            records.append(
                {
                    "group": group,
                    "seed": seed,
                    "source": source,
                    "directory": str(directory),
                    "status": "complete" if not reasons else "pending_or_invalid",
                    "audit_failures": reasons,
                    **metadata,
                }
            )
    parameter_counts = {
        record.get("parameter_count")
        for record in records
        if record["status"] == "complete"
    }
    complete = all(record["status"] == "complete" for record in records)
    if complete and len(parameter_counts) != 1:
        complete = False
        for record in records:
            record["audit_failures"].append("formal_parameter_count_mismatch")
            record["status"] = "pending_or_invalid"
    manifest = {
        "protocol": "cross_resolution_temporal_v2_formal_validation_only",
        "expected_runs": 40,
        "complete_runs": sum(record["status"] == "complete" for record in records),
        "all_complete": complete,
        "single_checkpoint_statistics": True,
        "ensemble_predictions_used": False,
        "test_2024_read": False,
        "runs": records,
    }
    manifest_path = formal / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    marker = formal / "FORMAL_COMPLETE"
    if complete:
        marker.touch()
    elif marker.exists():
        marker.unlink()
    print(
        f"[formal-audit] complete={manifest['complete_runs']}/{manifest['expected_runs']} "
        f"manifest={manifest_path}",
        flush=True,
    )
    if args.require_complete and not complete:
        failures = [
            f"{record['group']} seed{record['seed']}: {record['audit_failures']}"
            for record in records
            if record["status"] != "complete"
        ]
        raise SystemExit("Formal audit failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()

