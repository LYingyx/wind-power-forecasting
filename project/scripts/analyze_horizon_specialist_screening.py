from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_incremental_mechanism_v2 import group_predictions


SEEDS = [42, 46, 50]
HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)


def seed_mae(prediction: dict[str, np.ndarray]) -> float:
    return float(
        np.abs(
            prediction["cluster_prediction_mw"] - prediction["cluster_target_mw"]
        ).mean()
    )


def load_group(path: Path) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    missing = [
        path / f"seed{seed}/predictions_val.npz"
        for seed in SEEDS
        if not (path / f"seed{seed}/predictions_val.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing screening predictions:\n" + "\n".join(map(str, missing)))
    return group_predictions(path, SEEDS, "val")


def compare(
    name: str,
    candidate: tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]],
    reference_name: str,
    reference: tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]],
) -> dict[str, Any]:
    candidate_seeds, candidate_ensemble = candidate
    reference_seeds, reference_ensemble = reference
    differences = [
        seed_mae(left) - seed_mae(right)
        for left, right in zip(candidate_seeds, reference_seeds, strict=True)
    ]
    candidate_error = np.abs(
        candidate_ensemble["cluster_prediction_mw"]
        - candidate_ensemble["cluster_target_mw"]
    )
    reference_error = np.abs(
        reference_ensemble["cluster_prediction_mw"]
        - reference_ensemble["cluster_target_mw"]
    )
    row: dict[str, Any] = {
        "candidate": name,
        "reference": reference_name,
        "seed_mean_delta_mw": float(np.mean(differences)),
        "seed_deltas_mw": differences,
        "wins": int(np.sum(np.asarray(differences) < 0.0)),
        "ensemble_delta_mw": float(candidate_error.mean() - reference_error.mean()),
        "ensemble_is_diagnostic_only": True,
    }
    worst = -np.inf
    for band, selection in HORIZON_BANDS:
        seed_band_deltas = [
            float(
                np.abs(
                    left["cluster_prediction_mw"][:, selection]
                    - left["cluster_target_mw"][:, selection]
                ).mean()
                - np.abs(
                    right["cluster_prediction_mw"][:, selection]
                    - right["cluster_target_mw"][:, selection]
                ).mean()
            )
            for left, right in zip(candidate_seeds, reference_seeds, strict=True)
        ]
        delta = float(np.mean(seed_band_deltas))
        row[f"{band}_delta_mw"] = delta
        row[f"{band}_seed_deltas_mw"] = seed_band_deltas
        worst = max(worst, delta)
    row["worst_horizon_delta_mw"] = float(worst)
    row["passes_screen"] = bool(
        row["seed_mean_delta_mw"] <= -1.0
        and row["wins"] >= 2
        and row["worst_horizon_delta_mw"] <= 1.0
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply preregistered 3-seed screening rules")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--root", default="artifacts/horizon_specialist_v1")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = (project / args.root).resolve()
    output = root / "analysis_screening"
    output.mkdir(parents=True, exist_ok=True)
    normal_path = project / "artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h"
    normal = load_group(normal_path)
    available = {
        path.name: load_group(path)
        for path in sorted((root / "groups").iterdir())
        if path.is_dir()
        and all((path / f"seed{seed}/predictions_val.npz").is_file() for seed in SEEDS)
    }
    rows = [compare(name, value, "Normal", normal) for name, value in available.items()]
    if "h0_dual_full_taper" in available and "h1_dual_mixed_taper" in available:
        rows.append(
            compare(
                "h1_dual_mixed_taper",
                available["h1_dual_mixed_taper"],
                "h0_dual_full_taper",
                available["h0_dual_full_taper"],
            )
        )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparisons.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    primary = {row["candidate"]: row for row in rows if row["reference"] == "Normal"}
    h1_control = next(
        (
            row
            for row in rows
            if row["candidate"] == "h1_dual_mixed_taper"
            and row["reference"] == "h0_dual_full_taper"
        ),
        None,
    )
    decision = {
        "protocol": {
            "split": "2023 validation",
            "seeds": SEEDS,
            "screening_only": True,
            "primary_unit": "independent single checkpoint",
            "ensemble_role": "diagnostic only",
            "pass_rule": "mean improvement >=1 MW, >=2/3 wins, worst band degradation <=1 MW",
        },
        "comparisons": rows,
        "run_l4": bool(
            primary.get("l3_fixed_view_taper", {}).get("passes_screen", False)
        ),
        "run_h2": bool(
            primary.get("h1_dual_mixed_taper", {}).get("passes_screen", False)
            and h1_control is not None
            and float(h1_control["seed_mean_delta_mw"]) < 0.0
        ),
    }
    light_candidates = [
        row
        for name, row in primary.items()
        if name.startswith("l") and bool(row.get("passes_screen", False))
    ]
    high_candidates = [
        row
        for name, row in primary.items()
        if name in {"h1_dual_mixed_taper", "h2_dual_learned_taper"}
        and bool(row.get("passes_screen", False))
    ]
    decision["selected_lightweight"] = (
        min(light_candidates, key=lambda row: row["seed_mean_delta_mw"])["candidate"]
        if light_candidates
        else None
    )
    decision["selected_high_cost"] = (
        min(high_candidates, key=lambda row: row["seed_mean_delta_mw"])["candidate"]
        if high_candidates
        else None
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
