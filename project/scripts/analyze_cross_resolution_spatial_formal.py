from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SEEDS = tuple(range(42, 52))
SCREEN_SEEDS = {42, 46, 50}
PAIRS = {
    "S2_learned_2x2": ("s2_learned_2x2", "S1_latent_bilinear", "s1_latent_bilinear"),
    "S4_learned_4x4_nogeom": ("s4_learned_4x4_nogeom", "S3_fixed_gaussian_4x4", "s3_fixed_gaussian_4x4"),
}
SETTINGS = {
    "S0_raw_bilinear": ("raw_bilinear", 1, False),
    "S1_latent_bilinear": ("bilinear", 1, False),
    "S2_learned_2x2": ("learned_cells", 1, False),
    "S3_fixed_gaussian_4x4": ("fixed_gaussian", 3, False),
    "S4_learned_4x4_nogeom": ("learned_cells", 3, False),
    "S5_learned_4x4_wind": ("learned_cells", 3, True),
}
HORIZONS = {
    "0.25-1h": np.arange(0, 4),
    "1.25-3h": np.arange(4, 12),
    "3.25-6h": np.arange(12, 24),
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def paired_ci(values: np.ndarray) -> tuple[float, float]:
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def variant_path(root: Path, group: str, seed: int) -> Path:
    stage = "screening" if seed in SCREEN_SEEDS else "formal"
    return root / f"spatial/{stage}/groups/{group}/seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formally expanded spatial support controls")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = project / "artifacts/cross_resolution_temporal_v2"
    screen = json.loads((root / "spatial/analysis_screening/decision.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "temporal_selection.json").read_text(encoding="utf-8"))
    expanded = set(screen.get("formal_expansion_groups", ()))
    selected_temporal_paths = {int(record["seed"]): Path(record["directory"]) for record in selection["runs"]}

    rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    means: dict[str, float] = {}
    passed_learned: list[str] = []
    for label, (learned_group, reference_label, reference_group) in PAIRS.items():
        if learned_group not in expanded:
            continue
        if reference_group not in expanded:
            raise SystemExit(f"Missing parameter-matched reference expansion: {reference_group}")
        differences = []
        band_differences = {band: [] for band in HORIZONS}
        learned_maes = []
        reference_maes = []
        for seed in SEEDS:
            learned_dir = variant_path(root, learned_group, seed)
            reference_dir = variant_path(root, reference_group, seed)
            for directory in (learned_dir, reference_dir):
                for name in ("COMPLETE", "best_model.pt", "predictions_val.npz"):
                    if not (directory / name).is_file():
                        raise SystemExit(f"Incomplete spatial run: {directory / name}")
            learned = load(learned_dir / "predictions_val.npz")
            reference = load(reference_dir / "predictions_val.npz")
            np.testing.assert_array_equal(learned["origin_unix"], reference["origin_unix"])
            np.testing.assert_array_equal(learned["cluster_target_mw"], reference["cluster_target_mw"])
            learned_error = np.abs(learned["cluster_prediction_mw"] - learned["cluster_target_mw"])
            reference_error = np.abs(reference["cluster_prediction_mw"] - reference["cluster_target_mw"])
            differences.append(float((learned_error - reference_error).mean()))
            for band, indices in HORIZONS.items():
                value = float((learned_error[:, indices] - reference_error[:, indices]).mean())
                band_differences[band].append(value)
                horizon_rows.append(
                    {
                        "learned_group": label,
                        "fixed_reference": reference_label,
                        "seed": seed,
                        "horizon_band": band,
                        "learned_minus_fixed_mw": value,
                    }
                )
            learned_maes.append(float(learned_error.mean()))
            reference_maes.append(float(reference_error.mean()))
        values = np.asarray(differences)
        low, high = paired_ci(values)
        worst_band_mean = max(float(np.mean(values)) for values in band_differences.values())
        passed = bool(
            values.mean() <= -1.0
            and high < 0
            and np.sum(values < 0) >= 7
            and worst_band_mean <= 1.0
        )
        if passed:
            passed_learned.append(label)
        means[label] = float(np.mean(learned_maes))
        means[reference_label] = float(np.mean(reference_maes))
        rows.append(
            {
                "learned_group": label,
                "fixed_reference": reference_label,
                "learned_minus_fixed_mean_mw": float(values.mean()),
                "sd_mw": float(values.std(ddof=1)),
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "learned_wins": int(np.sum(values < 0)),
                "worst_horizon_band_mean_delta_mw": worst_band_mean,
                "formal_pass": passed,
                "seed_differences_mw": json.dumps(differences),
            }
        )

    # S5 is the already-trained temporal winner with the current learned 4x4
    # wind-aware front end, so all ten validation checkpoints are reusable.
    s5_maes = []
    for seed in SEEDS:
        run = load(selected_temporal_paths[seed] / "predictions_val.npz")
        s5_maes.append(float(np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"]).mean()))
    means["S5_learned_4x4_wind"] = float(np.mean(s5_maes))

    if passed_learned:
        # Once a learned projection survives its matched fixed control, choose
        # the simplest supported learned option within 1 MW of the best
        # supported learned/current front end.
        eligible = [*passed_learned, "S5_learned_4x4_wind"]
        best = min(means[label] for label in eligible)
        order = ("S2_learned_2x2", "S4_learned_4x4_nogeom", "S5_learned_4x4_wind")
        selected = next(label for label in order if label in eligible and means[label] <= best + 1.0)
        selection_basis = "formal_10seed_learned_support"
    else:
        selected = screen["screening_selection_within_1mw"]
        selection_basis = "screening_simplest_within_1mw_no_learned_claim"
    mode, cells, wind_geometry = SETTINGS[selected]
    decision = {
        "data_split": "2023_validation",
        "test_2024_read": False,
        "single_checkpoint_statistics": True,
        "expanded_groups": sorted(expanded),
        "formally_supported_learned_groups": passed_learned,
        "selected_spatial_group": selected,
        "selected_station_query_mode": mode,
        "selected_station_neighborhood_cells": cells,
        "selected_include_wind_geometry": wind_geometry,
        "selection_basis": selection_basis,
        "spatial_structure_frozen": True,
        "paper_spatial_claim": bool(passed_learned),
    }
    output = root / "spatial/analysis_formal"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_formal_comparisons.csv", rows)
    write_csv(output / "per_horizon_band.csv", horizon_rows)
    write_csv(
        output / "candidate_means.csv",
        [{"group": label, "mae_mean_mw": value} for label, value in sorted(means.items())],
    )
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
