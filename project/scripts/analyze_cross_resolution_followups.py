from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SCREEN_SEEDS = (42, 46, 50)
MECHANISM = {
    "T2_PhaseOff": "t2_phase_off",
    "T2_Linear": "t2_linear",
    "T2_NoDiff": "t2_no_diff",
}
SPATIAL = {
    "S0_raw_bilinear": "s0_raw_bilinear",
    "S1_latent_bilinear": "s1_latent_bilinear",
    "S2_learned_2x2": "s2_learned_2x2",
    "S3_fixed_gaussian_4x4": "s3_fixed_gaussian_4x4",
    "S4_learned_4x4_nogeom": "s4_learned_4x4_nogeom",
}
SPATIAL_PAIRS = (
    ("S0_vs_S1", "S0_raw_bilinear", "S1_latent_bilinear"),
    ("S1_vs_S2", "S1_latent_bilinear", "S2_learned_2x2"),
    ("S3_vs_S4", "S3_fixed_gaussian_4x4", "S4_learned_4x4_nogeom"),
    ("S1_vs_S3", "S1_latent_bilinear", "S3_fixed_gaussian_4x4"),
    ("S2_vs_S4", "S2_learned_2x2", "S4_learned_4x4_nogeom"),
    ("S4_vs_S5", "S4_learned_4x4_nogeom", "S5_learned_4x4_wind"),
)
HORIZONS = {
    "0.25-1h": np.arange(0, 4),
    "1.25-3h": np.arange(4, 12),
    "3.25-6h": np.arange(12, 24),
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def mae(run: dict[str, np.ndarray]) -> float:
    return float(np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"]).mean())


def interval(values: np.ndarray) -> tuple[float, float]:
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


def formal_paths(project: Path) -> dict[tuple[str, int], Path]:
    manifest = json.loads(
        (project / "artifacts/cross_resolution_temporal_v2/formal/run_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest["all_complete"]:
        raise SystemExit("Formal temporal manifest is incomplete")
    return {(record["group"], int(record["seed"])): Path(record["directory"]) for record in manifest["runs"]}


def analyze_mechanism(project: Path) -> None:
    root = project / "artifacts/cross_resolution_temporal_v2/mechanism/screening"
    output = project / "artifacts/cross_resolution_temporal_v2/mechanism/analysis_screening"
    output.mkdir(parents=True, exist_ok=True)
    paths = formal_paths(project)
    reference = {seed: load(paths[("T2_bracket_nonlinear", seed)] / "predictions_val.npz") for seed in SCREEN_SEEDS}
    rows, horizon_rows = [], []
    decisions: dict[str, Any] = {"data_split": "2023_validation", "test_2024_read": False}
    expansion = []
    for label, directory in MECHANISM.items():
        differences = []
        for seed in SCREEN_SEEDS:
            run = load(root / "groups" / directory / f"seed{seed}" / "predictions_val.npz")
            ref = reference[seed]
            np.testing.assert_array_equal(run["origin_unix"], ref["origin_unix"])
            error = np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"])
            ref_error = np.abs(ref["cluster_prediction_mw"] - ref["cluster_target_mw"])
            differences.append(float((error - ref_error).mean()))
            for band, indices in HORIZONS.items():
                horizon_rows.append(
                    {"group": label, "seed": seed, "horizon_band": band, "candidate_minus_t2_mw": float((error[:, indices] - ref_error[:, indices]).mean())}
                )
        values = np.asarray(differences)
        low, high = interval(values)
        rows.append(
            {"group": label, "candidate_minus_t2_mean_mw": float(values.mean()), "ci95_low_mw": low, "ci95_high_mw": high, "same_direction_count": int(np.sum(values > 0)), "seed_differences_mw": json.dumps(differences)}
        )
        if label in {"T2_PhaseOff", "T2_Linear"}:
            qualifies = values.mean() >= 0.75 and np.sum(values > 0) == 3
        else:
            qualifies = values.mean() <= 0.5
        decisions[f"{label}_expand_to_10"] = bool(qualifies)
        if qualifies:
            expansion.append(directory)
    decisions["formal_expansion_groups"] = expansion
    write_csv(output / "paired_screening.csv", rows)
    write_csv(output / "per_horizon_band.csv", horizon_rows)
    (output / "decision.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "SCREENING_COMPLETE").touch()
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


def analyze_spatial(project: Path) -> None:
    root = project / "artifacts/cross_resolution_temporal_v2/spatial/screening"
    output = project / "artifacts/cross_resolution_temporal_v2/spatial/analysis_screening"
    output.mkdir(parents=True, exist_ok=True)
    formal = formal_paths(project)
    temporal_selection = json.loads(
        (project / "artifacts/cross_resolution_temporal_v2/temporal_selection.json").read_text(encoding="utf-8")
    )
    runs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for label, directory in SPATIAL.items():
        runs[label] = {seed: load(root / "groups" / directory / f"seed{seed}" / "predictions_val.npz") for seed in SCREEN_SEEDS}
    selected_paths = {int(record["seed"]): Path(record["directory"]) for record in temporal_selection["runs"]}
    runs["S5_learned_4x4_wind"] = {
        seed: load(selected_paths[seed] / "predictions_val.npz") for seed in SCREEN_SEEDS
    }
    summary_rows = []
    for label, seed_runs in runs.items():
        values = [mae(seed_runs[seed]) for seed in SCREEN_SEEDS]
        summary_rows.append({"group": label, "mae_mean_mw": float(np.mean(values)), "mae_sd_mw": float(np.std(values, ddof=1)), "seed_mae_mw": json.dumps(values)})
    pair_rows, horizon_rows = [], []
    decisions: dict[str, Any] = {"data_split": "2023_validation", "test_2024_read": False}
    for pair, left, right in SPATIAL_PAIRS:
        differences, band_max = [], []
        for seed in SCREEN_SEEDS:
            left_run, right_run = runs[left][seed], runs[right][seed]
            np.testing.assert_array_equal(left_run["origin_unix"], right_run["origin_unix"])
            left_error = np.abs(left_run["cluster_prediction_mw"] - left_run["cluster_target_mw"])
            right_error = np.abs(right_run["cluster_prediction_mw"] - right_run["cluster_target_mw"])
            delta = left_error - right_error
            differences.append(float(delta.mean()))
            current_bands = []
            for band, indices in HORIZONS.items():
                value = float(delta[:, indices].mean())
                current_bands.append(value)
                horizon_rows.append({"comparison": pair, "seed": seed, "horizon_band": band, "left_minus_right_mw": value})
            band_max.append(max(current_bands))
        values = np.asarray(differences)
        low, high = interval(values)
        pair_rows.append({"comparison": pair, "left_minus_right_mean_mw": float(values.mean()), "ci95_low_mw": low, "ci95_high_mw": high, "left_wins": int(np.sum(values < 0)), "worst_seed_band_delta_mw": float(max(band_max)), "seed_differences_mw": json.dumps(differences)})
    # Learned 2x2 is S2 (right) against fixed latent S1; learned 4x4 is S4
    # (right) against fixed Gaussian S3.  Negative learned-minus-fixed wins.
    lookup = {row["comparison"]: row for row in pair_rows}
    learned_2x2 = -np.asarray(json.loads(lookup["S1_vs_S2"]["seed_differences_mw"]))
    learned_4x4 = -np.asarray(json.loads(lookup["S3_vs_S4"]["seed_differences_mw"]))
    def learned_worst_band(pair: str) -> float:
        return max(
            -float(np.mean([row["left_minus_right_mw"] for row in horizon_rows if row["comparison"] == pair and row["horizon_band"] == band]))
            for band in HORIZONS
        )
    decisions["S2_expand_to_10"] = bool(
        learned_2x2.mean() <= -1
        and np.sum(learned_2x2 < 0) == 3
        and learned_worst_band("S1_vs_S2") <= 1.0
    )
    decisions["S4_expand_to_10"] = bool(
        learned_4x4.mean() <= -1
        and np.sum(learned_4x4 < 0) == 3
        and learned_worst_band("S3_vs_S4") <= 1.0
    )
    expansion_groups = []
    if decisions["S2_expand_to_10"]:
        expansion_groups.extend(("s1_latent_bilinear", "s2_learned_2x2"))
    if decisions["S4_expand_to_10"]:
        expansion_groups.extend(("s3_fixed_gaussian_4x4", "s4_learned_4x4_nogeom"))
    decisions["formal_expansion_groups"] = list(dict.fromkeys(expansion_groups))
    best = min(float(row["mae_mean_mw"]) for row in summary_rows)
    simplicity = ("S0_raw_bilinear", "S1_latent_bilinear", "S3_fixed_gaussian_4x4", "S2_learned_2x2", "S4_learned_4x4_nogeom", "S5_learned_4x4_wind")
    means = {row["group"]: float(row["mae_mean_mw"]) for row in summary_rows}
    decisions["screening_selection_within_1mw"] = next(label for label in simplicity if means[label] <= best + 1.0)
    write_csv(output / "single_checkpoint_summary.csv", summary_rows)
    write_csv(output / "paired_comparisons.csv", pair_rows)
    write_csv(output / "per_horizon_band.csv", horizon_rows)
    (output / "decision.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "SCREENING_COMPLETE").touch()
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gated cross-resolution follow-ups")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage", choices=("mechanism", "spatial"), required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    analyze_mechanism(project) if args.stage == "mechanism" else analyze_spatial(project)


if __name__ == "__main__":
    main()
