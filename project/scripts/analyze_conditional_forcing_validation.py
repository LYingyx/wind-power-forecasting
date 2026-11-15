from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
from scipy.stats import t as student_t
import torch

try:
    from scripts.analyze_incremental_mechanism_v2 import (
        compare_groups,
        group_predictions,
        load_json,
        paired_utc_day_block_bootstrap,
        summarize_group,
    )
except ModuleNotFoundError:
    from analyze_incremental_mechanism_v2 import (  # type: ignore[no-redef]
        compare_groups,
        group_predictions,
        load_json,
        paired_utc_day_block_bootstrap,
        summarize_group,
    )


SEEDS = list(range(42, 52))
GROUPS = (
    "history_baseline",
    "reference_direct_d64",
    "factor_context_only",
    "factor_history_only",
    "factor_weather_only",
    "factor_history_weather",
    "initialization_random",
    "anchor_000",
    "anchor_005",
    "anchor_010",
    "anchor_020",
)
PERMUTATION_SEEDS = (7301, 7302, 7303, 7304, 7305)
DIVERSITY_GROUPS = (
    "reference_direct_d64",
    "factor_history_weather",
    "initialization_random",
    "anchor_000",
    "anchor_005",
    "anchor_010",
    "anchor_020",
)


def t_interval(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("A t interval requires at least two scalar values")
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1))
    half_width = float(
        student_t.ppf(0.975, df=len(array) - 1)
        * sample_std
        / np.sqrt(len(array))
    )
    return mean, sample_std, mean - half_width, mean + half_width


def seed_maes(group: Path, seeds: list[int]) -> np.ndarray:
    return np.asarray(
        [
            float(load_json(group / f"seed{seed}" / "metrics_val.json")["cluster_mae_mw"])
            for seed in seeds
        ],
        dtype=np.float64,
    )


def ensemble_error(group: Path, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    _, prediction = group_predictions(group, seeds)
    error = np.abs(
        prediction["cluster_prediction_mw"].astype(np.float64)
        - prediction["cluster_target_mw"].astype(np.float64)
    )
    return error, np.asarray(prediction["origin_unix"], dtype=np.int64)


def linear_seed_contrast(
    groups_root: Path,
    terms: list[tuple[str, float]],
    seeds: list[int],
) -> dict[str, Any]:
    values = np.zeros(len(seeds), dtype=np.float64)
    for group, coefficient in terms:
        values += coefficient * seed_maes(groups_root / group, seeds)
    mean, sample_std, low, high = t_interval(values)
    return {
        "terms": [{"group": group, "coefficient": coefficient} for group, coefficient in terms],
        "seed_values_mw": values.tolist(),
        "seed_mean_mw": mean,
        "seed_sample_std_mw": sample_std,
        "seed_ci95_low_mw": low,
        "seed_ci95_high_mw": high,
        "positive_seeds": int(np.sum(values > 0.0)),
    }


def block_bootstrap_matrix(
    delta: np.ndarray,
    origins: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    delta = np.asarray(delta, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    if delta.ndim != 2 or delta.shape[0] != len(origins):
        raise ValueError("Lead-wise deltas must have shape [origins, leads]")
    block_id = origins // 86400
    blocks = np.unique(block_id)
    block_sums = np.stack(
        [delta[block_id == block].sum(axis=0) for block in blocks], axis=0
    )
    block_counts = np.asarray(
        [(block_id == block).sum() for block in blocks], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, delta.shape[1]), dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = block_sums[selected].sum(axis=0) / block_counts[selected].sum()
    return (
        delta.mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
        int(len(blocks)),
    )


def factorial_analysis(
    groups_root: Path,
    seeds: list[int],
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = {
        "context": "factor_context_only",
        "history": "factor_history_only",
        "weather": "factor_weather_only",
        "full": "factor_history_weather",
    }
    errors: dict[str, np.ndarray] = {}
    origins = None
    for label, group in names.items():
        errors[label], group_origins = ensemble_error(groups_root / group, seeds)
        if origins is None:
            origins = group_origins
        else:
            np.testing.assert_array_equal(origins, group_origins)
    assert origins is not None

    weather_given_history = errors["history"] - errors["full"]
    weather_without_history = errors["context"] - errors["weather"]
    history_given_weather = errors["weather"] - errors["full"]
    history_without_weather = errors["context"] - errors["history"]
    interaction = weather_given_history - weather_without_history
    matrices = {
        "weather_benefit_given_history": weather_given_history,
        "weather_benefit_without_history": weather_without_history,
        "history_benefit_given_weather": history_given_weather,
        "history_benefit_without_weather": history_without_weather,
        "interaction_benefit": interaction,
    }
    horizon_statistics = {}
    for index, (label, matrix) in enumerate(matrices.items()):
        mean, low, high, days = block_bootstrap_matrix(
            matrix, origins, repeats, 4100 + index
        )
        horizon_statistics[label] = {
            "mean": mean,
            "low": low,
            "high": high,
            "days": days,
        }

    seed_weather_given_history = linear_seed_contrast(
        groups_root,
        [(names["history"], 1.0), (names["full"], -1.0)],
        seeds,
    )
    seed_weather_without_history = linear_seed_contrast(
        groups_root,
        [(names["context"], 1.0), (names["weather"], -1.0)],
        seeds,
    )
    seed_interaction = linear_seed_contrast(
        groups_root,
        [
            (names["history"], 1.0),
            (names["full"], -1.0),
            (names["context"], -1.0),
            (names["weather"], 1.0),
        ],
        seeds,
    )
    interaction_origin = interaction.mean(axis=1)
    interaction_delta, interaction_low, interaction_high, day_count = (
        paired_utc_day_block_bootstrap(
            interaction_origin,
            np.zeros_like(interaction_origin),
            origins,
            repeats,
            4200,
        )
    )
    payload = {
        "definition": (
            "positive benefit means the named input reduces MAE; interaction is "
            "weather benefit with H minus weather benefit without H"
        ),
        "seed_weather_benefit_given_history": seed_weather_given_history,
        "seed_weather_benefit_without_history": seed_weather_without_history,
        "seed_interaction_benefit": seed_interaction,
        "ensemble_interaction_benefit_mw": interaction_delta,
        "ensemble_interaction_ci95_low_mw": interaction_low,
        "ensemble_interaction_ci95_high_mw": interaction_high,
        "utc_day_blocks": day_count,
    }
    rows = []
    for lead in range(errors["full"].shape[1]):
        row: dict[str, Any] = {
            "lead": lead + 1,
            "lead_hours": (lead + 1) / 4.0,
            "context_ensemble_mae_mw": float(errors["context"][:, lead].mean()),
            "history_ensemble_mae_mw": float(errors["history"][:, lead].mean()),
            "weather_ensemble_mae_mw": float(errors["weather"][:, lead].mean()),
            "full_ensemble_mae_mw": float(errors["full"][:, lead].mean()),
        }
        for label, stats in horizon_statistics.items():
            row[f"{label}_mw"] = float(stats["mean"][lead])
            row[f"{label}_ci95_low_mw"] = float(stats["low"][lead])
            row[f"{label}_ci95_high_mw"] = float(stats["high"][lead])
        rows.append(row)
    return payload, rows


def diversity_analysis(groups_root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for group in DIVERSITY_GROUPS:
        predictions, ensemble = group_predictions(groups_root / group, seeds)
        target = ensemble["cluster_target_mw"].astype(np.float64)
        residuals = np.stack(
            [
                (
                    prediction["cluster_prediction_mw"].astype(np.float64)
                    - target
                ).reshape(-1)
                for prediction in predictions
            ],
            axis=0,
        )
        correlation = np.corrcoef(residuals)
        upper = correlation[np.triu_indices(len(seeds), k=1)]
        disagreements = []
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                disagreements.append(
                    float(
                        np.abs(
                            predictions[left]["cluster_prediction_mw"].astype(np.float64)
                            - predictions[right]["cluster_prediction_mw"].astype(np.float64)
                        ).mean()
                    )
                )
        summary = summarize_group(groups_root / group, seeds)
        rows.append(
            {
                "group": group,
                "mean_pairwise_error_correlation": float(upper.mean()),
                "sample_std_pairwise_error_correlation": float(upper.std(ddof=1)),
                "min_pairwise_error_correlation": float(upper.min()),
                "max_pairwise_error_correlation": float(upper.max()),
                "mean_pairwise_prediction_disagreement_mw": statistics.mean(disagreements),
                "ensemble_gain_vs_mean_single_mw": summary[
                    "ensemble_gain_vs_mean_single_mw"
                ],
            }
        )
    return rows


def residual_day_sufficient_statistics(
    group: Path, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions, ensemble = group_predictions(group, seeds)
    target = ensemble["cluster_target_mw"].astype(np.float64)
    residual = np.stack(
        [
            prediction["cluster_prediction_mw"].astype(np.float64) - target
            for prediction in predictions
        ],
        axis=0,
    )
    origins = np.asarray(ensemble["origin_unix"], dtype=np.int64)
    block_id = origins // 86400
    blocks = np.unique(block_id)
    counts = []
    sums = []
    square_sums = []
    cross_sums = []
    for block in blocks:
        block_residual = residual[:, block_id == block].reshape(len(seeds), -1)
        counts.append(block_residual.shape[1])
        sums.append(block_residual.sum(axis=1))
        square_sums.append(np.square(block_residual).sum(axis=1))
        cross_sums.append(block_residual @ block_residual.T)
    return (
        np.asarray(counts, dtype=np.int64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(square_sums, dtype=np.float64),
        np.asarray(cross_sums, dtype=np.float64),
        origins,
    )


def bootstrap_mean_pairwise_correlation(
    counts: np.ndarray,
    sums: np.ndarray,
    square_sums: np.ndarray,
    cross_sums: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> np.ndarray:
    repeats = bootstrap_counts.shape[0]
    seed_count = sums.shape[1]
    upper = np.triu_indices(seed_count, k=1)
    result = np.empty(repeats, dtype=np.float64)
    chunk_size = 250
    flat_cross = cross_sums.reshape(len(counts), -1)
    for start in range(0, repeats, chunk_size):
        selected = bootstrap_counts[start : start + chunk_size]
        total_count = selected @ counts
        total_sum = selected @ sums
        total_square = selected @ square_sums
        total_cross = (selected @ flat_cross).reshape(-1, seed_count, seed_count)
        centered_cross = total_cross - (
            total_sum[:, :, None] * total_sum[:, None, :] / total_count[:, None, None]
        )
        centered_square = total_square - np.square(total_sum) / total_count[:, None]
        denominator = np.sqrt(
            centered_square[:, :, None] * centered_square[:, None, :]
        )
        correlation = centered_cross / np.maximum(denominator, 1e-12)
        result[start : start + len(selected)] = correlation[:, upper[0], upper[1]].mean(
            axis=1
        )
    return result


def diversity_bootstrap_analysis(
    groups_root: Path,
    seeds: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    statistics_by_group = {
        group: residual_day_sufficient_statistics(groups_root / group, seeds)
        for group in DIVERSITY_GROUPS
    }
    block_counts = [len(values[0]) for values in statistics_by_group.values()]
    if len(set(block_counts)) != 1:
        raise ValueError("Diversity groups have different UTC-day block counts")
    day_count = block_counts[0]
    rng = np.random.default_rng(7100)
    bootstrap_counts = rng.multinomial(
        day_count,
        np.full(day_count, 1.0 / day_count),
        size=repeats,
    )
    bootstrap = {}
    for group, values in statistics_by_group.items():
        bootstrap[group] = bootstrap_mean_pairwise_correlation(
            values[0], values[1], values[2], values[3], bootstrap_counts
        )
    pairs = (
        ("factor_history_weather", "reference_direct_d64"),
        ("initialization_random", "factor_history_weather"),
        ("initialization_random", "reference_direct_d64"),
        ("factor_history_weather", "anchor_000"),
        ("anchor_005", "anchor_000"),
        ("anchor_010", "anchor_000"),
        ("anchor_020", "anchor_000"),
    )
    rows = []
    for candidate, reference in pairs:
        difference = bootstrap[candidate] - bootstrap[reference]
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "candidate_minus_reference_mean_pairwise_error_correlation": float(
                    difference.mean()
                ),
                "ci95_low": float(np.quantile(difference, 0.025)),
                "ci95_high": float(np.quantile(difference, 0.975)),
                "utc_day_blocks": day_count,
                "bootstrap_repeats": repeats,
            }
        )
    return rows


def component_prediction_path(root: Path, group: str, seed: int) -> Path:
    if group in {"factor_history_weather", "anchor_000"}:
        return (
            root
            / "component_evaluations"
            / group
            / f"seed{seed}"
            / "predictions_val.npz"
        )
    return root / "groups" / group / f"seed{seed}" / "predictions_val.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def anchor_diagnostics(root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    groups_root = root / "groups"
    rows = []
    for group, constraint_type, weight in (
        ("anchor_000", "none", 0.0),
        ("anchor_005", "pretrained_output_anchor", 0.05),
        ("anchor_010", "pretrained_output_anchor", 0.1),
        ("anchor_020", "pretrained_output_anchor", 0.2),
        ("factor_history_weather", "target_auxiliary", 0.5),
    ):
        drift = []
        correction = []
        baseline_target_mae = []
        anchor_replay_difference = []
        anchor_state_exact = []
        for seed in seeds:
            component = load_npz(component_prediction_path(root, group, seed))
            prior = load_npz(
                groups_root
                / "history_baseline"
                / f"seed{seed}"
                / "predictions_val.npz"
            )
            np.testing.assert_array_equal(
                component["origin_unix"], prior["origin_unix"]
            )
            current_baseline = component["baseline_cluster_prediction_mw"].astype(
                np.float64
            )
            prior_baseline = prior["cluster_prediction_mw"].astype(np.float64)
            target = component["cluster_target_mw"].astype(np.float64)
            drift.append(float(np.abs(current_baseline - prior_baseline).mean()))
            correction.append(
                float(
                    np.abs(
                        component["realized_cluster_correction_mw"].astype(np.float64)
                    ).mean()
                )
            )
            baseline_target_mae.append(float(np.abs(current_baseline - target).mean()))
            if "anchor_baseline_cluster_prediction_mw" in component:
                anchor_replay_difference.append(
                    float(
                        np.abs(
                            component[
                                "anchor_baseline_cluster_prediction_mw"
                            ].astype(np.float64)
                            - prior_baseline
                        ).mean()
                    )
                )
                candidate_checkpoint = torch.load(
                    groups_root / group / f"seed{seed}" / "best_model.pt",
                    map_location="cpu",
                    weights_only=False,
                )["model_state"]
                prior_checkpoint = torch.load(
                    groups_root
                    / "history_baseline"
                    / f"seed{seed}"
                    / "best_model.pt",
                    map_location="cpu",
                    weights_only=False,
                )["model_state"]
                exact = True
                for state_index, (_, value) in enumerate(
                    prior_checkpoint.items()
                ):
                    anchor_key = f"_baseline_anchor_state_{state_index:04d}"
                    if anchor_key not in candidate_checkpoint or not torch.equal(
                        candidate_checkpoint[anchor_key], value
                    ):
                        exact = False
                        break
                anchor_state_exact.append(exact)
        summary = summarize_group(groups_root / group, seeds)
        drift_mean, drift_std, drift_low, drift_high = t_interval(drift)
        rows.append(
            {
                "group": group,
                "constraint_type": constraint_type,
                "constraint_weight": weight,
                "single_model_mae_mean_mw": summary["cluster_mae_mw_mean"],
                "single_model_mae_sample_std_mw": summary[
                    "cluster_mae_mw_sample_std"
                ],
                "ensemble_mae_mw": summary["ensemble_cluster_mae_mw"],
                "baseline_drift_from_pretrained_mean_mw": drift_mean,
                "baseline_drift_sample_std_mw": drift_std,
                "baseline_drift_ci95_low_mw": drift_low,
                "baseline_drift_ci95_high_mw": drift_high,
                "baseline_drift_seed_values_mw": drift,
                "realized_correction_abs_mean_mw": statistics.mean(correction),
                "realized_correction_abs_seed_values_mw": correction,
                "current_baseline_target_mae_mean_mw": statistics.mean(
                    baseline_target_mae
                ),
                "current_baseline_target_mae_seed_values_mw": baseline_target_mae,
                "anchor_checkpoint_state_exact": (
                    all(anchor_state_exact) if anchor_state_exact else None
                ),
                "anchor_replay_batching_abs_difference_mean_mw": (
                    statistics.mean(anchor_replay_difference)
                    if anchor_replay_difference
                    else None
                ),
            }
        )
    return rows


def anchor_drift_comparisons(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_group = {row["group"]: row for row in rows}
    pairs = (
        ("anchor_005", "anchor_000"),
        ("anchor_010", "anchor_000"),
        ("anchor_020", "anchor_000"),
        ("factor_history_weather", "anchor_000"),
        ("anchor_010", "factor_history_weather"),
        ("anchor_020", "factor_history_weather"),
    )
    comparisons = []
    for candidate, reference in pairs:
        candidate_values = np.asarray(
            by_group[candidate]["baseline_drift_seed_values_mw"],
            dtype=np.float64,
        )
        reference_values = np.asarray(
            by_group[reference]["baseline_drift_seed_values_mw"],
            dtype=np.float64,
        )
        difference = candidate_values - reference_values
        mean, sample_std, low, high = t_interval(difference)
        comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "candidate_minus_reference_drift_mean_mw": mean,
                "difference_sample_std_mw": sample_std,
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "candidate_lower_drift_seeds": int(np.sum(difference < 0.0)),
                "paired_differences_mw": difference.tolist(),
            }
        )
    return comparisons


def weather_shuffle_analysis(
    root: Path,
    seeds: list[int],
    permutation_seeds: list[int],
    repeats: int,
) -> tuple[dict[str, Any], np.ndarray]:
    original_individual, original_ensemble = group_predictions(
        root / "groups" / "factor_history_weather", seeds
    )
    target = original_ensemble["cluster_target_mw"].astype(np.float64)
    origins = original_ensemble["origin_unix"]
    original_seed_mae = np.asarray(
        [
            np.abs(
                prediction["cluster_prediction_mw"].astype(np.float64) - target
            ).mean()
            for prediction in original_individual
        ]
    )
    original_ensemble_error = np.abs(
        original_ensemble["cluster_prediction_mw"].astype(np.float64) - target
    )
    shuffled_seed_mae = np.empty((len(permutation_seeds), len(seeds)))
    shuffled_ensemble_error = []
    metadata = []
    for permutation_index, permutation_seed in enumerate(permutation_seeds):
        archives = []
        for seed_index, seed in enumerate(seeds):
            directory = root / "weather_shuffle" / f"perm{permutation_seed}" / f"seed{seed}"
            archive = load_npz(directory / "predictions_val.npz")
            np.testing.assert_array_equal(archive["origin_unix"], origins)
            np.testing.assert_allclose(archive["cluster_target_mw"], target)
            shuffled_seed_mae[permutation_index, seed_index] = np.abs(
                archive["cluster_prediction_mw"].astype(np.float64) - target
            ).mean()
            archives.append(archive["cluster_prediction_mw"].astype(np.float64))
            permutation_metadata = load_json(directory / "metrics_val.json")[
                "weather_permutation"
            ]
            if seed_index == 0:
                metadata.append(permutation_metadata)
            else:
                if (
                    permutation_metadata["donor_index_sha256"]
                    != metadata[-1]["donor_index_sha256"]
                ):
                    raise ValueError("Model seeds did not use an identical weather permutation")
        shuffled_prediction = np.mean(archives, axis=0)
        shuffled_ensemble_error.append(np.abs(shuffled_prediction - target))
    shuffled_ensemble_error_array = np.stack(shuffled_ensemble_error, axis=0)
    seed_difference = shuffled_seed_mae.mean(axis=0) - original_seed_mae
    seed_mean, seed_std, seed_low, seed_high = t_interval(seed_difference)
    ensemble_mae = shuffled_ensemble_error_array.mean(axis=(1, 2))
    original_ensemble_mae = float(original_ensemble_error.mean())
    expected_shuffle_error = shuffled_ensemble_error_array.mean(axis=0)
    per_origin_shuffle = expected_shuffle_error.mean(axis=1)
    per_origin_original = original_ensemble_error.mean(axis=1)
    delta, low, high, days = paired_utc_day_block_bootstrap(
        per_origin_shuffle,
        per_origin_original,
        origins,
        repeats,
        5100,
    )
    return (
        {
            "permutation_seeds": permutation_seeds,
            "permutation_count": len(permutation_seeds),
            "metadata": metadata,
            "seedwise_shuffled_minus_original_mean_mw": seed_mean,
            "seedwise_difference_sample_std_mw": seed_std,
            "seedwise_ci95_low_mw": seed_low,
            "seedwise_ci95_high_mw": seed_high,
            "seedwise_original_wins": int(np.sum(seed_difference > 0.0)),
            "seedwise_differences_mw": seed_difference.tolist(),
            "original_ensemble_mae_mw": original_ensemble_mae,
            "shuffled_ensemble_mae_mean_mw": float(ensemble_mae.mean()),
            "shuffled_ensemble_mae_sample_std_mw": float(ensemble_mae.std(ddof=1)),
            "shuffled_ensemble_mae_min_mw": float(ensemble_mae.min()),
            "shuffled_ensemble_mae_max_mw": float(ensemble_mae.max()),
            "expected_shuffle_ensemble_minus_original_mw": delta,
            "ensemble_ci95_low_mw": low,
            "ensemble_ci95_high_mw": high,
            "utc_day_blocks": days,
            "bootstrap_repeats": repeats,
        },
        expected_shuffle_error - original_ensemble_error,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze state-dependent future-forcing validation experiments"
    )
    parser.add_argument(
        "--experiment-root", default="artifacts/conditional_forcing_validation_v1"
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument(
        "--permutation-seeds",
        default=",".join(str(seed) for seed in PERMUTATION_SEEDS),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value]
    permutation_seeds = [
        int(value) for value in args.permutation_seeds.split(",") if value
    ]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Formal analysis requires exactly ten unique model seeds")
    root = Path(args.experiment_root).resolve()
    if not (root / "ALL_COMPLETE").is_file():
        raise FileNotFoundError(f"Experiment is not complete: {root / 'ALL_COMPLETE'}")
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    groups_root = root / "groups"
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in GROUPS
        for seed in seeds
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed group seeds:\n" + "\n".join(missing))

    summaries = [summarize_group(groups_root / group, seeds) for group in GROUPS]
    comparison_pairs = (
        ("factor_history_weather", "factor_history_only"),
        ("factor_history_weather", "factor_weather_only"),
        ("factor_history_only", "factor_context_only"),
        ("factor_weather_only", "factor_context_only"),
        ("factor_history_weather", "factor_context_only"),
        ("factor_history_weather", "reference_direct_d64"),
        ("factor_context_only", "history_baseline"),
        ("factor_history_only", "history_baseline"),
        ("anchor_000", "factor_history_weather"),
        ("anchor_005", "anchor_000"),
        ("anchor_010", "anchor_000"),
        ("anchor_020", "anchor_000"),
        ("anchor_005", "factor_history_weather"),
        ("anchor_010", "factor_history_weather"),
        ("anchor_020", "factor_history_weather"),
        ("initialization_random", "factor_history_weather"),
        ("initialization_random", "reference_direct_d64"),
    )
    comparisons = [
        compare_groups(
            groups_root / candidate,
            groups_root / reference,
            seeds,
            args.bootstrap_repeats,
            6100 + index,
        )
        for index, (candidate, reference) in enumerate(comparison_pairs)
    ]
    factorial, horizon_rows = factorial_analysis(
        groups_root, seeds, args.bootstrap_repeats
    )
    diversity = diversity_analysis(groups_root, seeds)
    diversity_bootstrap = diversity_bootstrap_analysis(
        groups_root, seeds, args.bootstrap_repeats
    )
    anchor = anchor_diagnostics(root, seeds)
    anchor_drift = anchor_drift_comparisons(anchor)
    weather_shuffle, shuffle_horizon_delta = weather_shuffle_analysis(
        root, seeds, permutation_seeds, args.bootstrap_repeats
    )
    if len(horizon_rows) != shuffle_horizon_delta.shape[1]:
        raise ValueError("Shuffle and factorial horizons differ")
    shuffle_mean, shuffle_low, shuffle_high, _ = block_bootstrap_matrix(
        shuffle_horizon_delta,
        group_predictions(groups_root / "factor_history_weather", seeds)[1][
            "origin_unix"
        ],
        args.bootstrap_repeats,
        6200,
    )
    for index, row in enumerate(horizon_rows):
        row["weather_shuffle_penalty_mw"] = float(shuffle_mean[index])
        row["weather_shuffle_penalty_ci95_low_mw"] = float(shuffle_low[index])
        row["weather_shuffle_penalty_ci95_high_mw"] = float(shuffle_high[index])

    payload = {
        "protocol": {
            "model_seeds": seeds,
            "model_seed_count": len(seeds),
            "weather_permutation_seeds": permutation_seeds,
            "split": "2023 validation",
            "test_split_used": False,
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "factorial": factorial,
        "weather_shuffle": weather_shuffle,
        "diversity": diversity,
        "diversity_bootstrap_comparisons": diversity_bootstrap,
        "anchor_diagnostics": anchor,
        "anchor_drift_comparisons": anchor_drift,
    }
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "horizon.csv", horizon_rows)
    write_csv(output / "diversity.csv", diversity)
    write_csv(
        output / "diversity_bootstrap_comparisons.csv", diversity_bootstrap
    )
    write_csv(output / "anchor_diagnostics.csv", anchor)
    write_csv(output / "anchor_drift_comparisons.csv", anchor_drift)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "groups": len(summaries),
                "comparisons": len(comparisons),
                "model_seeds": len(seeds),
                "weather_permutations": len(permutation_seeds),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
