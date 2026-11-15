from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import t as student_t

from scripts.analyze_conditional_forcing_validation import block_bootstrap_matrix


SEEDS = list(range(42, 52))
HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)
SINGLE_TEACHER_MAP = {
    42: 43,
    43: 44,
    44: 45,
    45: 46,
    46: 47,
    47: 48,
    48: 49,
    49: 50,
    50: 51,
    51: 42,
}


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def paired_interval(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    statistic = mean / (std / np.sqrt(len(values))) if std else np.inf
    p_value = float(2.0 * student_t.sf(abs(statistic), len(values) - 1))
    return mean, std, mean - half, mean + half, p_value


def load_group(
    root: Path,
    predictions_name: str = "predictions_val.npz",
) -> dict[str, Any]:
    predictions = []
    targets = None
    origins = None
    seed_maes = []
    for seed in SEEDS:
        run = root / f"seed{seed}"
        if not (run / "COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete run: {run}")
        path = run / predictions_name
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["cluster_prediction_mw"].astype(np.float64)
            target = archive["cluster_target_mw"].astype(np.float64)
            origin = archive["origin_unix"].astype(np.int64)
        if targets is None:
            targets = target
            origins = origin
        else:
            np.testing.assert_allclose(targets, target, rtol=0.0, atol=1e-5)
            np.testing.assert_array_equal(origins, origin)
        predictions.append(prediction)
        seed_maes.append(float(np.abs(prediction - target).mean()))
    assert targets is not None and origins is not None
    prediction_array = np.stack(predictions)
    mean_prediction = prediction_array.mean(axis=0)
    seed_errors = (prediction_array - targets[None]).reshape(len(SEEDS), -1)
    correlations = np.corrcoef(seed_errors)
    off_diagonal = correlations[np.triu_indices(len(SEEDS), k=1)]
    return {
        "root": str(root.resolve()),
        "predictions": prediction_array,
        "target": targets,
        "origins": origins,
        "seed_maes": np.asarray(seed_maes),
        "mean_prediction": mean_prediction,
        "mean_ensemble_mae": float(np.abs(mean_prediction - targets).mean()),
        "mean_pairwise_disagreement": float(
            prediction_array.std(axis=0, ddof=1).mean()
        ),
        "mean_pairwise_signed_error_correlation": float(off_diagonal.mean()),
    }


def summarize(label: str, values: dict[str, Any]) -> dict[str, Any]:
    maes = values["seed_maes"]
    return {
        "group": label,
        "seeds": SEEDS,
        "single_model_mae_mean_mw": float(maes.mean()),
        "single_model_mae_sample_std_mw": float(maes.std(ddof=1)),
        "single_model_mae_by_seed_mw": maes.tolist(),
        "ten_model_ensemble_mae_mw": values["mean_ensemble_mae"],
        "single_minus_ensemble_gap_mw": float(
            maes.mean() - values["mean_ensemble_mae"]
        ),
        "seed_disagreement_mw": values["mean_pairwise_disagreement"],
        "signed_error_correlation": values[
            "mean_pairwise_signed_error_correlation"
        ],
        "samples": int(values["target"].shape[0]),
        "utc_days": int(len(np.unique(values["origins"] // 86400))),
        "root": values["root"],
    }


def compare(
    name: str,
    candidate_name: str,
    reference_name: str,
    groups: dict[str, dict[str, Any]],
    repeats: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = groups[candidate_name]
    reference = groups[reference_name]
    np.testing.assert_array_equal(candidate["origins"], reference["origins"])
    np.testing.assert_allclose(
        candidate["target"], reference["target"], rtol=0.0, atol=1e-5
    )
    seed_delta = candidate["seed_maes"] - reference["seed_maes"]
    mean, std, low, high, p_value = paired_interval(seed_delta)
    target = candidate["target"]
    point_delta = (
        np.abs(candidate["mean_prediction"] - target)
        - np.abs(reference["mean_prediction"] - target)
    )
    point, block_low, block_high, days = block_bootstrap_matrix(
        point_delta.mean(axis=1, keepdims=True),
        candidate["origins"],
        repeats,
        bootstrap_seed,
    )
    row = {
        "comparison": name,
        "candidate": candidate_name,
        "reference": reference_name,
        "seedwise_candidate_minus_reference_mw": mean,
        "seedwise_sample_std_mw": std,
        "seedwise_ci95_low_mw": low,
        "seedwise_ci95_high_mw": high,
        "candidate_wins": int(np.sum(seed_delta < 0.0)),
        "seedwise_p_two_sided": p_value,
        "ensemble_candidate_minus_reference_mw": float(point[0]),
        "ensemble_ci95_low_mw": float(block_low[0]),
        "ensemble_ci95_high_mw": float(block_high[0]),
        "utc_day_blocks": days,
        "bootstrap_repeats": repeats,
    }
    bands = []
    for index, (band, lead_slice) in enumerate(HORIZON_BANDS):
        band_point, band_low, band_high, _ = block_bootstrap_matrix(
            point_delta[:, lead_slice].mean(axis=1, keepdims=True),
            candidate["origins"],
            repeats,
            bootstrap_seed + index + 1,
        )
        candidate_single = np.abs(
            candidate["predictions"][:, :, lead_slice]
            - target[None, :, lead_slice]
        ).mean(axis=(1, 2))
        reference_single = np.abs(
            reference["predictions"][:, :, lead_slice]
            - target[None, :, lead_slice]
        ).mean(axis=(1, 2))
        band_seed = candidate_single - reference_single
        band_mean, _, band_seed_low, band_seed_high, _ = paired_interval(band_seed)
        bands.append(
            {
                "comparison": name,
                "horizon_band": band,
                "seedwise_candidate_minus_reference_mw": band_mean,
                "seedwise_ci95_low_mw": band_seed_low,
                "seedwise_ci95_high_mw": band_seed_high,
                "ensemble_candidate_minus_reference_mw": float(band_point[0]),
                "ensemble_ci95_low_mw": float(band_low[0]),
                "ensemble_ci95_high_mw": float(band_high[0]),
            }
        )
    return row, bands


def apply_holm(rows: list[dict[str, Any]], comparison_names: set[str]) -> None:
    indices = [
        index for index, row in enumerate(rows) if row["comparison"] in comparison_names
    ]
    p_values = np.asarray([rows[index]["seedwise_p_two_sided"] for index in indices])
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, local_index in enumerate(order):
        current = min(1.0, (len(indices) - rank) * p_values[local_index])
        running = max(running, current)
        adjusted[local_index] = running
    for local_index, row_index in enumerate(indices):
        rows[row_index]["holm_p_primary_family"] = float(adjusted[local_index])


def audit_teacher_provenance(project: Path, experiment_root: Path) -> dict[str, Any]:
    recent_single = experiment_root / "groups/recent_single_teacher_kd050"
    full_kd = experiment_root / "groups/full_history_ensemble_kd050"
    single_rows = []
    for student_seed, expected_teacher_seed in SINGLE_TEACHER_MAP.items():
        checkpoint = torch.load(
            recent_single / f"seed{student_seed}/best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        metadata = checkpoint.get("distillation_teacher")
        if metadata is None:
            raise ValueError(f"Missing teacher provenance for student seed {student_seed}")
        if metadata["kind"] != "single_checkpoint":
            raise ValueError(f"Unexpected single-teacher kind: {metadata}")
        if metadata["seeds"] != [expected_teacher_seed]:
            raise ValueError(
                f"Student {student_seed}: expected teacher {expected_teacher_seed}, "
                f"got {metadata['seeds']}"
            )
        single_rows.append(
            {
                "student_seed": student_seed,
                "teacher_seed": expected_teacher_seed,
                "archive": metadata["archive"],
            }
        )
    full_rows = []
    for seed in SEEDS:
        checkpoint = torch.load(
            full_kd / f"seed{seed}/best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        metadata = checkpoint.get("distillation_teacher")
        if metadata is None or metadata["kind"] != "ensemble_mean":
            raise ValueError(f"Unexpected full-history teacher metadata: {metadata}")
        if metadata["seeds"] != SEEDS:
            raise ValueError(f"Unexpected full-history teacher seeds: {metadata}")
        full_rows.append({"student_seed": seed, **metadata})

    full_archive = experiment_root / "teacher/train_teacher_full_history_ensemble.npz"
    with np.load(full_archive, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["teacher_seeds"], np.asarray(SEEDS))
        full_kind = str(archive["teacher_kind"].item())
    if full_kind != "ensemble_mean":
        raise ValueError(f"Unexpected full archive kind: {full_kind}")

    for teacher_seed in SEEDS:
        archive_path = (
            experiment_root
            / f"teacher/single_recent/train_teacher_seed{teacher_seed}.npz"
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            if str(archive["teacher_kind"].item()) != "single_checkpoint":
                raise ValueError(f"Unexpected teacher kind: {archive_path}")
            np.testing.assert_array_equal(
                archive["teacher_seeds"], np.asarray([teacher_seed])
            )
            if np.any(archive["teacher_station_factor_std"] != 0.0):
                raise ValueError(f"Single station dispersion is nonzero: {archive_path}")
            if np.any(archive["teacher_cluster_factor_std"] != 0.0):
                raise ValueError(f"Single cluster dispersion is nonzero: {archive_path}")
    return {
        "single_teacher_derangement": single_rows,
        "full_history_teacher": full_rows,
        "student_teacher_seed_matches": 0,
        "project": str(project),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze minimal consensus controls")
    parser.add_argument(
        "--experiment-root", default="artifacts/consensus_controls_v1"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    experiment_root = (project / args.experiment_root).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else experiment_root / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)

    specs = {
        "recent_normal": (
            project
            / "artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h",
            "predictions_val_normal.npz",
        ),
        "recent_single_teacher_kd050": (
            experiment_root / "groups/recent_single_teacher_kd050",
            "predictions_val.npz",
        ),
        "recent_ensemble_teacher_kd050": (
            project / "artifacts/ensemble_stabilization_v1/groups/kd_050",
            "predictions_val.npz",
        ),
        "full_history_normal": (
            project / "artifacts/conditional_forcing_validation_v1/groups/anchor_020",
            "predictions_val.npz",
        ),
        "full_history_ensemble_teacher_kd050": (
            experiment_root / "groups/full_history_ensemble_kd050",
            "predictions_val.npz",
        ),
    }
    groups = {
        label: load_group(root, predictions_name)
        for label, (root, predictions_name) in specs.items()
    }
    summaries = [summarize(label, values) for label, values in groups.items()]

    comparison_specs = (
        (
            "recent_single_teacher_vs_normal",
            "recent_single_teacher_kd050",
            "recent_normal",
        ),
        (
            "recent_ensemble_teacher_vs_normal",
            "recent_ensemble_teacher_kd050",
            "recent_normal",
        ),
        (
            "recent_ensemble_teacher_vs_single_teacher",
            "recent_ensemble_teacher_kd050",
            "recent_single_teacher_kd050",
        ),
        (
            "full_history_kd_vs_full_history_normal",
            "full_history_ensemble_teacher_kd050",
            "full_history_normal",
        ),
        (
            "full_history_kd_vs_recent_kd",
            "full_history_ensemble_teacher_kd050",
            "recent_ensemble_teacher_kd050",
        ),
        ("full_history_normal_vs_recent_normal", "full_history_normal", "recent_normal"),
    )
    comparisons = []
    horizon_rows = []
    for index, (name, candidate, reference) in enumerate(comparison_specs):
        row, bands = compare(
            name,
            candidate,
            reference,
            groups,
            args.bootstrap_repeats,
            83000 + 10 * index,
        )
        comparisons.append(row)
        horizon_rows.extend(bands)
    primary_names = {
        "recent_ensemble_teacher_vs_single_teacher",
        "full_history_kd_vs_recent_kd",
    }
    apply_holm(comparisons, primary_names)
    comparison_by_name = {row["comparison"]: row for row in comparisons}

    consensus_row = comparison_by_name[
        "recent_ensemble_teacher_vs_single_teacher"
    ]
    history_row = comparison_by_name["full_history_kd_vs_recent_kd"]
    consensus_supported = bool(
        consensus_row["seedwise_ci95_high_mw"] < 0.0
        and consensus_row["ensemble_ci95_high_mw"] < 0.0
        and consensus_row["holm_p_primary_family"] < 0.05
    )
    full_history_upgrade = bool(
        history_row["seedwise_ci95_high_mw"] < 0.0
        and history_row["ensemble_ci95_high_mw"] < 0.0
        and history_row["holm_p_primary_family"] < 0.05
    )

    provenance = audit_teacher_provenance(project, experiment_root)
    pair_rows = []
    recent_normal = groups["recent_normal"]["seed_maes"]
    recent_single = groups["recent_single_teacher_kd050"]["seed_maes"]
    recent_ensemble = groups["recent_ensemble_teacher_kd050"]["seed_maes"]
    for index, student_seed in enumerate(SEEDS):
        teacher_seed = SINGLE_TEACHER_MAP[student_seed]
        teacher_index = SEEDS.index(teacher_seed)
        pair_rows.append(
            {
                "student_seed": student_seed,
                "teacher_seed": teacher_seed,
                "normal_same_seed_mae_mw": float(recent_normal[index]),
                "single_teacher_validation_mae_mw": float(recent_normal[teacher_index]),
                "single_teacher_kd_student_mae_mw": float(recent_single[index]),
                "ensemble_teacher_kd_student_mae_mw": float(recent_ensemble[index]),
            }
        )

    payload = {
        "protocol": {
            "date": "2026-08-30",
            "selection_split": "2023 validation",
            "test_read": False,
            "seeds": SEEDS,
            "single_teacher_map": SINGLE_TEACHER_MAP,
            "distillation_weight": 0.5,
            "teacher_and_student_same_architecture": True,
            "primary_comparisons": sorted(primary_names),
            "primary_seedwise_multiplicity": "Holm correction over two comparisons",
            "ensemble_inference": "10000 sample-weighted paired UTC-day block bootstrap",
            "full_history_upgrade_rule": (
                "Upgrade only if both paired-seed and day-block ensemble 95% CIs "
                "favor full-history KD, with Holm-adjusted seedwise p<0.05."
            ),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "horizon_comparisons": horizon_rows,
        "teacher_provenance": provenance,
        "decisions": {
            "ensemble_consensus_advantage_supported": consensus_supported,
            "upgrade_to_full_history_kd": full_history_upgrade,
            "selected_state_mode": (
                "full_history" if full_history_upgrade else "cluster_recent_1h"
            ),
        },
    }
    write_json(output / "analysis.json", payload)
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "horizon_comparisons.csv", horizon_rows)
    write_csv(output / "single_teacher_pairs.csv", pair_rows)
    print(json.dumps(payload["decisions"], indent=2))
    print(f"[done] analysis={output}")


if __name__ == "__main__":
    main()
