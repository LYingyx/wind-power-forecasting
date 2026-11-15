from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.fft import dct, idct
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from scripts.analyze_paper_results import block_bootstrap_difference
except ModuleNotFoundError:
    from analyze_paper_results import block_bootstrap_difference  # type: ignore[no-redef]


SECONDS_PER_DAY = 86_400


def chronological_day_split(
    origin_unix: np.ndarray,
    fit_fraction: float = 0.6,
    selection_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Split whole UTC days so overlapping forecast windows never cross partitions."""
    days = np.floor_divide(np.asarray(origin_unix, dtype=np.int64), SECONDS_PER_DAY)
    unique_days = np.unique(days)
    if len(unique_days) < 10:
        raise ValueError("At least ten UTC days are required for chronological diagnostics")
    fit_end = max(1, int(np.floor(len(unique_days) * fit_fraction)))
    selection_end = max(fit_end + 1, int(np.floor(len(unique_days) * (fit_fraction + selection_fraction))))
    if selection_end >= len(unique_days):
        raise ValueError("Chronological split leaves no confirmation days")
    groups = {
        "fit": unique_days[:fit_end],
        "selection": unique_days[fit_end:selection_end],
        "confirmation": unique_days[selection_end:],
    }
    return {name: np.flatnonzero(np.isin(days, values)) for name, values in groups.items()}


def align_history_ensemble(
    prediction_paths: Iterable[Path], origin_unix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expected = np.asarray(origin_unix, dtype=np.int64)
    station_predictions: list[np.ndarray] = []
    target: np.ndarray | None = None
    for path in prediction_paths:
        with np.load(path) as archive:
            required = {"origin_unix", "station_prediction_mw", "station_target_mw"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            origins = np.asarray(archive["origin_unix"], dtype=np.int64)
            if len(np.unique(origins)) != len(origins):
                raise ValueError(f"Duplicate origins in {path}")
            order = np.argsort(origins)
            origins = origins[order]
            positions = np.searchsorted(origins, expected)
            if np.any(positions >= len(origins)) or not np.array_equal(origins[positions], expected):
                raise ValueError(f"Prediction origins do not match the cache: {path}")
            prediction = np.asarray(archive["station_prediction_mw"][order][positions], dtype=np.float32)
            current_target = np.asarray(archive["station_target_mw"][order][positions], dtype=np.float32)
            if target is None:
                target = current_target
            elif not np.allclose(target, current_target, rtol=1e-5, atol=1e-3):
                raise ValueError("History prediction archives contain different targets")
            station_predictions.append(prediction)
    if not station_predictions or target is None:
        raise ValueError("No history predictions supplied")
    return np.mean(station_predictions, axis=0, dtype=np.float32), target


def bilinear_weight_matrix(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    station_latitude: np.ndarray,
    station_longitude: np.ndarray,
) -> np.ndarray:
    latitude = np.asarray(grid_latitude, dtype=np.float64)
    longitude = np.asarray(grid_longitude, dtype=np.float64)
    if latitude[0] > latitude[-1]:
        latitude_order = np.arange(len(latitude) - 1, -1, -1)
        latitude_ascending = latitude[latitude_order]
    else:
        latitude_order = np.arange(len(latitude))
        latitude_ascending = latitude
    if longitude[0] > longitude[-1]:
        longitude_order = np.arange(len(longitude) - 1, -1, -1)
        longitude_ascending = longitude[longitude_order]
    else:
        longitude_order = np.arange(len(longitude))
        longitude_ascending = longitude
    weights = np.zeros((len(latitude) * len(longitude), len(station_latitude)), dtype=np.float32)
    for station, (station_lat, station_lon) in enumerate(zip(station_latitude, station_longitude)):
        row_hi = int(np.clip(np.searchsorted(latitude_ascending, station_lat), 1, len(latitude) - 1))
        col_hi = int(np.clip(np.searchsorted(longitude_ascending, station_lon), 1, len(longitude) - 1))
        row_lo, col_lo = row_hi - 1, col_hi - 1
        lat0, lat1 = latitude_ascending[row_lo], latitude_ascending[row_hi]
        lon0, lon1 = longitude_ascending[col_lo], longitude_ascending[col_hi]
        fy = float(np.clip((station_lat - lat0) / max(lat1 - lat0, 1e-12), 0.0, 1.0))
        fx = float(np.clip((station_lon - lon0) / max(lon1 - lon0, 1e-12), 0.0, 1.0))
        corners = (
            (latitude_order[row_lo], longitude_order[col_lo], (1.0 - fy) * (1.0 - fx)),
            (latitude_order[row_lo], longitude_order[col_hi], (1.0 - fy) * fx),
            (latitude_order[row_hi], longitude_order[col_lo], fy * (1.0 - fx)),
            (latitude_order[row_hi], longitude_order[col_hi], fy * fx),
        )
        for row, column, value in corners:
            weights[int(row) * len(longitude) + int(column), station] += value
    return weights


def extract_station_weather(
    weather_path: Path,
    weights: np.ndarray,
    output_path: Path,
    batch_size: int = 64,
) -> np.ndarray:
    """Create a reusable [cycle, hour*channel*station] fixed-bilinear feature array."""
    weather = np.load(weather_path, mmap_mode="r")
    cycles, hours, channels, rows, columns = weather.shape
    expected_shape = (cycles, hours * channels * weights.shape[1])
    if output_path.is_file():
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape == expected_shape and existing.dtype == np.float32:
            return existing
        raise ValueError(f"Cached weather features have wrong shape: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=expected_shape)
    for start in range(0, cycles, batch_size):
        stop = min(cycles, start + batch_size)
        batch = np.asarray(weather[start:stop], dtype=np.float32).reshape(stop - start, hours, channels, rows * columns)
        station = np.einsum("bhtg,gs->bhts", batch, weights, optimize=True)
        output[start:stop] = station.reshape(stop - start, -1)
    output.flush()
    del output
    os.replace(temporary, output_path)
    return np.load(output_path, mmap_mode="r")


def fit_pca_features(
    values: np.ndarray,
    fit_rows: np.ndarray,
    components: int,
    *,
    seed: int,
) -> tuple[PCA, np.ndarray]:
    component_count = min(int(components), len(fit_rows) - 1, values.shape[1])
    model = PCA(n_components=component_count, svd_solver="randomized", random_state=seed)
    model.fit(np.asarray(values[fit_rows], dtype=np.float32))
    transformed = np.empty((len(values), component_count), dtype=np.float32)
    batch_size = 2048
    for start in range(0, len(values), batch_size):
        stop = min(len(values), start + batch_size)
        transformed[start:stop] = model.transform(np.asarray(values[start:stop], dtype=np.float32))
    return model, transformed


class StandardizedRidge:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, solver="cholesky")

    def fit(self, features: np.ndarray, target: np.ndarray) -> "StandardizedRidge":
        scaled = self.scaler.fit_transform(features).astype(np.float32, copy=False)
        self.model.fit(scaled, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features).astype(np.float32, copy=False)
        return np.asarray(self.model.predict(scaled), dtype=np.float32)


def select_ridge(
    fit_features: np.ndarray,
    fit_target: np.ndarray,
    selection_features: np.ndarray,
    selection_target: np.ndarray,
    alphas: list[float],
    station_count: int,
    forecast_steps: int,
) -> tuple[StandardizedRidge, list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    selected: StandardizedRidge | None = None
    best = float("inf")
    for alpha in alphas:
        model = StandardizedRidge(alpha).fit(fit_features, fit_target)
        prediction = model.predict(selection_features)
        station_mae = float(np.mean(np.abs(prediction - selection_target)))
        prediction_3d = prediction.reshape(-1, station_count, forecast_steps)
        target_3d = selection_target.reshape(-1, station_count, forecast_steps)
        cluster_residual_mae = float(
            np.mean(np.abs(prediction_3d.sum(axis=1) - target_3d.sum(axis=1)))
        )
        records.append(
            {
                "alpha": float(alpha),
                "selection_station_residual_mae_mw": station_mae,
                "selection_cluster_residual_mae_mw": cluster_residual_mae,
            }
        )
        if cluster_residual_mae < best:
            selected, best = model, cluster_residual_mae
    assert selected is not None
    return selected, records


def cluster_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction.sum(axis=1) - target.sum(axis=1))))


def time_low_pass(values: np.ndarray, components: int) -> np.ndarray:
    coefficients = dct(values, axis=-1, norm="ortho")
    coefficients[..., int(components) :] = 0.0
    return idct(coefficients, axis=-1, norm="ortho").astype(np.float32)


def spatial_projection(values: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    projected = (flat @ components.T) @ components
    return projected.reshape(values.shape[0], values.shape[2], values.shape[1]).transpose(0, 2, 1).astype(np.float32)


def evaluate_restrictions(
    baseline_prediction: np.ndarray,
    target: np.ndarray,
    history_correction: np.ndarray,
    weather_increment: np.ndarray,
    spatial_components: np.ndarray,
    random_components: np.ndarray,
    temporal_counts: list[int],
    spatial_counts: list[int],
) -> list[dict[str, Any]]:
    history_prediction = baseline_prediction + history_correction
    rows: list[dict[str, Any]] = [
        {"restriction": "history_control", "cluster_mae_mw": cluster_mae(history_prediction, target)},
        {"restriction": "full_weather_increment", "cluster_mae_mw": cluster_mae(history_prediction + weather_increment, target)},
    ]
    for temporal in temporal_counts:
        increment = time_low_pass(weather_increment, temporal)
        rows.append(
            {
                "restriction": "temporal_dct",
                "temporal_components": temporal,
                "cluster_mae_mw": cluster_mae(history_prediction + increment, target),
            }
        )
    for spatial in spatial_counts:
        spatial_increment = spatial_projection(weather_increment, spatial_components[:spatial])
        random_increment = spatial_projection(weather_increment, random_components[:spatial])
        rows.extend(
            [
                {
                    "restriction": "spatial_residual_pca",
                    "spatial_components": spatial,
                    "cluster_mae_mw": cluster_mae(history_prediction + spatial_increment, target),
                },
                {
                    "restriction": "spatial_random_basis",
                    "spatial_components": spatial,
                    "cluster_mae_mw": cluster_mae(history_prediction + random_increment, target),
                },
            ]
        )
        for temporal in temporal_counts:
            joint = time_low_pass(spatial_increment, temporal)
            rows.append(
                {
                    "restriction": "joint_dct_residual_pca",
                    "temporal_components": temporal,
                    "spatial_components": spatial,
                    "cluster_mae_mw": cluster_mae(history_prediction + joint, target),
                }
            )
    return rows


def temporal_diagnostics(
    residual: np.ndarray,
    history_prediction: np.ndarray,
    weather_prediction: np.ndarray,
) -> list[dict[str, Any]]:
    target = dct(residual, axis=-1, norm="ortho")
    history = dct(history_prediction, axis=-1, norm="ortho")
    weather = dct(weather_prediction, axis=-1, norm="ortho")
    groups = {"low_0_3": slice(0, 4), "middle_4_11": slice(4, 12), "high_12_23": slice(12, 24)}
    rows = []
    total_energy = float(np.mean(np.sum(target**2, axis=-1)))
    for name, selection in groups.items():
        target_part = target[..., selection]
        history_mse = float(
            np.mean(np.sum((history[..., selection] - target_part) ** 2, axis=-1))
        )
        weather_mse = float(
            np.mean(np.sum((weather[..., selection] - target_part) ** 2, axis=-1))
        )
        energy = float(np.mean(np.sum(target_part**2, axis=-1)))
        rows.append(
            {
                "band": name,
                "residual_energy": energy,
                "relative_residual_energy": energy / max(total_energy, 1e-12),
                "history_control_mse": history_mse,
                "history_weather_mse": weather_mse,
                "weather_incremental_mse_reduction": history_mse - weather_mse,
                "weather_incremental_fraction_of_band_energy": (history_mse - weather_mse) / max(energy, 1e-12),
            }
        )
    positive = sum(max(0.0, row["weather_incremental_mse_reduction"]) for row in rows)
    for row in rows:
        row["share_of_positive_weather_mse_reduction"] = max(0.0, row["weather_incremental_mse_reduction"]) / max(positive, 1e-12)
    return rows


def spatial_diagnostics(
    residual: np.ndarray,
    history_prediction: np.ndarray,
    weather_prediction: np.ndarray,
    pca: PCA,
    count: int = 16,
) -> list[dict[str, Any]]:
    def coefficients(values: np.ndarray) -> np.ndarray:
        flat = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
        return flat @ pca.components_.T

    target_coefficients = coefficients(residual)
    history_coefficients = coefficients(history_prediction)
    weather_coefficients = coefficients(weather_prediction)
    rows = []
    for mode in range(min(count, pca.n_components_)):
        target = target_coefficients[:, mode]
        history_mse = float(np.mean((history_coefficients[:, mode] - target) ** 2))
        weather_mse = float(np.mean((weather_coefficients[:, mode] - target) ** 2))
        rows.append(
            {
                "mode": mode + 1,
                "training_residual_variance_ratio": float(pca.explained_variance_ratio_[mode]),
                "history_control_mse": history_mse,
                "history_weather_mse": weather_mse,
                "weather_incremental_mse_reduction": history_mse - weather_mse,
            }
        )
    positive = sum(max(0.0, row["weather_incremental_mse_reduction"]) for row in rows)
    for row in rows:
        row["share_of_positive_weather_mse_reduction_first16"] = max(0.0, row["weather_incremental_mse_reduction"]) / max(positive, 1e-12)
    return rows


def stratified_gain(
    values: np.ndarray,
    fit_values: np.ndarray,
    history_error: np.ndarray,
    weather_error: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    boundaries = np.quantile(fit_values, np.linspace(0.0, 1.0, len(labels) + 1)[1:-1])
    bins = np.digitize(values, boundaries)
    rows = []
    for index, label in enumerate(labels):
        mask = bins == index
        rows.append(
            {
                "group": label,
                "samples": int(mask.sum()),
                "history_control_mae_mw": float(history_error[mask].mean()),
                "history_weather_mae_mw": float(weather_error[mask].mean()),
                "weather_gain_mw": float(history_error[mask].mean() - weather_error[mask].mean()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NWP predictability conditional on historical power")
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--prediction-root", default="artifacts/incremental_predictability_v1/exports")
    parser.add_argument("--output-dir", default="artifacts/incremental_predictability_v1/diagnostics")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--history-components", type=int, default=96)
    parser.add_argument("--weather-components", type=int, default=128)
    parser.add_argument("--spatial-components", type=int, default=32)
    parser.add_argument("--alphas", default="1,10,100,1000")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    split_dir = cache_dir / args.split
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = [
        Path(args.prediction_root).resolve() / f"history_seed{seed}" / f"predictions_{args.split}_station.npz"
        for seed in (42, 43, 44)
    ]
    origin = np.load(split_dir / "origin_unix.npy")
    history = np.load(split_dir / "station_history.npy", mmap_mode="r")
    cycle_index = np.load(split_dir / "weather_cycle_index.npy")
    weather_offsets = np.load(split_dir / "weather_time_offset_hours.npy")
    origin_phase = np.load(split_dir / "origin_phase.npy")
    capacity = np.load(cache_dir / "static" / "station_capacity_mw.npy")
    history_baseline, target = align_history_ensemble(prediction_paths, origin)
    residual = target - history_baseline
    partitions = chronological_day_split(origin)
    fit, selection, confirmation = (partitions[name] for name in ("fit", "selection", "confirmation"))

    weight_matrix = bilinear_weight_matrix(
        np.load(cache_dir / "static" / "weather_latitude.npy"),
        np.load(cache_dir / "static" / "weather_longitude.npy"),
        np.load(cache_dir / "static" / "station_latitude.npy"),
        np.load(cache_dir / "static" / "station_longitude.npy"),
    )
    station_weather = extract_station_weather(
        split_dir / "weather.npy", weight_matrix, output_dir / f"station_weather_{args.split}.npy"
    )
    fit_cycles = np.unique(cycle_index[fit])
    _, weather_cycle_features = fit_pca_features(
        station_weather, fit_cycles, args.weather_components, seed=args.seed
    )
    history_flat = np.asarray(history, dtype=np.float32).reshape(len(history), -1)
    _, history_features = fit_pca_features(history_flat, fit, args.history_components, seed=args.seed)
    phase_angle = 2.0 * np.pi * origin_phase.astype(np.float32) / 4.0
    weather_features = np.concatenate(
        (
            weather_cycle_features[cycle_index],
            weather_offsets.astype(np.float32) / 8.0,
            np.sin(phase_angle)[:, None],
            np.cos(phase_angle)[:, None],
        ),
        axis=1,
    )
    interaction = (
        history_features[:, :8, None] * weather_features[:, None, :16]
    ).reshape(len(history), -1)
    feature_sets = {
        "history_control": history_features,
        "history_weather_additive": np.concatenate((history_features, weather_features), axis=1),
        "history_weather_interaction": np.concatenate((history_features, weather_features, interaction), axis=1),
    }
    flat_residual = residual.reshape(len(residual), -1)
    alphas = [float(value) for value in args.alphas.split(",") if value]
    models: dict[str, StandardizedRidge] = {}
    alpha_records: dict[str, list[dict[str, float]]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, features in feature_sets.items():
        model, records = select_ridge(
            features[fit],
            flat_residual[fit],
            features[selection],
            flat_residual[selection],
            alphas,
            target.shape[1],
            target.shape[2],
        )
        models[name], alpha_records[name] = model, records
        predictions[name] = model.predict(features[confirmation]).reshape(-1, target.shape[1], target.shape[2])

    rng = np.random.default_rng(args.seed)
    shuffled_features = weather_features.copy()
    for partition in partitions.values():
        shuffled_features[partition] = weather_features[rng.permutation(partition)]
    misaligned_features = weather_features.copy()
    for partition in partitions.values():
        misaligned_features[partition] = np.roll(weather_features[partition], 7 * 96, axis=0)
    negative_predictions: dict[str, np.ndarray] = {}
    selected_alpha = models["history_weather_additive"].alpha
    for name, altered_weather in (
        ("shuffled_weather", shuffled_features),
        ("misaligned_weather_7d", misaligned_features),
    ):
        features = np.concatenate((history_features, altered_weather), axis=1)
        model = StandardizedRidge(selected_alpha).fit(features[fit], flat_residual[fit])
        negative_predictions[name] = model.predict(features[confirmation]).reshape(-1, target.shape[1], target.shape[2])

    residual_fit_vectors = residual[fit].transpose(0, 2, 1).reshape(-1, residual.shape[1])
    spatial_pca = PCA(
        n_components=min(args.spatial_components, residual.shape[1]),
        svd_solver="randomized",
        random_state=args.seed,
    ).fit(residual_fit_vectors)
    random_basis, _ = np.linalg.qr(rng.normal(size=(residual.shape[1], residual.shape[1])))
    random_components = random_basis.T.astype(np.float32)

    baseline_confirm = history_baseline[confirmation]
    target_confirm = target[confirmation]
    residual_confirm = residual[confirmation]
    history_correction = predictions["history_control"]
    additive_correction = predictions["history_weather_additive"]
    interaction_correction = predictions["history_weather_interaction"]
    weather_increment = additive_correction - history_correction
    restrictions_selection: list[dict[str, Any]] = []
    selection_predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        selection_predictions[name] = model.predict(feature_sets[name][selection]).reshape(-1, target.shape[1], target.shape[2])
    restrictions_selection = evaluate_restrictions(
        history_baseline[selection],
        target[selection],
        selection_predictions["history_control"],
        selection_predictions["history_weather_additive"] - selection_predictions["history_control"],
        spatial_pca.components_,
        random_components,
        [2, 4, 8, 12, 24],
        [2, 4, 8, 16, 32],
    )
    restrictions_confirmation = evaluate_restrictions(
        baseline_confirm,
        target_confirm,
        history_correction,
        weather_increment,
        spatial_pca.components_,
        random_components,
        [2, 4, 8, 12, 24],
        [2, 4, 8, 16, 32],
    )

    model_rows: list[dict[str, Any]] = []
    all_predictions = {**predictions, **negative_predictions}
    for name, correction in all_predictions.items():
        final_prediction = baseline_confirm + correction
        model_rows.append(
            {
                "model": name,
                "cluster_mae_mw": cluster_mae(final_prediction, target_confirm),
                "station_mae_mw": float(np.mean(np.abs(final_prediction - target_confirm))),
                "alpha": float(models[name].alpha) if name in models else selected_alpha,
            }
        )
    model_rows.insert(
        0,
        {
            "model": "frozen_history_ensemble",
            "cluster_mae_mw": cluster_mae(baseline_confirm, target_confirm),
            "station_mae_mw": float(np.mean(np.abs(baseline_confirm - target_confirm))),
            "alpha": None,
        },
    )

    history_cluster_error = np.mean(
        np.abs((baseline_confirm + history_correction).sum(axis=1) - target_confirm.sum(axis=1)), axis=1
    )
    additive_cluster_error = np.mean(
        np.abs((baseline_confirm + additive_correction).sum(axis=1) - target_confirm.sum(axis=1)), axis=1
    )
    interaction_cluster_error = np.mean(
        np.abs((baseline_confirm + interaction_correction).sum(axis=1) - target_confirm.sum(axis=1)), axis=1
    )
    additive_delta, additive_low, additive_high = block_bootstrap_difference(
        additive_cluster_error,
        history_cluster_error,
        origin[confirmation],
        args.bootstrap_repeats,
        args.seed,
    )
    interaction_delta, interaction_low, interaction_high = block_bootstrap_difference(
        interaction_cluster_error,
        additive_cluster_error,
        origin[confirmation],
        args.bootstrap_repeats,
        args.seed + 1,
    )

    temporal_rows = temporal_diagnostics(
        residual_confirm, history_correction, additive_correction
    )
    spatial_rows = spatial_diagnostics(
        residual_confirm, history_correction, additive_correction, spatial_pca
    )
    history_last_cluster = np.asarray(history[confirmation, :, -1]) @ capacity
    history_fit_last_cluster = np.asarray(history[fit, :, -1]) @ capacity
    state_rows = stratified_gain(
        history_last_cluster,
        history_fit_last_cluster,
        history_cluster_error,
        additive_cluster_error,
        ["low_current_power", "middle_current_power", "high_current_power"],
    )

    channel_names = np.load(cache_dir / "static" / "nwp_channel_name.npy").astype(str)
    channel_lookup = {name.lower(): index for index, name in enumerate(channel_names)}
    u_index = next((channel_lookup[name] for name in ("u10", "10u") if name in channel_lookup), None)
    v_index = next((channel_lookup[name] for name in ("v10", "10v") if name in channel_lookup), None)
    regime_rows: list[dict[str, Any]] = []
    if u_index is not None and v_index is not None:
        raw_weather = np.load(split_dir / "weather.npy", mmap_mode="r")
        nwp_mean = np.load(cache_dir / "static" / "nwp_mean.npy")
        nwp_std = np.load(cache_dir / "static" / "nwp_std.npy")
        regional_u = raw_weather[:, :, u_index].mean(axis=(-2, -1)) * nwp_std[u_index] + nwp_mean[u_index]
        regional_v = raw_weather[:, :, v_index].mean(axis=(-2, -1)) * nwp_std[v_index] + nwp_mean[v_index]
        regional_speed = np.sqrt(regional_u**2 + regional_v**2)
        trend = regional_speed[:, -1] - regional_speed[:, 0]
        regime_rows = stratified_gain(
            trend[cycle_index[confirmation]],
            trend[cycle_index[fit]],
            history_cluster_error,
            additive_cluster_error,
            ["decreasing_wind", "steady_wind", "increasing_wind"],
        )

    lead_history_mae = np.mean(
        np.abs((baseline_confirm + history_correction).sum(axis=1) - target_confirm.sum(axis=1)), axis=0
    )
    lead_additive_mae = np.mean(
        np.abs((baseline_confirm + additive_correction).sum(axis=1) - target_confirm.sum(axis=1)), axis=0
    )
    lead_rows = [
        {
            "lead_step": index + 1,
            "lead_hours": (index + 1) / 4.0,
            "history_control_mae_mw": float(lead_history_mae[index]),
            "history_weather_mae_mw": float(lead_additive_mae[index]),
            "weather_gain_mw": float(lead_history_mae[index] - lead_additive_mae[index]),
        }
        for index in range(target.shape[2])
    ]

    full_mae = next(row["cluster_mae_mw"] for row in model_rows if row["model"] == "history_weather_additive")
    best_selection_restriction = min(
        (row for row in restrictions_selection if row["restriction"] != "history_control"),
        key=lambda row: row["cluster_mae_mw"],
    )
    matching_confirmation = [
        row
        for row in restrictions_confirmation
        if all(row.get(key) == value for key, value in best_selection_restriction.items() if key != "cluster_mae_mw")
    ][0]
    decisions = {
        "nwp_has_incremental_predictability": bool(additive_high < 0.0),
        "state_interaction_supported": bool(interaction_high < 0.0),
        "selected_restriction": {key: value for key, value in best_selection_restriction.items() if key != "cluster_mae_mw"},
        "selected_restriction_confirmation_mae_mw": matching_confirmation["cluster_mae_mw"],
        "full_increment_confirmation_mae_mw": full_mae,
        "restriction_preserves_full_gain_within_1mw": bool(matching_confirmation["cluster_mae_mw"] <= full_mae + 1.0),
    }
    payload: dict[str, Any] = {
        "scope": {
            "split": args.split,
            "fit_samples": int(len(fit)),
            "selection_samples": int(len(selection)),
            "confirmation_samples": int(len(confirmation)),
            "fit_day_range": [int(origin[fit].min()), int(origin[fit].max())],
            "selection_day_range": [int(origin[selection].min()), int(origin[selection].max())],
            "confirmation_day_range": [int(origin[confirmation].min()), int(origin[confirmation].max())],
            "history_prediction_seeds": [42, 43, 44],
            "test_data_used": False,
        },
        "model_results": model_rows,
        "paired_bootstrap": {
            "history_weather_additive_minus_history_control": {
                "delta_mae_mw": additive_delta,
                "ci95_low_mw": additive_low,
                "ci95_high_mw": additive_high,
            },
            "state_interaction_minus_additive": {
                "delta_mae_mw": interaction_delta,
                "ci95_low_mw": interaction_low,
                "ci95_high_mw": interaction_high,
            },
        },
        "ridge_selection": alpha_records,
        "spatial_training_cumulative_variance": {
            str(count): float(spatial_pca.explained_variance_ratio_[:count].sum())
            for count in (1, 2, 4, 8, 16, min(32, spatial_pca.n_components_))
        },
        "decisions": decisions,
    }
    save_json(output_dir / "diagnostics.json", payload)
    for filename, rows in (
        ("model_results.csv", model_rows),
        ("restriction_selection.csv", restrictions_selection),
        ("restriction_confirmation.csv", restrictions_confirmation),
        ("temporal_bands.csv", temporal_rows),
        ("spatial_modes.csv", spatial_rows),
        ("lead_time_gain.csv", lead_rows),
        ("state_groups.csv", state_rows),
        ("weather_regimes.csv", regime_rows),
    ):
        if rows:
            write_csv(output_dir / filename, rows)
    np.savez_compressed(
        output_dir / "confirmation_predictions.npz",
        origin_unix=origin[confirmation],
        station_target_mw=target_confirm,
        frozen_history_prediction_mw=baseline_confirm,
        history_control_prediction_mw=baseline_confirm + history_correction,
        history_weather_prediction_mw=baseline_confirm + additive_correction,
        history_weather_interaction_prediction_mw=baseline_confirm + interaction_correction,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
