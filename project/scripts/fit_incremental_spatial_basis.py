from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

try:
    from scripts.analyze_incremental_predictability import align_history_ensemble
except ModuleNotFoundError:
    from analyze_incremental_predictability import align_history_ensemble  # type: ignore[no-redef]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit spatial bases from 2022 history-model residuals only"
    )
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument(
        "--prediction-root", default="artifacts/incremental_predictability_v1/exports"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/incremental_predictability_v1/bases"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    origin = np.load(cache_dir / "train" / "origin_unix.npy")
    prediction_paths = [
        Path(args.prediction_root).resolve()
        / f"history_seed{seed}"
        / "predictions_train_station.npz"
        for seed in (42, 43, 44)
    ]
    history_prediction, target = align_history_ensemble(prediction_paths, origin)
    residual = target - history_prediction
    vectors = residual.transpose(0, 2, 1).reshape(-1, residual.shape[1])
    pca = PCA(n_components=residual.shape[1], svd_solver="full").fit(vectors)
    rng = np.random.default_rng(args.seed)
    random_basis, _ = np.linalg.qr(rng.normal(size=(residual.shape[1], residual.shape[1])))

    np.save(output / "residual_pca_mw.npy", pca.components_.astype(np.float32))
    np.save(output / "random_orthogonal.npy", random_basis.T.astype(np.float32))
    payload = {
        "fit_split": "train",
        "fit_year": 2022,
        "samples": int(len(residual)),
        "vectors": int(len(vectors)),
        "history_prediction_seeds": [42, 43, 44],
        "residual_units": "station_MW",
        "cumulative_explained_variance": {
            str(count): float(pca.explained_variance_ratio_[:count].sum())
            for count in (1, 2, 4, 8, 16, 32, 56)
        },
    }
    temporary = output / f".basis_metrics.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "basis_metrics.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
