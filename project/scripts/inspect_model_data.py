from __future__ import annotations

import argparse
import json

import numpy as np

from wind_forcing.config import load_config
from wind_forcing.data import ForecastDataset, create_forecast_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prepared forced-evolution data")
    parser.add_argument("--config", default="configs/forced_evolution_pad5.yaml")
    parser.add_argument("--preload-weather", default="mmap")
    args = parser.parse_args()
    config = load_config(args.config)
    summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        dataset = ForecastDataset(
            config["data"]["cache_dir"],
            split,
            preload_weather=args.preload_weather,
            verbose=True,
        )
        loader = create_forecast_loader(
            dataset,
            cycles_per_batch=2,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))
        capacities = dataset.static["station_capacity_mw"].reshape(1, -1, 1)
        cluster = (batch["station_future"].numpy() * capacities).sum(axis=1)
        cluster /= float(dataset.manifest["total_capacity_mw"])
        error = float(np.max(np.abs(cluster - batch["cluster_future"].numpy())))
        if error >= 2e-6:
            raise ValueError(f"Cluster sum verification failed for {split}: {error}")
        summary[split] = {
            "samples": len(dataset),
            "cycles": int(dataset.manifest["splits"][split]["cycle_count"]),
            "batch_samples": int(batch["station_history"].shape[0]),
            "batch_unique_weather_cycles": int(batch["weather"].shape[0]),
            "weather_shape": list(batch["weather"].shape[1:]),
            "station_history_shape": list(batch["station_history"].shape[1:]),
            "station_future_shape": list(batch["station_future"].shape[1:]),
            "max_cluster_sum_error": error,
        }
        dataset.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
