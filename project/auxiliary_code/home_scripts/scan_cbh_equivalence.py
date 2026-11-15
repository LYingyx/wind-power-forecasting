import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_original_cbh_20260917_v1")
sys.path.insert(0, str(ROOT / "code"))

from cbh_adapter import CBHAdapterSR  # noqa: E402
from train_adapter import CBHData, Ensemble, evaluate_samples  # noqa: E402
from paths import V2  # noqa: E402


def main():
    device = "cuda:5"
    seeds = list(range(42, 52))
    farms = json.loads((V2 / "data/selected_farms.json").read_text())
    report = torch.tensor(np.load(V2 / "data/val_report_indices.npy"), device=device)
    torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    all_rows = []
    for station, station_row in enumerate(farms["stations"]):
        data = CBHData("val", station, seeds, device)
        models = [
            CBHAdapterSR("LINEAR", torch.ones(14), seed, station_row["station_index"])
            for seed in seeds
        ]
        ensemble = Ensemble(models, device)
        prediction = evaluate_samples(ensemble, data, report, "LINEAR", 1)[:, :, 0]
        row = []
        for seed_index, seed in enumerate(seeds):
            original = torch.tensor(
                np.load(ROOT / "original_eval" / f"seed{seed}" / "predictions_fp32.npy")[:, station]
            )
            row.append(float((prediction[seed_index] - original).abs().max()))
        all_rows.append(row)
        print(station_row["station_id"], max(row), row, flush=True)
        del data, models, ensemble, prediction
        torch.cuda.empty_cache()
    values = np.asarray(all_rows)
    print("global", float(values.max()), "station", farms["station_ids"][int(np.unravel_index(values.argmax(), values.shape)[0])], "seed", seeds[int(np.unravel_index(values.argmax(), values.shape)[1])])


if __name__ == "__main__":
    main()
