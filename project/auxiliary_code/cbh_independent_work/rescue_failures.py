"""Atomically reclaim failed site/seed jobs on GPUs with spare memory."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch

from independent_models import OUT, load_static, atomic_json
from train_independent_seed import now, run_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescue-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--min-free-mib", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    static = load_static()
    failure_dir = OUT / "status/failures"
    claim_dir = OUT / "status/rescue_claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    while True:
        if args.min_free_mib and args.physical_gpu is not None:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={args.physical_gpu}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            free_mib = int(query.stdout.strip().splitlines()[0])
            if free_mib < args.min_free_mib:
                time.sleep(15)
                continue
        claimed = None
        for failure in sorted(failure_dir.glob("site_*_seed_*.json")):
            destination = claim_dir / f"{failure.stem}__{args.rescue_id}.json"
            try:
                os.replace(failure, destination)
            except FileNotFoundError:
                continue
            claimed = destination
            break
        if claimed is not None:
            record = json.loads(claimed.read_text())
            site, seed = int(record["site"]), int(record["seed"])
            success = OUT / "independent_seed" / f"site_{site:02d}" / f"seed_{seed}" / "SUCCESS.json"
            if success.exists():
                completed.append({"site": site, "seed": seed, "status": "already_complete"})
                continue
            try:
                run_job(site, seed, static, device, args)
                completed.append({"site": site, "seed": seed, "status": "rescued"})
            except Exception as error:
                # Requeue for another rescue worker.  Use atomic_json because
                # the production run may have left a partial job directory.
                atomic_json(
                    failure_dir / f"site_{site:02d}_seed_{seed}.json",
                    {
                        "site": site,
                        "seed": seed,
                        "rescue_id": args.rescue_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "failed": now(),
                    },
                )
                completed.append({"site": site, "seed": seed, "status": "rescue_failed"})
                time.sleep(10)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            continue
        main_workers_done = len(list((OUT / "status/workers").glob("worker_*.json"))) == 64
        success_count = len(list((OUT / "independent_seed").glob("site_*/seed_*/SUCCESS.json")))
        if main_workers_done and success_count == 560 and not list(failure_dir.glob("*.json")):
            break
        time.sleep(15)
    atomic_json(
        OUT / "status/workers" / f"rescue_{args.rescue_id}.json",
        {"rescue_id": args.rescue_id, "completed_jobs": completed, "completed": now()},
    )


if __name__ == "__main__":
    main()
