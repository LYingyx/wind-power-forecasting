from __future__ import annotations

import argparse

from wind_forcing.data import AggregateCacheBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the aggregate-only cluster cache")
    parser.add_argument("--source-cache", default="data/model_cache_pad5_f8")
    parser.add_argument("--output", default="data/aggregate_cluster_v1")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    builder = AggregateCacheBuilder(args.source_cache, args.output)
    builder.build()
    builder.audit(verify_checksums=args.verify_checksums)


if __name__ == "__main__":
    main()
