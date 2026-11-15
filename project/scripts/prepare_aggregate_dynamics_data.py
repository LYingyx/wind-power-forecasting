from __future__ import annotations

import argparse

from wind_forcing.data import AggregateDynamicsTimeCacheBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal current-weather timing metadata")
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--aggregate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    builder = AggregateDynamicsTimeCacheBuilder(args.source_h5, args.aggregate_cache, args.output)
    builder.build()
    print(builder.audit(verify_checksums=args.verify_checksums), flush=True)


if __name__ == "__main__":
    main()
