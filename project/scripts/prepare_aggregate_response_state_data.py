from __future__ import annotations

import argparse

from wind_forcing.data import AggregateResponseCacheBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal historical Pangu--power pairs")
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--aggregate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-cycles", type=int, default=32)
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    builder = AggregateResponseCacheBuilder(
        args.source_h5,
        args.aggregate_cache,
        args.output,
        chunk_cycles=args.chunk_cycles,
    )
    builder.build()
    builder.audit(verify_checksums=args.verify_checksums)


if __name__ == "__main__":
    main()
