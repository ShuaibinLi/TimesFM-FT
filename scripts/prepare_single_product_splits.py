#!/usr/bin/env python3
"""Build day-safe single-product NPZ splits from 500 ms WMP Parquet files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq

LOGGER = logging.getLogger("prepare_single_product_splits")
DEFAULT_SOURCE_ROOT = (
    "gs://vatic-hft-user-data/shuaibin.li/timesfm_ft/"
    "wmp_500ms_20221101_20260130"
)
SPLITS = ("train", "val", "test")


def _read_dates(path: Path) -> list[str]:
    dates = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(dates) != len(set(dates)):
        raise ValueError(f"duplicate dates in {path}")
    return dates


def build_split(
    *,
    product: str,
    split: str,
    dates: list[str],
    source_root: str,
    destination: Path,
    context_length: int,
    horizon_length: int,
    stride: int,
    interval_ns: int,
) -> None:
    fs, source_path = fsspec.core.url_to_fs(source_root.rstrip("/"))
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    sample_dates: list[np.ndarray] = []
    samples_by_day: dict[str, int] = {}

    for index, day in enumerate(dates, start=1):
        parquet_path = f"{source_path}/{product}/date={day}/part0.parquet"
        with fs.open(parquet_path, "rb") as stream:
            table = pq.read_table(stream, columns=["timestamp_ns", "wmp"])
        timestamp = table.column("timestamp_ns").to_numpy(zero_copy_only=False)
        wmp = table.column("wmp").to_numpy(zero_copy_only=False).astype(np.float32)
        if not np.all(np.diff(timestamp) == interval_ns):
            raise ValueError(f"{parquet_path} is not a {interval_ns} ns grid")
        if not np.all(np.isfinite(wmp)):
            raise ValueError(f"{parquet_path} contains non-finite WMP")

        cutoffs = np.arange(
            context_length - 1,
            len(wmp) - horizon_length,
            stride,
            dtype=np.int64,
        )
        context_starts = cutoffs - context_length + 1
        future_starts = cutoffs + 1
        contexts.append(
            np.stack([wmp[start : start + context_length] for start in context_starts])
        )
        futures.append(
            np.stack([wmp[start : start + horizon_length] for start in future_starts])
        )
        timestamps.append(timestamp[cutoffs].astype(np.int64))
        sample_dates.append(np.full(len(cutoffs), int(day), dtype=np.int32))
        samples_by_day[day] = len(cutoffs)
        if index % 25 == 0 or index == len(dates):
            LOGGER.info(
                "read product=%s split=%s days=%d/%d",
                product,
                split,
                index,
                len(dates),
            )

    context_values = np.concatenate(contexts)
    future_values = np.concatenate(futures)
    cutoff_timestamps = np.concatenate(timestamps)
    dates_array = np.concatenate(sample_dates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        context_values=context_values,
        future_values=future_values,
        timestamps=cutoff_timestamps,
        dates=dates_array,
    )
    metadata = {
        "product": product,
        "split": split,
        "source_root": source_root,
        "date_count": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "sampling_interval_seconds": interval_ns / 1_000_000_000,
        "context_length": context_length,
        "context_seconds": context_length * interval_ns / 1_000_000_000,
        "horizon_length": horizon_length,
        "horizon_seconds": horizon_length * interval_ns / 1_000_000_000,
        "stride": stride,
        "samples": len(context_values),
        "samples_by_day": samples_by_day,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "wrote product=%s split=%s samples=%d path=%s",
        product,
        split,
        len(context_values),
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=("ZN", "ES", "all"), default="all")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split-dir", type=Path, default=Path("configs/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--sampling-interval-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if min(args.context_length, args.horizon_length, args.stride) <= 0:
        parser.error("context length, horizon length, and stride must be positive")
    if args.sampling_interval_seconds <= 0:
        parser.error("sampling interval must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    products = ("ZN", "ES") if args.product == "all" else (args.product,)
    interval_ns = int(args.sampling_interval_seconds * 1_000_000_000)
    for product in products:
        product_dir = args.output_root / f"{product.lower()}-wmp-500ms" / "splits"
        for split in SPLITS:
            dates = _read_dates(args.split_dir / f"dates-{split}.txt")
            build_split(
                product=product,
                split=split,
                dates=dates,
                source_root=args.source_root,
                destination=product_dir
                / f"{split}_c{args.context_length}_h{args.horizon_length}.npz",
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                stride=args.stride,
                interval_ns=interval_ns,
            )


if __name__ == "__main__":
    main()
