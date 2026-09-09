#!/usr/bin/env python3
"""Build audited, memory-mapped single-product datasets from 500 ms WMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("prepare_single_product_splits")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    "gs://vatic-hft-user-data/shuaibin.li/timesfm_ft/"
    "wmp_500ms_20221101_20260130"
)
SPLITS = ("train", "val", "test")
SESSION_TIMEZONE = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 15)
EARLY_CLOSES = (time(13, 0), time(13, 15))
SESSION_BOUNDARY_TOLERANCE_SECONDS = 30
PRODUCT_TICK_SIZES = {"ZN": 0.015625, "ES": 0.25}


def _read_dates(path: Path) -> list[str]:
    dates = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not dates:
        raise ValueError(f"no dates in {path}")
    if len(dates) != len(set(dates)):
        raise ValueError(f"duplicate dates in {path}")
    if dates != sorted(dates):
        raise ValueError(f"dates are not sorted in {path}")
    return dates


def audit_splits(split_dir: Path) -> dict[str, tuple[list[str], Path]]:
    result = {
        split: (_read_dates(split_dir / f"dates-{split}.txt"), split_dir / f"dates-{split}.txt")
        for split in SPLITS
    }
    sets = {split: set(values) for split, (values, _) in result.items()}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(sets[left] & sets[right])
            if overlap:
                raise ValueError(
                    f"{left}/{right} date overlap: {overlap[:5]}"
                )
    if not (
        result["train"][0][-1] < result["val"][0][0]
        and result["val"][0][-1] < result["test"][0][0]
    ):
        raise ValueError("splits are not strictly chronological")
    return result


def _part_sort_key(path: str) -> int:
    match = re.search(r"part(\d+)\.parquet$", path)
    if match is None:
        raise ValueError(f"unexpected parquet part name: {path}")
    return int(match.group(1))


def _parts_for_day(
    fs: fsspec.AbstractFileSystem,
    source_path: str,
    product: str,
    day: str,
) -> list[str]:
    parts = sorted(
        fs.glob(f"{source_path}/{product}/date={day}/part*.parquet"),
        key=_part_sort_key,
    )
    if not parts:
        raise FileNotFoundError(f"no parquet parts for {product} {day}")
    return parts


def _row_count(fs: fsspec.AbstractFileSystem, parts: list[str]) -> int:
    total = 0
    for part in parts:
        with fs.open(part, "rb") as stream:
            total += pq.ParquetFile(stream).metadata.num_rows
    return total


def _read_day(
    fs: fsspec.AbstractFileSystem,
    parts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    tables = []
    for part in parts:
        with fs.open(part, "rb") as stream:
            tables.append(pq.read_table(stream, columns=["timestamp_ns", "wmp"]))
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    timestamp = table.column("timestamp_ns").to_numpy(zero_copy_only=False)
    wmp = table.column("wmp").to_numpy(zero_copy_only=False).astype(np.float64)
    return timestamp, wmp


def _validate_day(
    timestamp: np.ndarray,
    wmp: np.ndarray,
    *,
    product: str,
    day: str,
    interval_ns: int,
    minimum_rows: int,
) -> None:
    if len(timestamp) < minimum_rows:
        raise ValueError(
            f"{product} {day} has {len(timestamp)} rows; requires {minimum_rows}"
        )
    if int(timestamp[0]) % interval_ns != 0:
        raise ValueError(f"{product} {day} first timestamp is not grid-aligned")
    if not np.all(np.diff(timestamp) == interval_ns):
        raise ValueError(f"{product} {day} is not a {interval_ns} ns grid")
    if not np.all(np.isfinite(wmp)):
        raise ValueError(f"{product} {day} contains non-finite WMP")

    first = datetime.fromtimestamp(int(timestamp[0]) / 1e9, SESSION_TIMEZONE)
    last = datetime.fromtimestamp(int(timestamp[-1]) / 1e9, SESSION_TIMEZONE)
    expected_date = datetime.strptime(day, "%Y%m%d").date()
    if first.date() != expected_date or last.date() != expected_date:
        raise ValueError(
            f"{product} {day} timestamps map to {first.date()}..{last.date()}"
        )
    if first.time() < SESSION_OPEN or first.time() > SESSION_CLOSE:
        raise ValueError(f"{product} {day} starts outside the RTH session: {first}")
    if last.time() > SESSION_CLOSE:
        raise ValueError(f"{product} {day} ends after the RTH session: {last}")
    session_open = datetime.combine(
        expected_date,
        SESSION_OPEN,
        tzinfo=SESSION_TIMEZONE,
    )
    if (first - session_open).total_seconds() > SESSION_BOUNDARY_TOLERANCE_SECONDS:
        raise ValueError(f"{product} {day} starts too late for RTH: {first}")
    normal_close = datetime.combine(
        expected_date,
        SESSION_CLOSE,
        tzinfo=SESSION_TIMEZONE,
    )
    near_normal_close = (
        abs((last - normal_close).total_seconds())
        <= SESSION_BOUNDARY_TOLERANCE_SECONDS
    )
    near_early_close = any(
        abs(
            (
                last
                - datetime.combine(
                    expected_date,
                    close_time,
                    tzinfo=SESSION_TIMEZONE,
                )
            ).total_seconds()
        )
        <= SESSION_BOUNDARY_TOLERANCE_SECONDS
        for close_time in EARLY_CLOSES
    )
    if not (near_normal_close or near_early_close):
        raise ValueError(
            f"{product} {day} has an unrecognized RTH close: {last}"
        )


def _date_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(
    *,
    product: str,
    split: str,
    dates: list[str],
    dates_path: Path,
    source_root: str,
    destination: Path,
    context_length: int,
    horizon_length: int,
    stride: int,
    interval_ns: int,
    target_mode: str = "level",
    tick_size: float = 1.0,
    overwrite: bool = False,
) -> None:
    if target_mode not in {"level", "delta_ticks"}:
        raise ValueError(f"unsupported target_mode={target_mode!r}")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    fs, source_path = fsspec.core.url_to_fs(source_root.rstrip("/"))
    day_parts: dict[str, list[str]] = {}
    sample_counts: dict[str, int] = {}
    for day in dates:
        parts = _parts_for_day(fs, source_path, product, day)
        rows = _row_count(fs, parts)
        first_cutoff = context_length if target_mode == "delta_ticks" else context_length - 1
        sample_count = len(range(first_cutoff, rows - horizon_length, stride))
        if sample_count <= 0:
            raise ValueError(f"{product} {day} produces no windows from {rows} rows")
        day_parts[day] = parts
        sample_counts[day] = sample_count

    total_samples = sum(sample_counts.values())
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to rebuild"
            )
        shutil.rmtree(destination)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    contexts = np.lib.format.open_memmap(
        temporary / "context_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, context_length),
    )
    futures = np.lib.format.open_memmap(
        temporary / "future_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, horizon_length),
    )
    timestamps = np.lib.format.open_memmap(
        temporary / "timestamps.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_samples,),
    )
    sample_dates = np.lib.format.open_memmap(
        temporary / "dates.npy",
        mode="w+",
        dtype=np.int32,
        shape=(total_samples,),
    )
    cutoff_wmp = (
        np.lib.format.open_memmap(
            temporary / "cutoff_wmp.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_samples,),
        )
        if target_mode == "delta_ticks"
        else None
    )

    offset = 0
    for index, day in enumerate(dates, start=1):
        timestamp, wmp = _read_day(fs, day_parts[day])
        _validate_day(
            timestamp,
            wmp,
            product=product,
            day=day,
            interval_ns=interval_ns,
            minimum_rows=(
                context_length
                + horizon_length
                + (1 if target_mode == "delta_ticks" else 0)
            ),
        )
        first_cutoff = (
            context_length if target_mode == "delta_ticks" else context_length - 1
        )
        cutoffs = np.arange(
            first_cutoff,
            len(wmp) - horizon_length,
            stride,
            dtype=np.int64,
        )
        count = len(cutoffs)
        if count != sample_counts[day]:
            raise RuntimeError(f"{product} {day} row count changed during build")
        target = slice(offset, offset + count)
        if target_mode == "delta_ticks":
            delta_ticks = (np.diff(wmp) / tick_size).astype(np.float32)
            context_windows = np.lib.stride_tricks.sliding_window_view(
                delta_ticks, context_length
            )
            future_windows = np.lib.stride_tricks.sliding_window_view(
                delta_ticks, horizon_length
            )
            contexts[target] = context_windows[cutoffs - context_length]
            futures[target] = future_windows[cutoffs]
            assert cutoff_wmp is not None
            cutoff_wmp[target] = wmp[cutoffs]
        else:
            context_windows = np.lib.stride_tricks.sliding_window_view(
                wmp, context_length
            )
            future_windows = np.lib.stride_tricks.sliding_window_view(
                wmp, horizon_length
            )
            contexts[target] = context_windows[cutoffs - context_length + 1]
            futures[target] = future_windows[cutoffs + 1]
        timestamps[target] = timestamp[cutoffs]
        sample_dates[target] = int(day)
        offset += count
        if index % 25 == 0 or index == len(dates):
            LOGGER.info(
                "write product=%s split=%s days=%d/%d",
                product,
                split,
                index,
                len(dates),
            )

    arrays = [contexts, futures, timestamps, sample_dates]
    if cutoff_wmp is not None:
        arrays.append(cutoff_wmp)
    for array in arrays:
        array.flush()
    del contexts, futures, timestamps, sample_dates, cutoff_wmp
    manifest = {
        "format_version": 1,
        "format": "timesfm-ft-npy-bundle",
        "product": product,
        "split": split,
        "source_root": source_root,
        "schema": {
            "context_values": ["float32", total_samples, context_length],
            "future_values": ["float32", total_samples, horizon_length],
            "timestamps": ["int64", total_samples],
            "dates": ["int32", total_samples],
            **(
                {"cutoff_wmp": ["float32", total_samples]}
                if target_mode == "delta_ticks"
                else {}
            ),
        },
        "date_count": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "date_file": str(dates_path),
        "date_file_sha256": _date_file_sha256(dates_path),
        "sampling_interval_seconds": interval_ns / 1_000_000_000,
        "target_mode": target_mode,
        "tick_size": tick_size,
        "source_levels_per_sample": (
            context_length + horizon_length + 1
            if target_mode == "delta_ticks"
            else context_length + horizon_length
        ),
        "context_length": context_length,
        "context_seconds": context_length * interval_ns / 1_000_000_000,
        "horizon_length": horizon_length,
        "horizon_seconds": horizon_length * interval_ns / 1_000_000_000,
        "stride": stride,
        "samples": total_samples,
        "samples_by_day": sample_counts,
        "all_finite": True,
        "session": {
            "timezone": str(SESSION_TIMEZONE),
            "start": SESSION_OPEN.isoformat(),
            "end": SESSION_CLOSE.isoformat(),
            "early_closes_allowed": True,
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    LOGGER.info(
        "wrote product=%s split=%s samples=%d path=%s",
        product,
        split,
        total_samples,
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=("ZN", "ES", "all"), default="all")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "splits",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--sampling-interval-seconds", type=float, default=0.5)
    parser.add_argument(
        "--target-mode",
        choices=("level", "delta_ticks"),
        default="delta_ticks",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        help="Override the product tick size.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if min(args.context_length, args.horizon_length, args.stride) <= 0:
        parser.error("context length, horizon length, and stride must be positive")
    if args.sampling_interval_seconds <= 0:
        parser.error("sampling interval must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    split_dates = audit_splits(args.split_dir)
    products = ("ZN", "ES") if args.product == "all" else (args.product,)
    interval_ns = round(args.sampling_interval_seconds * 1_000_000_000)
    for product in products:
        tick_size = args.tick_size or PRODUCT_TICK_SIZES[product]
        product_dir = args.output_root / f"{product.lower()}-wmp-500ms" / "splits"
        for split in SPLITS:
            dates, dates_path = split_dates[split]
            destination = product_dir / (
                f"{split}_delta_c{args.context_length}_h{args.horizon_length}"
                if args.target_mode == "delta_ticks"
                else f"{split}_c{args.context_length}_h{args.horizon_length}"
            )
            if args.skip_existing and destination.exists():
                LOGGER.info("skip existing product=%s split=%s", product, split)
                continue
            build_split(
                product=product,
                split=split,
                dates=dates,
                dates_path=dates_path,
                source_root=args.source_root,
                destination=destination,
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                stride=args.stride,
                interval_ns=interval_ns,
                target_mode=args.target_mode,
                tick_size=tick_size,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
