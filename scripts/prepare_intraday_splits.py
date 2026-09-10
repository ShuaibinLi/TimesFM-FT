#!/usr/bin/env python3
"""Convert frozen 1-minute Parquet data into audited session bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timesfm_ft.data import BUNDLE_FORMAT, BUNDLE_VERSION, read_dates

LOGGER = logging.getLogger("prepare_intraday_splits")
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def audit_split_files(dates_dir: Path) -> dict[str, Path]:
    paths = {split: dates_dir / f"dates-{split}.txt" for split in SPLITS}
    values = {split: read_dates(path) for split, path in paths.items()}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = set(values[left]) & set(values[right])
            if overlap:
                raise ValueError(f"{left}/{right} date overlap: {sorted(overlap)[:5]}")
    if not (values["train"][-1] < values["val"][0] and values["val"][-1] < values["test"][0]):
        raise ValueError("splits must be strictly chronological")
    return paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    required = {
        "dataset_id",
        "product",
        "path_template",
        "timestamp_column",
        "target",
        "session",
        "past_only_features",
        "past_future_features",
    }
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"schema is missing fields: {missing}")
    target_required = {
        "name",
        "column",
        "unit",
        "price_source",
        "return_type",
        "timestamp_semantics",
        "availability_lag_minutes",
        "missing_policy",
    }
    missing_target = sorted(target_required - set(schema["target"]))
    if missing_target:
        raise ValueError(f"target definition is incomplete: {missing_target}")
    if schema["target"]["missing_policy"] != "mask":
        raise ValueError("v1.3 requires target missing_policy='mask'")
    if schema["target"]["timestamp_semantics"] != "bar_end":
        raise ValueError("v1.3 window alignment requires bar_end targets")
    if schema["target"]["availability_lag_minutes"] != 0:
        raise ValueError("v1.3 requires target availability_lag_minutes=0")
    derivation = schema["target"].get("derivation")
    if derivation is not None:
        required_derivation = {
            "kind",
            "price_column",
            "tick_size",
            "timestamp_source_column",
            "interval_minutes",
        }
        missing_derivation = sorted(required_derivation - set(derivation))
        if missing_derivation:
            raise ValueError(f"target derivation is incomplete: {missing_derivation}")
        if derivation["kind"] != "trailing_price_difference_ticks":
            raise ValueError(f"unsupported target derivation kind={derivation['kind']!r}")
        if not isinstance(derivation["tick_size"], (int, float)) or derivation["tick_size"] <= 0:
            raise ValueError("target derivation tick_size must be positive")
        if derivation["interval_minutes"] != 1:
            raise ValueError("intraday v1.3 supports only one-minute target derivation")
        for key in ("price_column", "timestamp_source_column"):
            if not isinstance(derivation[key], str) or not derivation[key]:
                raise ValueError(f"target derivation {key} must be a non-empty string")
    for entry in schema["past_only_features"]:
        required_feature = {
            "name",
            "column",
            "family",
            "availability_lag_minutes",
        }
        missing_feature = sorted(required_feature - set(entry))
        if missing_feature:
            raise ValueError(f"past-only feature definition is incomplete: {missing_feature}")
        if (
            not isinstance(entry["availability_lag_minutes"], int)
            or entry["availability_lag_minutes"] < 0
        ):
            raise ValueError("feature availability lag must be a non-negative integer")
    past_names = [entry["name"] for entry in schema["past_only_features"]]
    known_names = [entry["name"] for entry in schema["past_future_features"]]
    if len(past_names) != len(set(past_names)):
        raise ValueError("past-only feature names must be unique")
    if len(known_names) != len(set(known_names)):
        raise ValueError("past-future feature names must be unique")
    if set(past_names) & set(known_names):
        raise ValueError("past-only and past-future names must be disjoint")
    return schema


def _parts_for_date(
    fs: fsspec.AbstractFileSystem,
    source_path: str,
    template: str,
    date: int,
) -> list[str]:
    relative = template.format(date=date).lstrip("/")
    pattern = f"{source_path.rstrip('/')}/{relative}"
    parts = sorted(fs.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"no Parquet files matched {pattern}")
    return parts


def _read_date(
    fs: fsspec.AbstractFileSystem,
    parts: list[str],
    columns: list[str],
) -> tuple[pa.Table, list[dict[str, Any]]]:
    tables = []
    records: list[dict[str, Any]] = []
    for part in parts:
        with fs.open(part, "rb") as stream:
            payload = stream.read()
        tables.append(pq.read_table(pa.BufferReader(payload), columns=columns))
        records.append(
            {
                "path": part,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return table, records


def _expected_timestamps(
    date_value: int,
    *,
    timezone: ZoneInfo,
    first_bar_time: time,
    minutes: int,
) -> np.ndarray:
    day = datetime.strptime(str(date_value), "%Y%m%d").date()
    first = datetime.combine(day, first_bar_time, tzinfo=timezone)
    return np.asarray(
        [
            int((first + timedelta(minutes=index)).timestamp() * 1_000_000_000)
            for index in range(minutes)
        ],
        dtype=np.int64,
    )


def _derived_known_feature(kind: str, minutes: int) -> np.ndarray:
    index = np.arange(minutes, dtype=np.float64)
    if kind == "sin_time_of_day":
        return np.sin(2.0 * np.pi * index / minutes).astype(np.float32)
    if kind == "cos_time_of_day":
        return np.cos(2.0 * np.pi * index / minutes).astype(np.float32)
    if kind == "time_to_close":
        return ((minutes - 1 - index) / max(minutes - 1, 1)).astype(np.float32)
    raise ValueError(f"unsupported past-future feature kind={kind!r}")


def _derive_trailing_tick_return(
    *,
    raw_timestamp: np.ndarray,
    price: np.ndarray,
    expected_timestamp: np.ndarray,
    tick_size: float,
    interval_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    interval_ns = interval_minutes * 60 * 1_000_000_000
    if raw_timestamp.ndim != 1 or price.shape != raw_timestamp.shape:
        raise ValueError("raw timestamp and price must be aligned one-dimensional arrays")
    if len(raw_timestamp) < len(expected_timestamp) + 1:
        raise ValueError("derived return requires at least one pre-session warmup row")
    if np.any(raw_timestamp[1:] <= raw_timestamp[:-1]):
        raise ValueError("raw target timestamps must be strictly increasing")
    bar_end = ((raw_timestamp + interval_ns - 1) // interval_ns) * interval_ns
    if np.any(raw_timestamp >= bar_end):
        raise ValueError("raw row must be strictly before its causal bar-end timestamp")
    if np.any(bar_end[1:] <= bar_end[:-1]):
        raise ValueError("derived bar-end timestamps must be unique and increasing")

    positions = np.searchsorted(bar_end, expected_timestamp)
    if np.any(positions >= len(bar_end)) or not np.array_equal(
        bar_end[positions],
        expected_timestamp,
    ):
        raise ValueError("raw rows do not cover the frozen model-minute grid")
    if np.any(positions == 0):
        raise ValueError("first model minute has no preceding warmup price")

    trailing_return = np.full(len(price), np.nan, dtype=np.float64)
    trailing_return[1:] = (price[1:] - price[:-1]) / tick_size
    return trailing_return[positions], positions


def _open_memmaps(
    destination: Path,
    *,
    days: int,
    minutes: int,
    past_only: int,
    past_future: int,
) -> dict[str, np.memmap]:
    shapes: dict[str, tuple[int, ...]] = {
        "target_values": (days, minutes),
        "target_mask": (days, minutes),
        "past_only_values": (days, past_only, minutes),
        "past_only_mask": (days, past_only, minutes),
        "past_future_values": (days, past_future, minutes),
        "past_future_mask": (days, past_future, minutes),
        "timestamps": (days, minutes),
        "dates": (days,),
        "session_lengths": (days,),
    }
    dtypes = {
        "target_values": np.float32,
        "target_mask": np.bool_,
        "past_only_values": np.float32,
        "past_only_mask": np.bool_,
        "past_future_values": np.float32,
        "past_future_mask": np.bool_,
        "timestamps": np.int64,
        "dates": np.int32,
        "session_lengths": np.int16,
    }
    return {
        name: np.lib.format.open_memmap(
            destination / f"{name}.npy",
            mode="w+",
            dtype=dtypes[name],
            shape=shape,
        )
        for name, shape in shapes.items()
    }


def build_split(
    *,
    schema: dict[str, Any],
    schema_path: Path,
    source_root: str,
    dates_path: Path,
    split: str,
    destination: Path,
    overwrite: bool,
) -> None:
    dates = read_dates(dates_path)
    session = schema["session"]
    minutes = int(session["minutes"])
    timezone = ZoneInfo(session["timezone"])
    first_bar_time = time.fromisoformat(session["first_bar_time"])
    timestamp_column = schema["timestamp_column"]
    target_column = schema["target"]["column"]
    target_derivation = schema["target"].get("derivation")
    past_columns = [entry["column"] for entry in schema["past_only_features"]]
    past_lags = [entry["availability_lag_minutes"] for entry in schema["past_only_features"]]
    if target_derivation is None:
        raw_timestamp_column = timestamp_column
        source_columns = [timestamp_column, target_column, *past_columns]
    else:
        raw_timestamp_column = target_derivation["timestamp_source_column"]
        source_columns = [
            raw_timestamp_column,
            target_derivation["price_column"],
            *past_columns,
        ]
    if len(source_columns) != len(set(source_columns)):
        raise ValueError("source target/feature columns must be unique")

    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} exists; pass --overwrite")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    arrays = _open_memmaps(
        temporary,
        days=len(dates),
        minutes=minutes,
        past_only=len(past_columns),
        past_future=len(schema["past_future_features"]),
    )
    fs, source_path = fsspec.core.url_to_fs(source_root.rstrip("/"))
    known_rows = [
        _derived_known_feature(entry["kind"], minutes) for entry in schema["past_future_features"]
    ]
    known_features = (
        np.stack(known_rows, axis=0).astype(np.float32, copy=False)
        if known_rows
        else np.empty((0, minutes), dtype=np.float32)
    )
    target_missing_by_day: dict[str, int] = {}
    feature_missing_counts = np.zeros(len(past_columns), dtype=np.int64)
    source_files: list[dict[str, Any]] = []

    for index, date_value in enumerate(dates):
        parts = _parts_for_date(
            fs,
            source_path,
            schema["path_template"],
            date_value,
        )
        table, date_source_files = _read_date(fs, parts, source_columns)
        source_files.extend(date_source_files)
        raw_timestamp = table.column(raw_timestamp_column).to_numpy(zero_copy_only=False)
        order = np.argsort(raw_timestamp, kind="stable")
        raw_timestamp = np.asarray(raw_timestamp[order], dtype=np.int64)
        expected = _expected_timestamps(
            date_value,
            timezone=timezone,
            first_bar_time=first_bar_time,
            minutes=minutes,
        )
        if target_derivation is None:
            timestamp = raw_timestamp
            row_indices = np.arange(len(timestamp), dtype=np.int64)
            if not np.array_equal(timestamp, expected):
                raise ValueError(f"{date_value} does not match the frozen {minutes}-minute grid")
            target = np.asarray(
                table.column(target_column).to_numpy(zero_copy_only=False)[order],
                dtype=np.float32,
            )
        else:
            price = np.asarray(
                table.column(target_derivation["price_column"]).to_numpy(zero_copy_only=False)[
                    order
                ],
                dtype=np.float64,
            )
            target_values, row_indices = _derive_trailing_tick_return(
                raw_timestamp=raw_timestamp,
                price=price,
                expected_timestamp=expected,
                tick_size=float(target_derivation["tick_size"]),
                interval_minutes=int(target_derivation["interval_minutes"]),
            )
            timestamp = expected
            target = np.asarray(target_values, dtype=np.float32)
        target_mask = ~np.isfinite(target)
        target_missing_by_day[str(date_value)] = int(target_mask.sum())
        arrays["target_values"][index] = np.nan_to_num(
            target,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        arrays["target_mask"][index] = target_mask
        for feature_index, column in enumerate(past_columns):
            values = np.asarray(
                table.column(column).to_numpy(zero_copy_only=False)[order][row_indices],
                dtype=np.float32,
            )
            source_mask = ~np.isfinite(values)
            lag = past_lags[feature_index]
            shifted = np.zeros(minutes, dtype=np.float32)
            mask = np.ones(minutes, dtype=np.bool_)
            if lag == 0:
                shifted[:] = values
                mask[:] = source_mask
            elif lag < minutes:
                shifted[lag:] = values[:-lag]
                mask[lag:] = source_mask[:-lag]
            arrays["past_only_values"][index, feature_index] = np.nan_to_num(
                shifted,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            arrays["past_only_mask"][index, feature_index] = mask
            feature_missing_counts[feature_index] += int(mask.sum())
        arrays["past_future_values"][index] = known_features
        arrays["past_future_mask"][index] = False
        arrays["timestamps"][index] = timestamp
        arrays["dates"][index] = date_value
        arrays["session_lengths"][index] = minutes
        if (index + 1) % 25 == 0 or index + 1 == len(dates):
            LOGGER.info("split=%s prepared=%d/%d", split, index + 1, len(dates))

    for array in arrays.values():
        array.flush()
    schema_description = {name: [str(array.dtype), *array.shape] for name, array in arrays.items()}
    del arrays
    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "dataset_id": schema["dataset_id"],
        "product": schema["product"],
        "split": split,
        "source_root": source_root,
        "path_template": schema["path_template"],
        "source_files": source_files,
        "source_snapshot_sha256": hashlib.sha256(
            json.dumps(
                source_files,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "preparer": Path(__file__).name,
        "preparer_sha256": _sha256(Path(__file__).resolve()),
        "feature_schema": str(schema_path),
        "feature_schema_sha256": _sha256(schema_path),
        "target_name": schema["target"]["name"],
        "target": schema["target"],
        "frequency_minutes": 1,
        "session_minutes": minutes,
        "session": session,
        "past_only_features": [entry["name"] for entry in schema["past_only_features"]],
        "past_only_families": {
            entry["name"]: entry["family"] for entry in schema["past_only_features"]
        },
        "past_only_availability_lag_minutes": {
            entry["name"]: entry["availability_lag_minutes"]
            for entry in schema["past_only_features"]
        },
        "past_future_features": [entry["name"] for entry in schema["past_future_features"]],
        "data_quality": {
            "target_missing_total": sum(target_missing_by_day.values()),
            "target_missing_by_day": target_missing_by_day,
            "past_only_missing_total": {
                name: int(feature_missing_counts[index])
                for index, name in enumerate(
                    entry["name"] for entry in schema["past_only_features"]
                )
            },
        },
        "dates": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "date_file": str(dates_path),
        "date_file_sha256": _sha256(dates_path),
        "schema": schema_description,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        backup = destination.with_name(f".{destination.name}.backup")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except BaseException:
            os.replace(backup, destination)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(temporary, destination)
    LOGGER.info("wrote split=%s bundle=%s", split, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "configs/datasets/intraday_1min_schema.json",
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--dates-dir",
        type=Path,
        default=REPO_ROOT / "configs/splits",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/intraday-1min",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    schema_path = args.schema.resolve()
    schema = _load_schema(schema_path)
    split_paths = audit_split_files(args.dates_dir)
    for split in SPLITS:
        build_split(
            schema=schema,
            schema_path=schema_path,
            source_root=args.source_root,
            dates_path=split_paths[split],
            split=split,
            destination=args.output_root / split,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
