#!/usr/bin/env python3
"""Freeze chronological train/validation/test dates for selected100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent / "spine" / "configs" / "wmp_500ms_dates_20221101_20260130.txt"
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datas" / "zn_rank_selected100_1min_20221101_20260130"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "splits" / "zn-rank-selected100"


def _dates(path: Path) -> list[int]:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(values) != len(set(values)) or values != sorted(values):
        raise ValueError("source dates must be unique and chronological")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--required-raw-rows", type=int, default=395)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1:
        parser.error("split fractions must be in (0, 1)")
    if args.train_fraction + args.val_fraction >= 1:
        parser.error("train_fraction + val_fraction must be below 1")

    source_dates = _dates(args.source)
    dates: list[int] = []
    excluded: list[dict[str, int | str]] = []
    for date_value in source_dates:
        parts = sorted((args.data_root / f"date={date_value}").glob("part*.parquet"))
        if not parts:
            raise FileNotFoundError(f"missing raw Parquet for date={date_value}")
        rows = sum(pq.ParquetFile(part).metadata.num_rows for part in parts)
        if rows == args.required_raw_rows:
            dates.append(date_value)
        else:
            excluded.append(
                {
                    "date": date_value,
                    "raw_rows": rows,
                    "reason": "non_full_session",
                }
            )
    train_end = int(len(dates) * args.train_fraction)
    val_end = train_end + int(len(dates) * args.val_fraction)
    splits = {
        "train": dates[:train_end],
        "val": dates[train_end:val_end],
        "test": dates[val_end:],
    }
    if min(map(len, splits.values())) == 0:
        raise ValueError("each chronological split must contain dates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (args.output_dir / f"dates-{name}.txt").write_text(
            "".join(f"{value}\n" for value in values),
            encoding="utf-8",
        )
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "data_root": str(args.data_root.resolve()),
        "source_dates": len(source_dates),
        "eligible_full_session_dates": len(dates),
        "required_raw_rows": args.required_raw_rows,
        "excluded_dates": excluded,
        "policy": "chronological_by_trading_date",
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        "splits": {
            name: {
                "dates": len(values),
                "first": values[0],
                "last": values[-1],
            }
            for name, values in splits.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
