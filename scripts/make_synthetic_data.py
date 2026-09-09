#!/usr/bin/env python3
"""Generate small audited intraday bundles for smoke tests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from timesfm_ft.data import BUNDLE_FORMAT, BUNDLE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
PAST_ONLY = (
    "momentum_5m",
    "realized_vol_15m",
    "book_imbalance",
    "market_return_1m",
)
PAST_FUTURE = ("sin_time_of_day", "cos_time_of_day", "time_to_close")


def _dates(count: int, start: datetime) -> tuple[list[int], datetime]:
    result: list[int] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(int(current.strftime("%Y%m%d")))
        current += timedelta(days=1)
    return result, current


def _write_split(
    destination: Path,
    *,
    split: str,
    dates: list[int],
    session_minutes: int,
    seed: int,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = len(dates)
    noise = rng.normal(0.0, 0.35, size=(days, session_minutes)).astype(np.float32)
    target = np.empty_like(noise)
    target[:, 0] = noise[:, 0]
    for minute in range(1, session_minutes):
        target[:, minute] = 0.15 * target[:, minute - 1] + noise[:, minute]
    index = np.arange(session_minutes, dtype=np.float32)
    past = np.empty((days, len(PAST_ONLY), session_minutes), dtype=np.float32)
    past[:, 0] = np.cumsum(target, axis=1)
    past[:, 1] = np.sqrt(
        np.maximum(
            np.cumsum(target**2, axis=1) / np.arange(1, session_minutes + 1, dtype=np.float32),
            1e-8,
        )
    )
    past[:, 2] = rng.normal(size=(days, session_minutes))
    past[:, 3] = 0.4 * target + rng.normal(0.0, 0.4, size=(days, session_minutes))
    known = np.stack(
        (
            np.sin(2 * np.pi * index / session_minutes),
            np.cos(2 * np.pi * index / session_minutes),
            (session_minutes - 1 - index) / max(session_minutes - 1, 1),
        )
    ).astype(np.float32)
    known = np.broadcast_to(known[None, :, :], (days, *known.shape)).copy()
    first_timestamp = int(
        datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc).timestamp() * 1_000_000_000
    )
    timestamps = np.asarray(
        [
            [
                first_timestamp + day_index * 86_400_000_000_000 + minute * 60_000_000_000
                for minute in range(session_minutes)
            ]
            for day_index in range(days)
        ],
        dtype=np.int64,
    )
    arrays = {
        "target_values": target,
        "target_mask": np.zeros_like(target, dtype=np.bool_),
        "past_only_values": past,
        "past_only_mask": np.zeros_like(past, dtype=np.bool_),
        "past_future_values": known,
        "past_future_mask": np.zeros_like(known, dtype=np.bool_),
        "timestamps": timestamps,
        "dates": np.asarray(dates, dtype=np.int32),
        "session_lengths": np.full(days, session_minutes, dtype=np.int16),
    }
    for name, values in arrays.items():
        np.save(destination / f"{name}.npy", values, allow_pickle=False)
    dates_path = destination.parent / "reference" / f"dates-{split}.txt"
    dates_path.parent.mkdir(parents=True, exist_ok=True)
    dates_path.write_text(
        "".join(f"{value}\n" for value in dates),
        encoding="utf-8",
    )
    import hashlib

    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "dataset_id": "synthetic_intraday_1min_v1",
        "product": "SYNTH",
        "split": split,
        "target_name": "return_1m",
        "target": {
            "name": "return_1m",
            "unit": "synthetic",
            "price_source": "synthetic",
            "return_type": "simple",
            "timestamp_semantics": "bar_end",
            "availability_lag_minutes": 0,
        },
        "frequency_minutes": 1,
        "session_minutes": session_minutes,
        "past_only_features": list(PAST_ONLY),
        "past_future_features": list(PAST_FUTURE),
        "date_file_sha256": hashlib.sha256(dates_path.read_bytes()).hexdigest(),
        "schema": {name: [str(values.dtype), *values.shape] for name, values in arrays.items()},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dates_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/synthetic-intraday-1min",
    )
    parser.add_argument("--session-minutes", type=int, default=160)
    parser.add_argument("--train-days", type=int, default=4)
    parser.add_argument("--val-days", type=int, default=2)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    current = datetime(2026, 1, 2)
    for offset, (split, count) in enumerate(
        (
            ("train", args.train_days),
            ("val", args.val_days),
            ("test", args.test_days),
        )
    ):
        dates, current = _dates(count, current)
        _write_split(
            args.output_root / split,
            split=split,
            dates=dates,
            session_minutes=args.session_minutes,
            seed=args.seed + offset,
        )


if __name__ == "__main__":
    main()
