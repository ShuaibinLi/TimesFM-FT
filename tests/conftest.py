from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for source_path in (ROOT / "src", ROOT / "3rdparty" / "timesfm" / "src"):
    sys.path.insert(0, str(source_path))

from timesfm_ft.data import BUNDLE_FORMAT, BUNDLE_VERSION  # noqa: E402


@pytest.fixture
def bundle_factory(tmp_path):
    def write(
        name: str,
        *,
        split: str = "train",
        days: int = 2,
        minutes: int = 12,
        past_names: tuple[str, ...] = ("p1", "p2"),
        known_names: tuple[str, ...] = ("tod",),
        start_date: int = 20250102,
    ):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        target = np.arange(days * minutes, dtype=np.float32).reshape(days, minutes)
        past = np.stack(
            [target + offset for offset in range(len(past_names))],
            axis=1,
        )
        known = np.stack(
            [
                np.broadcast_to(
                    np.linspace(0, 1, minutes, dtype=np.float32),
                    (days, minutes),
                )
                + offset
                for offset in range(len(known_names))
            ],
            axis=1,
        )
        base = 1_700_000_000_000_000_000
        timestamps = np.stack(
            [
                base
                + day * 86_400_000_000_000
                + np.arange(minutes, dtype=np.int64) * 60_000_000_000
                for day in range(days)
            ]
        )
        dates = np.arange(start_date, start_date + days, dtype=np.int32)
        arrays = {
            "target_values": target,
            "target_mask": np.zeros_like(target, dtype=np.bool_),
            "past_only_values": past,
            "past_only_mask": np.zeros_like(past, dtype=np.bool_),
            "past_future_values": known,
            "past_future_mask": np.zeros_like(known, dtype=np.bool_),
            "timestamps": timestamps,
            "dates": dates,
            "session_lengths": np.full(days, minutes, dtype=np.int16),
        }
        for array_name, values in arrays.items():
            np.save(path / f"{array_name}.npy", values)
        dates_path = tmp_path / f"dates-{name}.txt"
        dates_path.write_text(
            "".join(f"{value}\n" for value in dates),
            encoding="utf-8",
        )
        manifest = {
            "format": BUNDLE_FORMAT,
            "format_version": BUNDLE_VERSION,
            "dataset_id": "test_intraday",
            "product": "TEST",
            "split": split,
            "target_name": "return_1m",
            "target": {
                "name": "return_1m",
                "unit": "test",
                "price_source": "test",
                "return_type": "simple",
                "timestamp_semantics": "bar_end",
                "availability_lag_minutes": 0,
                "missing_policy": "mask",
            },
            "frequency_minutes": 1,
            "session_minutes": minutes,
            "past_only_features": list(past_names),
            "past_only_availability_lag_minutes": {name: 0 for name in past_names},
            "past_future_features": list(known_names),
            "date_file_sha256": hashlib.sha256(dates_path.read_bytes()).hexdigest(),
            "schema": {
                array_name: [str(values.dtype), *values.shape]
                for array_name, values in arrays.items()
            },
        }
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return path, dates_path

    return write
