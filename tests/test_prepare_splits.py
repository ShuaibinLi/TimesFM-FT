from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timesfm_ft.data import IntradayWindowDataset

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_intraday_splits",
    ROOT / "scripts/prepare_intraday_splits.py",
)
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_prepare_builds_audited_minute_bundle(tmp_path):
    source = tmp_path / "source"
    dates = (20250102, 20250103)
    minutes = 8
    for date_value in dates:
        day_dir = source / f"date={date_value}"
        day_dir.mkdir(parents=True)
        start = datetime.strptime(str(date_value), "%Y%m%d").replace(tzinfo=timezone.utc)
        timestamps = np.asarray(
            [int((start + timedelta(minutes=index)).timestamp() * 1e9) for index in range(minutes)],
            dtype=np.int64,
        )
        target = np.arange(minutes, dtype=np.float32)
        if date_value == dates[0]:
            target[4] = np.nan
        pq.write_table(
            pa.table(
                {
                    "timestamp_ns": timestamps,
                    "return_1m": target,
                    "feature": np.linspace(0, 1, minutes, dtype=np.float32),
                }
            ),
            day_dir / "part0.parquet",
        )
    dates_path = tmp_path / "dates.txt"
    dates_path.write_text("20250102\n20250103\n")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "dataset_id": "prepared",
                "product": "TEST",
                "path_template": "date={date}/*.parquet",
                "timestamp_column": "timestamp_ns",
                "target": {
                    "name": "return_1m",
                    "column": "return_1m",
                    "unit": "ticks",
                    "price_source": "wmp",
                    "return_type": "simple",
                    "timestamp_semantics": "bar_end",
                    "availability_lag_minutes": 0,
                    "missing_policy": "mask",
                },
                "session": {
                    "timezone": "UTC",
                    "first_bar_time": "00:00:00",
                    "minutes": minutes,
                },
                "past_only_features": [
                    {
                        "name": "feature",
                        "column": "feature",
                        "family": "state",
                        "availability_lag_minutes": 1,
                    }
                ],
                "past_future_features": [{"name": "tod", "kind": "sin_time_of_day"}],
            }
        )
    )
    destination = tmp_path / "bundle"
    schema = prepare._load_schema(schema_path)
    prepare.build_split(
        schema=schema,
        schema_path=schema_path,
        source_root=str(source),
        dates_path=dates_path,
        split="train",
        destination=destination,
        overwrite=False,
    )
    dataset = IntradayWindowDataset(
        destination,
        context_min=3,
        context_max=4,
        horizon_length=2,
        stride=1,
        past_only_features=("feature",),
        past_future_features=("tod",),
        expected_split="train",
        expected_dataset_id="prepared",
        expected_dates_path=dates_path,
    )
    assert len(dataset) == 7
    sample = dataset[0]
    assert sample["past_future_values"].shape == (1, 5)
    assert sample["context_mask"][1, 0]
    assert sample["context_values"][1, 1] == 0.0
    assert sample["unknown_future_mask"][0].tolist() == [False, True]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(manifest["source_files"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["source_files"])
    assert len(manifest["source_snapshot_sha256"]) == 64
    assert len(manifest["preparer_sha256"]) == 64

    original_manifest = (destination / "manifest.json").read_bytes()
    bad_part = source / f"date={dates[0]}" / "part0.parquet"
    bad_table = pq.read_table(bad_part)
    bad_timestamps = bad_table.column("timestamp_ns").to_numpy().copy()
    bad_timestamps[0] += 1
    pq.write_table(
        bad_table.set_column(
            0,
            "timestamp_ns",
            pa.array(bad_timestamps),
        ),
        bad_part,
    )
    with pytest.raises(ValueError, match="frozen"):
        prepare.build_split(
            schema=schema,
            schema_path=schema_path,
            source_root=str(source),
            dates_path=dates_path,
            split="train",
            destination=destination,
            overwrite=True,
        )
    assert (destination / "manifest.json").read_bytes() == original_manifest


def test_prepare_derives_trailing_tick_return_from_causal_wmid_grid(tmp_path):
    source = tmp_path / "source"
    date_value = 20250102
    day_dir = source / f"date={date_value}"
    day_dir.mkdir(parents=True)
    midnight = datetime.strptime(str(date_value), "%Y%m%d").replace(tzinfo=timezone.utc)
    boundaries = np.asarray(
        [int((midnight + timedelta(minutes=index)).timestamp() * 1e9) for index in range(1, 6)],
        dtype=np.int64,
    )
    raw_hwts = boundaries - 1
    tick_size = 0.015625
    wmid = 100.0 + np.arange(5, dtype=np.float64) * tick_size
    feature = np.arange(5, dtype=np.float32)
    pq.write_table(
        pa.table({"hwts": raw_hwts, "wmid": wmid, "feature": feature}),
        day_dir / "part0.parquet",
    )
    dates_path = tmp_path / "dates.txt"
    dates_path.write_text(f"{date_value}\n")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "dataset_id": "derived",
                "product": "ZN",
                "path_template": "date={date}/*.parquet",
                "timestamp_column": "timestamp_ns",
                "target": {
                    "name": "return_1m",
                    "column": "return_1m",
                    "unit": "ticks",
                    "price_source": "WMid",
                    "return_type": "tick_displacement",
                    "timestamp_semantics": "bar_end",
                    "availability_lag_minutes": 0,
                    "missing_policy": "mask",
                    "derivation": {
                        "kind": "trailing_price_difference_ticks",
                        "price_column": "wmid",
                        "tick_size": tick_size,
                        "timestamp_source_column": "hwts",
                        "interval_minutes": 1,
                    },
                },
                "session": {
                    "timezone": "UTC",
                    "first_bar_time": "00:02:00",
                    "minutes": 4,
                },
                "past_only_features": [
                    {
                        "name": "feature",
                        "column": "feature",
                        "family": "state",
                        "availability_lag_minutes": 0,
                    }
                ],
                "past_future_features": [],
            }
        )
    )
    destination = tmp_path / "bundle"
    prepare.build_split(
        schema=prepare._load_schema(schema_path),
        schema_path=schema_path,
        source_root=str(source),
        dates_path=dates_path,
        split="train",
        destination=destination,
        overwrite=False,
    )

    np.testing.assert_array_equal(
        np.load(destination / "timestamps.npy")[0],
        boundaries[1:],
    )
    np.testing.assert_allclose(
        np.load(destination / "target_values.npy")[0],
        np.ones(4, dtype=np.float32),
    )
    assert not np.load(destination / "target_mask.npy")[0].any()
    np.testing.assert_array_equal(
        np.load(destination / "past_only_values.npy")[0, 0],
        feature[1:],
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["target"]["derivation"]["kind"] == "trailing_price_difference_ticks"


def test_split_audit_rejects_overlap(tmp_path):
    (tmp_path / "dates-train.txt").write_text("20250102\n20250103\n")
    (tmp_path / "dates-val.txt").write_text("20250103\n20250104\n")
    (tmp_path / "dates-test.txt").write_text("20250105\n")
    with pytest.raises(ValueError, match="overlap"):
        prepare.audit_split_files(tmp_path)
