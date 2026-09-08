from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timesfm_ft.data import NpzWindowDataset


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_single_product_splits.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_single_product_splits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_split_reads_multiple_parts_and_writes_audited_bundle(tmp_path):
    module = _module()
    day = "20250102"
    source_root = tmp_path / "source"
    partition = source_root / "ZN" / f"date={day}"
    partition.mkdir(parents=True)
    start = int(
        datetime(2025, 1, 2, 9, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
        * 1_000_000_000
    )
    timestamps = start + np.arange(25_200, dtype=np.int64) * 500_000_000
    wmp = 110.0 + np.arange(25_200, dtype=np.float64) * 0.0001
    table = pa.table({"timestamp_ns": timestamps, "wmp": wmp})
    pq.write_table(table.slice(0, 12_600), partition / "part0.parquet")
    pq.write_table(table.slice(12_600), partition / "part1.parquet")
    date_file = tmp_path / "dates-train.txt"
    date_file.write_text(f"{day}\n")
    destination = tmp_path / "train_c16_h8"

    module.build_split(
        product="ZN",
        split="train",
        dates=[day],
        dates_path=date_file,
        source_root=str(source_root),
        destination=destination,
        context_length=16,
        horizon_length=8,
        stride=8,
        interval_ns=500_000_000,
    )

    dataset = NpzWindowDataset(
        destination,
        context_length=16,
        horizon_length=8,
        sampling_interval_seconds=0.5,
        expected_stride=8,
        expected_product="ZN",
        expected_split="train",
        expected_dates={20250102},
        expected_dates_path=date_file,
        require_metadata=True,
    )
    assert len(dataset) == len(range(15, 25_192, 8))
    assert dataset.metadata is not None
    assert dataset.metadata["date_file_sha256"]


def test_session_validation_rejects_truncated_open():
    module = _module()
    start = int(
        datetime(2025, 1, 2, 12, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        * 1_000_000_000
    )
    timestamps = start + np.arange(21_600, dtype=np.int64) * 500_000_000
    with pytest.raises(ValueError, match="starts too late"):
        module._validate_day(
            timestamps,
            np.ones(len(timestamps)),
            product="ZN",
            day="20250102",
            interval_ns=500_000_000,
            minimum_rows=320,
        )
