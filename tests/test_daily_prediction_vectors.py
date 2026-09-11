from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_daily_prediction_vectors",
    ROOT / "scripts/build_daily_prediction_vectors.py",
)
assert SPEC and SPEC.loader
daily = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(daily)


def test_builds_one_unique_lead_one_point_per_target_minute(tmp_path):
    dates = np.asarray([20250102] * 3 + [20250103] * 3, dtype=np.int32)
    timestamps = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int64) * daily.MINUTE_NS
    targets = np.arange(12, dtype=np.float32).reshape(6, 2)
    predictions = np.zeros((6, 2, 3), dtype=np.float32)
    predictions[:, :, 1] = targets
    target_mask = np.zeros((6, 2), dtype=np.bool_)
    target_mask[-1, 0] = True
    source = tmp_path / "predictions.npz"
    np.savez(
        source,
        predictions=predictions,
        targets=targets,
        target_mask=target_mask,
        quantiles=np.asarray([0.1, 0.5, 0.9]),
        dates=dates,
        timestamps=timestamps,
        minute_indices=np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int16),
        context_lengths=np.full(6, 64, dtype=np.int16),
    )

    output = tmp_path / "daily"
    summary = daily.build_daily_vectors(source, output_dir=output)
    assert summary["points"] == 5
    assert summary["overall_ic"] == 1.0
    assert summary["direction_accuracy"] == 1.0
    payload = np.load(output / "daily_vectors.npz")
    np.testing.assert_array_equal(payload["lengths"], np.asarray([3, 2]))
    assert payload["predictions"].shape == (2, 3)
    assert not payload["target_mask"][0].any()
    assert payload["target_mask"][1].tolist() == [False, False, True]


def test_tiles_non_overlapping_forecast_blocks(tmp_path):
    source = tmp_path / "predictions.npz"
    targets = np.arange(8, dtype=np.float32).reshape(4, 2)
    predictions = np.zeros((4, 2, 3), dtype=np.float32)
    predictions[:, :, 1] = targets
    np.savez(
        source,
        predictions=predictions,
        targets=targets,
        target_mask=np.zeros((4, 2), dtype=np.bool_),
        quantiles=np.asarray([0.1, 0.5, 0.9]),
        dates=np.full(4, 20250102, dtype=np.int32),
        timestamps=np.arange(1, 5, dtype=np.int64) * daily.MINUTE_NS,
        minute_indices=np.arange(1, 5, dtype=np.int16),
        context_lengths=np.full(4, 64, dtype=np.int16),
    )
    output = tmp_path / "blocks"
    summary = daily.build_daily_vectors(
        source,
        output_dir=output,
        mode="blocks",
        block_size=2,
    )
    assert summary["points"] == 4
    assert summary["overall_ic"] == 1.0
    payload = np.load(output / "daily_vectors.npz")
    np.testing.assert_array_equal(payload["lengths"], np.asarray([4]))
    np.testing.assert_array_equal(payload["leads"][0], np.asarray([1, 2, 1, 2]))
    assert len(np.unique(payload["target_timestamps"][0])) == 4
