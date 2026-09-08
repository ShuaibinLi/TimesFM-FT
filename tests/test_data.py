from __future__ import annotations

import json

import numpy as np
import pytest

from timesfm_ft.data import NpzWindowDataset


def test_loads_single_and_multi_input_contract(tmp_path):
    future = np.ones((3, 6), dtype=np.float32)
    for variates in (1, 4):
        path = tmp_path / f"v{variates}.npz"
        context = np.ones((3, variates, 16), dtype=np.float32)
        np.savez(path, context_values=context, future_values=future)
        dataset = NpzWindowDataset(
            path,
            context_length=16,
            horizon_length=6,
            max_variates=4,
        )
        assert len(dataset) == 3
        assert dataset.num_variates == variates
        assert dataset[0]["context_values"].shape == (variates, 16)
        assert dataset[0]["future_values"].shape == (6,)


def test_nonfinite_values_become_masked(tmp_path):
    context = np.ones((2, 1, 16), dtype=np.float32)
    context[0, 0, 3] = np.nan
    future = np.ones((2, 6), dtype=np.float32)
    path = tmp_path / "masked.npz"
    np.savez(path, context_values=context, future_values=future)

    dataset = NpzWindowDataset(
        path,
        context_length=16,
        horizon_length=6,
    )
    assert dataset[0]["context_mask"][0, 3]
    assert dataset[0]["context_values"][0, 3] == 0


def test_rejects_missing_cutoff_price(tmp_path):
    context = np.ones((2, 1, 16), dtype=np.float32)
    context[0, 0, -1] = np.nan
    future = np.ones((2, 6), dtype=np.float32)
    path = tmp_path / "bad.npz"
    np.savez(path, context_values=context, future_values=future)

    with pytest.raises(ValueError, match="forecast cutoff"):
        NpzWindowDataset(path, context_length=16, horizon_length=6)


def test_loads_memory_mapped_bundle_and_validates_metadata(tmp_path):
    bundle = tmp_path / "train"
    bundle.mkdir()
    context = np.ones((3, 16), dtype=np.float32)
    future = np.ones((3, 6), dtype=np.float32)
    timestamps = np.array([1_000, 2_000, 3_000], dtype=np.int64)
    dates = np.full(3, 20250102, dtype=np.int32)
    for name, value in {
        "context_values": context,
        "future_values": future,
        "timestamps": timestamps,
        "dates": dates,
    }.items():
        np.save(bundle / f"{name}.npy", value)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "product": "ZN",
                "split": "train",
                "context_length": 16,
                "horizon_length": 6,
                "stride": 1,
                "sampling_interval_seconds": 0.000001,
            }
        )
    )

    dataset = NpzWindowDataset(
        bundle,
        context_length=16,
        horizon_length=6,
        sampling_interval_seconds=0.000001,
        expected_stride=1,
        expected_product="ZN",
        expected_split="train",
        expected_dates={20250102},
        require_metadata=True,
    )
    assert len(dataset) == 3
    root = dataset.context_values
    memory_mapped = isinstance(root, np.memmap)
    while getattr(root, "base", None) is not None:
        root = root.base
        memory_mapped = memory_mapped or isinstance(root, np.memmap)
    assert memory_mapped


def test_rejects_metadata_mismatch(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(
        path,
        context_values=np.ones((2, 16), dtype=np.float32),
        future_values=np.ones((2, 6), dtype=np.float32),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "product": "ES",
                "split": "train",
                "context_length": 16,
                "horizon_length": 6,
                "stride": 1,
                "sampling_interval_seconds": 0.5,
            }
        )
    )
    with pytest.raises(ValueError, match="metadata product"):
        NpzWindowDataset(
            path,
            context_length=16,
            horizon_length=6,
            expected_product="ZN",
            require_metadata=False,
        )
