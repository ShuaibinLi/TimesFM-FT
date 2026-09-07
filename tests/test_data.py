from __future__ import annotations

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
