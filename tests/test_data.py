from __future__ import annotations

import numpy as np
import pytest
import torch

from timesfm_ft.data import (
    ContextBucketBatchSampler,
    IntradayWindowDataset,
    collate_intraday_windows,
)


def _dataset(bundle_factory, **overrides):
    path, dates_path = bundle_factory("bundle")
    kwargs = {
        "context_min": 4,
        "context_max": 8,
        "horizon_length": 3,
        "stride": 1,
        "past_only_features": ("p2",),
        "past_future_features": ("tod",),
        "expected_split": "train",
        "expected_dataset_id": "test_intraday",
        "expected_product": "TEST",
        "expected_target_name": "return_1m",
        "expected_frequency_minutes": 1,
        "expected_session_minutes": 12,
        "expected_dates_path": dates_path,
    }
    kwargs.update(overrides)
    return IntradayWindowDataset(path, **kwargs)


def test_dynamic_context_and_future_alignment(bundle_factory):
    dataset = _dataset(bundle_factory)
    assert len(dataset) == 12
    assert dataset.context_lengths.tolist()[:6] == [4, 5, 6, 7, 8, 8]
    sample = dataset[1]
    assert sample["context_length"] == 5
    torch.testing.assert_close(sample["context_values"][0], torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(
        sample["unknown_future_values"][0],
        torch.tensor([5.0, 6.0, 7.0]),
    )
    assert sample["past_future_values"].shape == (1, 8)
    assert sample["unknown_future_values"].shape == (2, 3)
    torch.testing.assert_close(
        sample["unknown_future_values"][1],
        torch.tensor([6.0, 7.0, 8.0]),
    )
    assert sample["date"] == 20250102
    assert dataset[6]["date"] == 20250103


def test_collate_pads_only_to_patch_bucket_and_preserves_known_future(
    bundle_factory,
):
    dataset = _dataset(bundle_factory)
    batch = collate_intraday_windows(
        [dataset[1], dataset[2]],
        patch_length=4,
    )
    assert batch["context_values"].shape == (2, 2, 8)
    assert batch["past_future_values"].shape == (2, 1, 11)
    assert batch["unknown_future_values"].shape == (2, 2, 3)
    assert batch["context_padding_mask"][0, :3].all()
    assert not batch["context_padding_mask"][0, 3:].any()
    assert batch["context_mask"][0, :, :3].all()
    assert not batch["context_mask"][0, :, 3:].any()
    torch.testing.assert_close(
        batch["past_future_values"][0, 0, 8:],
        dataset[1]["past_future_values"][0, 5:],
    )


def test_bucket_sampler_never_mixes_patch_widths(bundle_factory):
    dataset = _dataset(bundle_factory)
    sampler = ContextBucketBatchSampler(
        dataset,
        batch_size=3,
        patch_length=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )
    for indices in sampler:
        widths = {dataset.padded_context_length(index, patch_length=4) for index in indices}
        assert len(widths) == 1


def test_feature_budget_and_unknown_feature_fail_closed(bundle_factory):
    with pytest.raises(ValueError, match="unknown past-only"):
        _dataset(bundle_factory, past_only_features=("missing",))
    with pytest.raises(ValueError, match="exceeds limit"):
        _dataset(
            bundle_factory,
            past_only_features=("p1", "p2"),
            max_variates=2,
        )


def test_target_definition_is_part_of_bundle_identity(bundle_factory):
    path, dates_path = bundle_factory("target-contract")
    with pytest.raises(ValueError, match="target-definition"):
        IntradayWindowDataset(
            path,
            context_min=4,
            context_max=8,
            horizon_length=3,
            stride=1,
            expected_target_unit="wrong-unit",
            expected_dates_path=dates_path,
        )


def test_invalid_future_is_masked_and_invalid_cutoff_is_not_sampled(bundle_factory):
    path, _ = bundle_factory("masked-target")
    mask_path = path / "target_mask.npy"
    mask = np.load(mask_path)
    mask[0, 4] = True
    np.save(mask_path, mask)
    dataset = IntradayWindowDataset(
        path,
        context_min=4,
        context_max=8,
        horizon_length=3,
        stride=1,
    )
    assert len(dataset) == 11
    first = dataset[0]
    assert first["unknown_future_mask"][0].tolist() == [
        True,
        False,
        False,
    ]
    assert first["timestamp"] != int(dataset.timestamps[0, 4])


def test_post_cutoff_past_only_changes_cannot_change_model_inputs(bundle_factory):
    path, _ = bundle_factory("causal")
    before = IntradayWindowDataset(
        path,
        context_min=4,
        context_max=8,
        horizon_length=3,
        stride=1,
        past_only_features=("p1",),
    )[0]["context_values"].clone()
    values_path = path / "past_only_values.npy"
    values = np.load(values_path)
    values[0, :, 4:] = 1_000_000.0
    np.save(values_path, values)
    after = IntradayWindowDataset(
        path,
        context_min=4,
        context_max=8,
        horizon_length=3,
        stride=1,
        past_only_features=("p1",),
    )[0]["context_values"]
    torch.testing.assert_close(after, before)


def test_rejects_non_minute_grid(bundle_factory):
    path, _ = bundle_factory("bad")
    timestamps_path = path / "timestamps.npy"
    timestamps = np.load(timestamps_path)
    timestamps[0, 3] += 1
    np.save(timestamps_path, timestamps)
    with pytest.raises(ValueError, match="exact 1-minute"):
        IntradayWindowDataset(
            path,
            context_min=4,
            context_max=8,
            horizon_length=3,
            stride=1,
        )


def test_masked_covariates_still_require_finite_fill_values(bundle_factory):
    path, _ = bundle_factory("bad-fill")
    values_path = path / "past_only_values.npy"
    mask_path = path / "past_only_mask.npy"
    values = np.load(values_path)
    mask = np.load(mask_path)
    values[0, 0, 2] = np.nan
    mask[0, 0, 2] = True
    np.save(values_path, values)
    np.save(mask_path, mask)
    with pytest.raises(ValueError, match="finite fill"):
        IntradayWindowDataset(
            path,
            context_min=4,
            context_max=8,
            horizon_length=3,
            stride=1,
        )


def test_390_minute_session_has_exact_business_anchor_boundaries(
    bundle_factory,
):
    path, _ = bundle_factory(
        "full-session",
        days=1,
        minutes=390,
    )
    dataset = IntradayWindowDataset(
        path,
        context_min=64,
        context_max=192,
        horizon_length=64,
        stride=1,
    )
    assert len(dataset) == 263
    assert int(dataset._anchor_indices[0]) == 63
    assert int(dataset._anchor_indices[-1]) == 325
    assert dataset[0]["unknown_future_values"][0, 0] == 64
    assert dataset[-1]["unknown_future_values"][0, -1] == 389
