from __future__ import annotations

import pytest
import torch
from timesfm3 import util

from timesfm_ft.data import collate_intraday_windows
from timesfm_ft.dense import (
    build_dense_training_batch,
    gather_final_anchor,
    shifted_output_patches,
)


def _batch(*, future_mask: torch.Tensor | None = None):
    target_context = torch.arange(8, dtype=torch.float32)
    past_context = target_context + 10
    target_future = torch.arange(8, 12, dtype=torch.float32)
    past_future = target_future + 10
    known = torch.arange(12, dtype=torch.float32)[None, None, :]
    return {
        "context_values": torch.stack((target_context, past_context))[None],
        "context_mask": torch.zeros(1, 2, 8, dtype=torch.bool),
        "context_padding_mask": torch.zeros(1, 8, dtype=torch.bool),
        "past_future_values": known,
        "past_future_mask": torch.zeros_like(known, dtype=torch.bool),
        "unknown_future_values": torch.stack((target_future, past_future))[None],
        "unknown_future_mask": torch.stack(
            (
                torch.zeros(4, dtype=torch.bool) if future_mask is None else future_mask[0],
                torch.zeros(4, dtype=torch.bool),
            )
        )[None],
        "context_lengths": torch.tensor([8], dtype=torch.int16),
        "timestamps": torch.tensor([1], dtype=torch.int64),
        "dates": torch.tensor([20250102], dtype=torch.int32),
        "minute_indices": torch.tensor([7], dtype=torch.int16),
        "last_returns": torch.tensor([7.0]),
        "context_volatility": torch.tensor([1.0]),
    }


def test_shifted_patch_labels_are_next_output_patches():
    values = torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2)
    labels, wrap = shifted_output_patches(values, rolls=2)
    expected, expected_wrap = util.get_output_patch_via_roll(values, 2)
    torch.testing.assert_close(labels, expected)
    assert torch.equal(wrap, expected_wrap.expand_as(wrap))
    torch.testing.assert_close(labels[0, 0, 1], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert not wrap[0, 0, 1].any()
    assert wrap[0, 0, -1].all()


def test_dense_batch_builds_roles_labels_and_eligible_anchors():
    dense = build_dense_training_batch(
        _batch(),
        context_min=4,
        input_patch_length=2,
        output_patch_length=4,
    )
    assert dense.values.shape == (1, 3, 6, 2)
    assert dense.patch_is_target[0, :2].all()
    assert not dense.patch_is_target[0, 2].any()
    assert not dense.masks[0, 0, 4:].any()
    assert dense.masks[0, 1, 4:].all()
    torch.testing.assert_close(
        dense.target_labels[0, 3],
        torch.tensor([8.0, 9.0, 10.0, 11.0]),
    )
    torch.testing.assert_close(
        dense.past_only_labels[0, 0, 3],
        torch.tensor([18.0, 19.0, 20.0, 21.0]),
    )
    assert dense.eligible_anchor_mask.tolist() == [[False, True, True, True, False, False]]
    torch.testing.assert_close(
        gather_final_anchor(
            dense.target_labels,
            dense.final_anchor_indices,
        ),
        torch.tensor([[8.0, 9.0, 10.0, 11.0]]),
    )


def test_invalid_shifted_path_removes_only_affected_dense_anchors():
    future_mask = torch.tensor([[False, False, True, False]])
    dense = build_dense_training_batch(
        _batch(future_mask=future_mask),
        context_min=4,
        input_patch_length=2,
        output_patch_length=4,
    )
    assert dense.eligible_anchor_mask.tolist() == [[False, True, True, False, False, False]]
    assert dense.target_label_mask[0, 3, 2]


def test_masked_history_does_not_count_toward_minimum_visible_history():
    batch = _batch()
    batch["context_mask"][0, 0, 0] = True
    dense = build_dense_training_batch(
        batch,
        context_min=4,
        input_patch_length=2,
        output_patch_length=4,
    )
    assert dense.eligible_anchor_mask.tolist() == [[False, False, True, True, False, False]]


@pytest.mark.parametrize(
    ("context_length", "eligible_context_tokens"),
    [
        (63, ()),
        (64, (1,)),
        (65, (2,)),
        (83, (2,)),
        (96, (1, 2)),
        (192, (1, 2, 3, 4, 5)),
    ],
)
def test_production_patch_eligibility_matches_scalar_oracle(
    context_length,
    eligible_context_tokens,
):
    horizon = 64
    context = torch.arange(context_length, dtype=torch.float32)
    future = torch.arange(
        context_length,
        context_length + horizon,
        dtype=torch.float32,
    )
    sample = {
        "context_values": context[None, :],
        "context_mask": torch.zeros(1, context_length, dtype=torch.bool),
        "past_future_values": torch.empty(0, context_length + horizon),
        "past_future_mask": torch.empty(0, context_length + horizon, dtype=torch.bool),
        "unknown_future_values": future[None, :],
        "unknown_future_mask": torch.zeros(1, horizon, dtype=torch.bool),
        "context_length": context_length,
        "timestamp": 1,
        "date": 20250102,
        "minute_index": context_length - 1,
        "last_return": float(context[-1]),
        "context_volatility": 1.0,
    }
    batch = collate_intraday_windows([sample], patch_length=32)
    dense = build_dense_training_batch(
        batch,
        context_min=64,
        input_patch_length=32,
        output_patch_length=64,
    )
    padded_context = ((context_length + 31) // 32) * 32
    assert batch["context_values"].shape[-1] == padded_context
    assert batch["context_values"][0, 0, -1] == context_length - 1
    actual = tuple(
        torch.nonzero(dense.eligible_anchor_mask[0, : dense.context_patch_count]).flatten().tolist()
    )
    assert actual == eligible_context_tokens
    final = int(dense.final_anchor_indices[0])
    assert dense.target_labels[0, final, 0] == context_length
    assert dense.target_labels[0, final, 63] == context_length + 63
