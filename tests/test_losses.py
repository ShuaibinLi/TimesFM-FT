from __future__ import annotations

import pytest
import torch

from timesfm_ft.losses import PinballLoss


def test_pinball_is_zero_for_perfect_forecast():
    target = torch.tensor([[0.5, -0.25]])
    predictions = target[:, :, None].repeat(1, 1, 3)
    output = PinballLoss([0.1, 0.5, 0.9])(predictions, target)
    torch.testing.assert_close(output.total, torch.tensor(0.0))


def test_pinball_respects_target_mask_and_uses_float32():
    target = torch.tensor([[1.0, 100.0]])
    predictions = torch.tensor([[[0.0, 0.0, 0.0], [float("nan")] * 3]])
    output = PinballLoss([0.1, 0.5, 0.9])(
        predictions.to(torch.bfloat16),
        target,
        target_mask=torch.tensor([[False, True]]),
    )
    assert output.total.dtype == torch.float32
    assert torch.isfinite(output.total)


def test_pinball_rejects_bad_quantiles_and_all_masked_batch():
    with pytest.raises(ValueError, match="strictly increasing"):
        PinballLoss([0.5, 0.1])
    loss = PinballLoss([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="no valid"):
        loss(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2),
            target_mask=torch.ones(1, 2, dtype=torch.bool),
        )
