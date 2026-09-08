from __future__ import annotations

import pytest
import torch

from timesfm_ft.losses import ForecastLoss


def _loss() -> ForecastLoss:
    return ForecastLoss(
        [0.1, 0.5, 0.9],
        tick_size=0.01,
        pinball_weight=1.0,
        median_huber_weight=1.0,
        crossing_weight=0.1,
    )


def test_balanced_objective_excludes_median_from_pinball():
    target = torch.tensor([[100.0]])
    predictions = torch.tensor([[[100.0, 100.01, 100.0]]])
    output = _loss()(
        predictions,
        target,
        current_price=torch.tensor([100.0]),
    )
    assert output.pinball == 0.0
    assert output.median_huber > 0.0


def test_loss_computes_in_float32_and_respects_mask():
    target = torch.tensor([[100.01, 100.02]])
    predictions = target[:, :, None].repeat(1, 1, 3).to(torch.bfloat16)
    output = _loss()(
        predictions,
        target,
        current_price=torch.tensor([100.0]),
        target_mask=torch.tensor([[False, True]]),
    )
    assert output.total.dtype == torch.float32
    assert torch.isfinite(output.total)


def test_crossing_penalty_detects_adjacent_quantile_inversion():
    target = torch.tensor([[100.0]])
    predictions = torch.tensor([[[100.02, 100.01, 100.00]]])
    output = _loss()(
        predictions,
        target,
        current_price=torch.tensor([100.0]),
    )
    assert output.crossing > 0.0


def test_all_masked_batch_is_rejected():
    with pytest.raises(ValueError, match="no valid target"):
        _loss()(
            torch.ones(1, 1, 3),
            torch.ones(1, 1),
            current_price=torch.ones(1),
            target_mask=torch.ones(1, 1, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantiles": [0.0, 0.5, 0.9]}, "in \\(0, 1\\)"),
        ({"quantiles": [0.1, 0.5, 1.0]}, "in \\(0, 1\\)"),
        ({"quantiles": [0.1, float("nan"), 0.9]}, "finite"),
        (
            {"quantiles": [0.1, 0.5, 0.9], "huber_delta_ticks": 0.0},
            "huber_delta_ticks",
        ),
    ],
)
def test_loss_rejects_invalid_parameters(kwargs, message):
    kwargs.setdefault("tick_size", 0.01)
    with pytest.raises(ValueError, match=message):
        ForecastLoss(**kwargs)
