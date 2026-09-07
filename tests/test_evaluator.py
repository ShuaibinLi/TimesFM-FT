from __future__ import annotations

import pytest
import torch

from timesfm_ft.evaluator import EvaluationAccumulator


def test_evaluation_accumulator_reports_point_and_baseline_metrics():
    accumulator = EvaluationAccumulator(
        horizon=2,
        quantiles=(0.1, 0.5, 0.9),
        tick_size=1.0,
        sampling_interval_seconds=0.5,
    )
    targets = torch.tensor([[1.0, 2.0]])
    predictions = targets[:, :, None].repeat(1, 1, 3)
    accumulator.update(
        predictions,
        targets,
        current_price=torch.tensor([0.0]),
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
    )

    summary, rows = accumulator.results()
    assert summary["mae_ticks"] == 0.0
    assert summary["rmse_ticks"] == 0.0
    assert summary["oos_r2_vs_persistence"] == 1.0
    assert summary["mean_pinball_ticks"] == 0.0
    assert summary["quantile_crossing_rate"] == 0.0
    assert rows[0]["horizon_seconds"] == 0.5
    assert rows[1]["horizon_seconds"] == 1.0


def test_evaluation_accumulator_respects_masks_and_detects_crossing():
    accumulator = EvaluationAccumulator(
        horizon=2,
        quantiles=(0.1, 0.5, 0.9),
        tick_size=1.0,
        sampling_interval_seconds=0.5,
    )
    predictions = torch.tensor([[[2.0, 1.0, 3.0], [100.0, 0.0, -100.0]]])
    targets = torch.tensor([[1.0, 5.0]])
    accumulator.update(
        predictions,
        targets,
        current_price=torch.tensor([0.0]),
        target_mask=torch.tensor([[False, True]]),
    )

    summary, _ = accumulator.results()
    assert summary["valid_points"] == 1
    assert summary["mae_ticks"] == 0.0
    assert summary["quantile_crossing_rate"] == pytest.approx(0.5)
