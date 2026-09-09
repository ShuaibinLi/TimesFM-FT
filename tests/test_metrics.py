from __future__ import annotations

import pytest
import torch

from timesfm_ft.metrics import ForecastMetricsAccumulator


def _metadata(samples: int):
    return {
        "last_returns": torch.zeros(samples),
        "context_lengths": torch.tensor([64, 96, 192][:samples]),
        "dates": torch.full((samples,), 20250102, dtype=torch.int32),
        "timestamps": torch.arange(samples, dtype=torch.int64) + 1,
        "minute_indices": torch.tensor([63, 120, 310][:samples]),
        "context_volatility": torch.tensor([0.1, 0.2, 0.3][:samples]),
    }


def test_metrics_report_lead_cumulative_ic_slices_and_trading_proxy():
    accumulator = ForecastMetricsAccumulator(
        horizon=3,
        quantiles=(0.1, 0.5, 0.9),
        report_horizons=(1, 3),
        trading_horizon=3,
        cost_per_turnover=0.1,
    )
    targets = torch.tensor([[1.0, 0.5, 0.25], [2.0, 1.0, 0.5], [-1.0, -0.5, -0.25]])
    predictions = targets[:, :, None].repeat(1, 1, 3)
    accumulator.update(
        predictions,
        targets,
        torch.zeros_like(targets, dtype=torch.bool),
        **_metadata(3),
    )
    summary, leads, cumulative, slices = accumulator.results()
    assert summary["rmse"] == 0.0
    assert summary["mean_pinball"] == 0.0
    assert summary["q10_q90_coverage"] == 1.0
    assert summary["mean_q10_q90_width"] == 0.0
    assert leads[0]["ic"] == pytest.approx(1.0)
    assert cumulative[-1]["rank_ic"] == pytest.approx(1.0)
    assert cumulative[-1]["mean_daily_rank_ic"] == pytest.approx(1.0)
    assert cumulative[-1]["long_conditional_mean"] > 0
    assert cumulative[-1]["short_conditional_mean"] < 0
    assert summary["trading_proxy"]["net_mean"] < summary["trading_proxy"]["gross_mean"]
    assert {row["slice"] for row in slices} >= {
        "context_64_95",
        "session_close",
        "volatility_high",
    }


def test_metrics_detect_crossing_and_nonfinite_predictions():
    accumulator = ForecastMetricsAccumulator(
        horizon=1,
        quantiles=(0.1, 0.5, 0.9),
        report_horizons=(1,),
        trading_horizon=1,
        cost_per_turnover=0.0,
    )
    targets = torch.ones(1, 1)
    accumulator.update(
        torch.tensor([[[2.0, 1.0, 0.0]]]),
        targets,
        torch.zeros_like(targets, dtype=torch.bool),
        **_metadata(1),
    )
    summary, *_ = accumulator.results()
    assert summary["quantile_crossing_rate"] == 1.0

    bad = ForecastMetricsAccumulator(
        horizon=1,
        quantiles=(0.1, 0.5, 0.9),
        report_horizons=(1,),
        trading_horizon=1,
        cost_per_turnover=0.0,
    )
    with pytest.raises(ValueError, match="non-finite"):
        bad.update(
            torch.tensor([[[0.0, float("nan"), 1.0]]]),
            targets,
            torch.zeros_like(targets, dtype=torch.bool),
            **_metadata(1),
        )
