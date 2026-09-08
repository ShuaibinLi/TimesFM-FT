"""Mask-aware streaming forecast metrics shared by training and evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


class ForecastMetricsAccumulator:
    """Accumulates point, persistence, direction, quantile, and crossing metrics."""

    def __init__(
        self,
        *,
        horizon: int,
        quantiles: tuple[float, ...],
        tick_size: float,
        sampling_interval_seconds: float,
    ) -> None:
        self.horizon = horizon
        self.quantiles = np.asarray(quantiles, dtype=np.float64)
        self.tick_size = tick_size
        self.sampling_interval_seconds = sampling_interval_seconds
        self.median_index = int(np.argmin(np.abs(self.quantiles - 0.5)))
        self.count = np.zeros(horizon, dtype=np.int64)
        self.absolute_error_sum = np.zeros(horizon, dtype=np.float64)
        self.squared_error_sum = np.zeros(horizon, dtype=np.float64)
        self.baseline_absolute_error_sum = np.zeros(horizon, dtype=np.float64)
        self.baseline_squared_error_sum = np.zeros(horizon, dtype=np.float64)
        self.pinball_sum = np.zeros(horizon, dtype=np.float64)
        self.coverage_count = np.zeros((horizon, len(quantiles)), dtype=np.int64)
        self.crossing_count = np.zeros(horizon, dtype=np.int64)
        self.direction_correct_count = np.zeros(horizon, dtype=np.int64)
        self.direction_count = np.zeros(horizon, dtype=np.int64)
        self.sample_count = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        current_price: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> None:
        prediction_values = predictions.detach().float().cpu().numpy()
        target_values = targets.detach().float().cpu().numpy()
        origins = current_price.detach().float().cpu().numpy()[:, None]
        valid = ~target_mask.detach().cpu().numpy().astype(bool)

        prediction_ticks = (prediction_values - origins[:, :, None]) / self.tick_size
        target_ticks = (target_values - origins) / self.tick_size
        median_prediction = prediction_ticks[:, :, self.median_index]
        median_error = median_prediction - target_ticks
        self.count += valid.sum(axis=0)
        self.absolute_error_sum += np.where(valid, np.abs(median_error), 0.0).sum(axis=0)
        self.squared_error_sum += np.where(valid, median_error**2, 0.0).sum(axis=0)
        self.baseline_absolute_error_sum += np.where(
            valid, np.abs(target_ticks), 0.0
        ).sum(axis=0)
        self.baseline_squared_error_sum += np.where(valid, target_ticks**2, 0.0).sum(
            axis=0
        )

        errors = target_ticks[:, :, None] - prediction_ticks
        quantiles = self.quantiles[None, None, :]
        pinball = np.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        self.pinball_sum += np.where(valid[:, :, None], pinball, 0.0).sum(
            axis=(0, 2)
        )
        self.coverage_count += (
            (target_ticks[:, :, None] <= prediction_ticks) & valid[:, :, None]
        ).sum(axis=0)
        crossings = prediction_ticks[:, :, :-1] > prediction_ticks[:, :, 1:]
        self.crossing_count += (crossings & valid[:, :, None]).sum(axis=(0, 2))

        nonzero = valid & (target_ticks != 0.0)
        self.direction_count += nonzero.sum(axis=0)
        self.direction_correct_count += (
            (np.sign(median_prediction) == np.sign(target_ticks)) & nonzero
        ).sum(axis=0)
        self.sample_count += predictions.shape[0]

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator > 0 else None

    @staticmethod
    def _sqrt(value: float | None) -> float | None:
        return math.sqrt(value) if value is not None else None

    def results(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        total_count = int(self.count.sum())
        squared_error = float(self.squared_error_sum.sum())
        baseline_squared_error = float(self.baseline_squared_error_sum.sum())
        total_direction_count = int(self.direction_count.sum())
        total_crossing_pairs = total_count * max(len(self.quantiles) - 1, 0)
        global_coverage = self.coverage_count.sum(axis=0)
        coverage = {
            str(float(quantile)): self._safe_ratio(int(count), total_count)
            for quantile, count in zip(
                self.quantiles, global_coverage, strict=True
            )
        }
        coverage_errors = [
            abs(value - quantile)
            for quantile, value in zip(
                self.quantiles, coverage.values(), strict=True
            )
            if value is not None
        ]
        mse = self._safe_ratio(squared_error, total_count)
        baseline_mse = self._safe_ratio(baseline_squared_error, total_count)
        summary = {
            "samples": self.sample_count,
            "valid_points": total_count,
            "mae_ticks": self._safe_ratio(
                float(self.absolute_error_sum.sum()), total_count
            ),
            "rmse_ticks": self._sqrt(mse),
            "persistence_mae_ticks": self._safe_ratio(
                float(self.baseline_absolute_error_sum.sum()), total_count
            ),
            "persistence_rmse_ticks": self._sqrt(baseline_mse),
            "oos_r2_vs_persistence": (
                1.0 - squared_error / baseline_squared_error
                if baseline_squared_error > 0
                else None
            ),
            "directional_accuracy": self._safe_ratio(
                int(self.direction_correct_count.sum()), total_direction_count
            ),
            "mean_pinball_ticks": self._safe_ratio(
                float(self.pinball_sum.sum()),
                total_count * len(self.quantiles),
            ),
            "quantile_coverage": coverage,
            "mean_absolute_coverage_error": (
                float(np.mean(coverage_errors)) if coverage_errors else None
            ),
            "quantile_crossing_rate": self._safe_ratio(
                int(self.crossing_count.sum()), total_crossing_pairs
            ),
        }

        rows: list[dict[str, Any]] = []
        for index in range(self.horizon):
            count = int(self.count[index])
            horizon_sse = float(self.squared_error_sum[index])
            horizon_baseline_sse = float(self.baseline_squared_error_sum[index])
            row: dict[str, Any] = {
                "step": index + 1,
                "horizon_seconds": (index + 1) * self.sampling_interval_seconds,
                "valid_points": count,
                "mae_ticks": self._safe_ratio(
                    float(self.absolute_error_sum[index]), count
                ),
                "rmse_ticks": self._sqrt(
                    self._safe_ratio(horizon_sse, count)
                ),
                "persistence_mae_ticks": self._safe_ratio(
                    float(self.baseline_absolute_error_sum[index]), count
                ),
                "persistence_rmse_ticks": self._sqrt(
                    self._safe_ratio(horizon_baseline_sse, count)
                ),
                "oos_r2_vs_persistence": (
                    1.0 - horizon_sse / horizon_baseline_sse
                    if horizon_baseline_sse > 0
                    else None
                ),
                "directional_accuracy": self._safe_ratio(
                    int(self.direction_correct_count[index]),
                    int(self.direction_count[index]),
                ),
                "mean_pinball_ticks": self._safe_ratio(
                    float(self.pinball_sum[index]),
                    count * len(self.quantiles),
                ),
                "quantile_crossing_rate": self._safe_ratio(
                    int(self.crossing_count[index]),
                    count * max(len(self.quantiles) - 1, 0),
                ),
            }
            for quantile_index, quantile in enumerate(self.quantiles):
                row[f"coverage_q{int(round(quantile * 100)):02d}"] = self._safe_ratio(
                    int(self.coverage_count[index, quantile_index]),
                    count,
                )
            rows.append(row)
        return summary, rows
