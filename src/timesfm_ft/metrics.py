"""Forecast, calibration, slice, and trading-proxy metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _rank_ic(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rank(left), _rank(right)) if len(left) >= 2 else None


def _mean_grouped_ic(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    ranked: bool,
) -> float | None:
    values: list[float] = []
    for group in np.unique(groups):
        selected = groups == group
        value = (
            _rank_ic(prediction[selected], target[selected])
            if ranked
            else _pearson(prediction[selected], target[selected])
        )
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def _max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    equity = np.cumsum(pnl, dtype=np.float64)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))[:-1]
    return float(np.max(peaks - equity))


def _prediction_deciles(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[list[float], float | None, float | None]:
    if len(prediction) < 10:
        return [], None, None
    order = np.argsort(prediction, kind="mergesort")
    buckets = np.array_split(order, 10)
    means = [float(np.mean(target[bucket])) for bucket in buckets if len(bucket)]
    monotonicity = _rank_ic(
        np.arange(len(means), dtype=np.float64),
        np.asarray(means),
    )
    spread = means[-1] - means[0] if len(means) == 10 else None
    return means, monotonicity, spread


class ForecastMetricsAccumulator:
    """Accumulates multi-horizon metrics without mixing intraday sessions."""

    def __init__(
        self,
        *,
        horizon: int,
        quantiles: tuple[float, ...],
        report_horizons: tuple[int, ...],
        trading_horizon: int,
        cost_per_turnover: float,
    ) -> None:
        self.horizon = horizon
        self.quantiles = np.asarray(quantiles, dtype=np.float64)
        self.report_horizons = report_horizons
        self.trading_horizon = trading_horizon
        self.cost_per_turnover = cost_per_turnover
        self.median_index = int(np.argmin(np.abs(self.quantiles - 0.5)))
        if abs(float(self.quantiles[self.median_index]) - 0.5) > 1e-6:
            raise ValueError("metrics require an explicit 0.5 quantile")
        self.lower_index = int(np.argmin(np.abs(self.quantiles - 0.1)))
        self.upper_index = int(np.argmin(np.abs(self.quantiles - 0.9)))
        if (
            abs(float(self.quantiles[self.lower_index]) - 0.1) > 1e-6
            or abs(float(self.quantiles[self.upper_index]) - 0.9) > 1e-6
        ):
            raise ValueError("metrics require explicit 0.1 and 0.9 quantiles")
        self.count = np.zeros(horizon, dtype=np.int64)
        self.absolute_error_sum = np.zeros(horizon, dtype=np.float64)
        self.squared_error_sum = np.zeros(horizon, dtype=np.float64)
        self.zero_squared_error_sum = np.zeros(horizon, dtype=np.float64)
        self.last_squared_error_sum = np.zeros(horizon, dtype=np.float64)
        self.pinball_sum = np.zeros(horizon, dtype=np.float64)
        self.coverage_count = np.zeros((horizon, len(self.quantiles)), dtype=np.int64)
        self.crossing_count = np.zeros(horizon, dtype=np.int64)
        self.direction_correct = np.zeros(horizon, dtype=np.int64)
        self.direction_count = np.zeros(horizon, dtype=np.int64)
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._valid: list[np.ndarray] = []
        self._last_returns: list[np.ndarray] = []
        self._context_lengths: list[np.ndarray] = []
        self._dates: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._minute_indices: list[np.ndarray] = []
        self._context_volatility: list[np.ndarray] = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        last_returns: torch.Tensor,
        context_lengths: torch.Tensor,
        dates: torch.Tensor,
        timestamps: torch.Tensor,
        minute_indices: torch.Tensor,
        context_volatility: torch.Tensor,
    ) -> None:
        if predictions.ndim != 3 or predictions.shape[1] != self.horizon:
            raise ValueError("invalid prediction shape")
        if predictions.shape[-1] != len(self.quantiles):
            raise ValueError("prediction quantile count mismatch")
        if targets.shape != predictions.shape[:2] or target_mask.shape != targets.shape:
            raise ValueError("target arrays must match prediction batch/horizon")
        valid_tensor = ~target_mask.bool()
        prediction_valid = valid_tensor[:, :, None].expand_as(predictions)
        if not torch.isfinite(targets[valid_tensor]).all().item():
            raise ValueError("valid targets contain non-finite values")
        if not torch.isfinite(predictions[prediction_valid]).all().item():
            raise ValueError("valid predictions contain non-finite values")

        prediction_values = predictions.detach().float().cpu().numpy()
        target_values = targets.detach().float().cpu().numpy()
        valid = valid_tensor.detach().cpu().numpy()
        last = last_returns.detach().float().cpu().numpy()
        median = prediction_values[:, :, self.median_index]
        error = median - target_values
        self.count += valid.sum(axis=0)
        self.absolute_error_sum += np.where(valid, np.abs(error), 0.0).sum(axis=0)
        self.squared_error_sum += np.where(valid, error**2, 0.0).sum(axis=0)
        self.zero_squared_error_sum += np.where(valid, target_values**2, 0.0).sum(axis=0)
        last_error = last[:, None] - target_values
        self.last_squared_error_sum += np.where(valid, last_error**2, 0.0).sum(axis=0)

        errors = target_values[:, :, None] - prediction_values
        quantiles = self.quantiles[None, None, :]
        pinball = np.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        self.pinball_sum += np.where(valid[:, :, None], pinball, 0.0).sum(axis=(0, 2))
        self.coverage_count += (
            (target_values[:, :, None] <= prediction_values) & valid[:, :, None]
        ).sum(axis=0)
        crossings = prediction_values[:, :, :-1] > prediction_values[:, :, 1:]
        self.crossing_count += (crossings & valid[:, :, None]).sum(axis=(0, 2))
        nonzero = valid & (target_values != 0.0)
        self.direction_count += nonzero.sum(axis=0)
        self.direction_correct += ((np.sign(median) == np.sign(target_values)) & nonzero).sum(
            axis=0
        )

        self._predictions.append(prediction_values)
        self._targets.append(target_values)
        self._valid.append(valid)
        self._last_returns.append(last)
        self._context_lengths.append(context_lengths.cpu().numpy())
        self._dates.append(dates.cpu().numpy())
        self._timestamps.append(timestamps.cpu().numpy())
        self._minute_indices.append(minute_indices.cpu().numpy())
        self._context_volatility.append(context_volatility.cpu().numpy())

    def _concatenate(self) -> dict[str, np.ndarray]:
        if not self._predictions:
            raise ValueError("metrics accumulator has no samples")
        return {
            "predictions": np.concatenate(self._predictions),
            "targets": np.concatenate(self._targets),
            "valid": np.concatenate(self._valid),
            "last_returns": np.concatenate(self._last_returns),
            "context_lengths": np.concatenate(self._context_lengths),
            "dates": np.concatenate(self._dates),
            "timestamps": np.concatenate(self._timestamps),
            "minute_indices": np.concatenate(self._minute_indices),
            "context_volatility": np.concatenate(self._context_volatility),
        }

    def prediction_arrays(self) -> dict[str, np.ndarray]:
        arrays = self._concatenate()
        return {key: value for key, value in arrays.items() if key != "valid"} | {
            "target_mask": ~arrays["valid"],
            "quantiles": self.quantiles,
        }

    def results(
        self,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        arrays = self._concatenate()
        predictions = arrays["predictions"]
        targets = arrays["targets"]
        valid = arrays["valid"]
        median = predictions[:, :, self.median_index]
        total_count = int(self.count.sum())
        total_sse = float(self.squared_error_sum.sum())
        zero_sse = float(self.zero_squared_error_sum.sum())
        last_sse = float(self.last_squared_error_sum.sum())
        global_coverage = self.coverage_count.sum(axis=0)
        coverage = {
            f"q{int(round(quantile * 100)):02d}": _safe_ratio(int(count), total_count)
            for quantile, count in zip(self.quantiles, global_coverage, strict=True)
        }
        coverage_error = [
            abs(float(value) - float(quantile))
            for value, quantile in zip(coverage.values(), self.quantiles, strict=True)
            if value is not None
        ]
        total_crossing_pairs = total_count * max(len(self.quantiles) - 1, 0)
        lower = predictions[:, :, self.lower_index]
        upper = predictions[:, :, self.upper_index]
        interval_width = np.where(valid, upper - lower, 0.0)
        interval_covered = valid & (targets >= lower) & (targets <= upper)
        summary: dict[str, Any] = {
            "samples": int(len(targets)),
            "valid_points": total_count,
            "mae": _safe_ratio(float(self.absolute_error_sum.sum()), total_count),
            "rmse": math.sqrt(total_sse / total_count) if total_count else None,
            "zero_return_rmse": math.sqrt(zero_sse / total_count) if total_count else None,
            "last_return_rmse": math.sqrt(last_sse / total_count) if total_count else None,
            "oos_r2_vs_zero": 1.0 - total_sse / zero_sse if zero_sse > 0 else None,
            "oos_r2_vs_last": 1.0 - total_sse / last_sse if last_sse > 0 else None,
            "directional_accuracy": _safe_ratio(
                int(self.direction_correct.sum()), int(self.direction_count.sum())
            ),
            "mean_pinball": _safe_ratio(
                float(self.pinball_sum.sum()),
                total_count * len(self.quantiles),
            ),
            "quantile_coverage": coverage,
            "mean_absolute_coverage_error": (
                float(np.mean(coverage_error)) if coverage_error else None
            ),
            "quantile_crossing_rate": _safe_ratio(
                int(self.crossing_count.sum()), total_crossing_pairs
            ),
            "q10_q90_coverage": _safe_ratio(int(interval_covered.sum()), total_count),
            "mean_q10_q90_width": _safe_ratio(float(interval_width.sum()), total_count),
        }

        lead_rows: list[dict[str, Any]] = []
        for index in range(self.horizon):
            mask = valid[:, index]
            prediction = median[mask, index]
            target = targets[mask, index]
            dates = arrays["dates"][mask]
            count = int(mask.sum())
            sse = float(self.squared_error_sum[index])
            zero = float(self.zero_squared_error_sum[index])
            last = float(self.last_squared_error_sum[index])
            row: dict[str, Any] = {
                "lead_minutes": index + 1,
                "valid_points": count,
                "mae": _safe_ratio(float(self.absolute_error_sum[index]), count),
                "rmse": math.sqrt(sse / count) if count else None,
                "zero_return_rmse": math.sqrt(zero / count) if count else None,
                "last_return_rmse": math.sqrt(last / count) if count else None,
                "oos_r2_vs_zero": 1.0 - sse / zero if zero > 0 else None,
                "oos_r2_vs_last": 1.0 - sse / last if last > 0 else None,
                "ic": _pearson(prediction, target),
                "rank_ic": _rank_ic(prediction, target),
                "mean_daily_ic": _mean_grouped_ic(prediction, target, dates, ranked=False),
                "mean_daily_rank_ic": _mean_grouped_ic(prediction, target, dates, ranked=True),
                "directional_accuracy": _safe_ratio(
                    int(self.direction_correct[index]),
                    int(self.direction_count[index]),
                ),
                "mean_pinball": _safe_ratio(
                    float(self.pinball_sum[index]),
                    count * len(self.quantiles),
                ),
                "quantile_crossing_rate": _safe_ratio(
                    int(self.crossing_count[index]),
                    count * max(len(self.quantiles) - 1, 0),
                ),
                "q10_q90_coverage": _safe_ratio(int(interval_covered[:, index].sum()), count),
                "mean_q10_q90_width": _safe_ratio(float(interval_width[:, index].sum()), count),
            }
            for q_index, quantile in enumerate(self.quantiles):
                row[f"coverage_q{int(round(quantile * 100)):02d}"] = _safe_ratio(
                    int(self.coverage_count[index, q_index]), count
                )
            lead_rows.append(row)

        cumulative_rows = [
            self._cumulative_row(arrays, horizon) for horizon in self.report_horizons
        ]
        summary["trading_proxy"] = self._trading_proxy(arrays)
        slice_rows = self._slice_rows(arrays)
        return summary, lead_rows, cumulative_rows, slice_rows

    def _cumulative_row(
        self,
        arrays: dict[str, np.ndarray],
        horizon: int,
        selection: np.ndarray | None = None,
    ) -> dict[str, Any]:
        eligible = (
            np.ones(len(arrays["targets"]), dtype=np.bool_)
            if selection is None
            else selection.copy()
        )
        valid = eligible & arrays["valid"][:, :horizon].all(axis=1)
        masked_path_excluded = int(eligible.sum() - valid.sum())
        targets = arrays["targets"][valid, :horizon].sum(axis=1)
        medians = arrays["predictions"][valid, :horizon, self.median_index].sum(axis=1)
        last = arrays["last_returns"][valid] * horizon
        dates = arrays["dates"][valid]
        error = medians - targets
        zero_sse = float(np.dot(targets, targets))
        last_error = last - targets
        last_sse = float(np.dot(last_error, last_error))
        sse = float(np.dot(error, error))
        nonzero = targets != 0
        long = medians > 0
        short = medians < 0
        long_mean = float(targets[long].mean()) if long.any() else None
        short_mean = float(targets[short].mean()) if short.any() else None
        decile_means, decile_monotonicity, decile_spread = _prediction_deciles(medians, targets)
        return {
            "horizon_minutes": horizon,
            "samples": int(len(targets)),
            "masked_path_excluded": masked_path_excluded,
            "mae": float(np.mean(np.abs(error))) if len(error) else None,
            "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None,
            "ic": _pearson(medians, targets),
            "rank_ic": _rank_ic(medians, targets),
            "mean_daily_ic": _mean_grouped_ic(medians, targets, dates, ranked=False),
            "mean_daily_rank_ic": _mean_grouped_ic(medians, targets, dates, ranked=True),
            "directional_accuracy": (
                float(np.mean(np.sign(medians[nonzero]) == np.sign(targets[nonzero])))
                if nonzero.any()
                else None
            ),
            "oos_r2_vs_zero": 1.0 - sse / zero_sse if zero_sse > 0 else None,
            "oos_r2_vs_last": 1.0 - sse / last_sse if last_sse > 0 else None,
            "long_count": int(long.sum()),
            "short_count": int(short.sum()),
            "long_conditional_mean": long_mean,
            "short_conditional_mean": short_mean,
            "long_short_spread": (
                long_mean - short_mean if long_mean is not None and short_mean is not None else None
            ),
            "prediction_decile_means": decile_means,
            "prediction_decile_monotonicity": decile_monotonicity,
            "prediction_decile_spread": decile_spread,
        }

    def _trading_proxy(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        horizon = self.trading_horizon
        valid = arrays["valid"][:, :horizon].all(axis=1)
        target = arrays["targets"][valid, :horizon].sum(axis=1)
        forecast = arrays["predictions"][valid, :horizon, self.median_index].sum(axis=1)
        dates = arrays["dates"][valid]
        timestamps = arrays["timestamps"][valid]
        order = np.lexsort((timestamps, dates))
        target = target[order]
        forecast = forecast[order]
        dates = dates[order]
        position = np.sign(forecast)
        previous = np.zeros_like(position)
        if len(position) > 1:
            previous[1:] = np.where(dates[1:] == dates[:-1], position[:-1], 0.0)
        turnover = np.abs(position - previous)
        gross = position * target
        net = gross - self.cost_per_turnover * turnover
        return {
            "kind": "overlapping_signal_research_proxy",
            "horizon_minutes": horizon,
            "samples": int(len(net)),
            "gross_mean": float(gross.mean()) if len(gross) else None,
            "net_mean": float(net.mean()) if len(net) else None,
            "mean_turnover": float(turnover.mean()) if len(turnover) else None,
            "hit_rate": float(np.mean(gross > 0)) if len(gross) else None,
            "max_drawdown": _max_drawdown(net),
            "cost_per_turnover": self.cost_per_turnover,
        }

    def _slice_rows(self, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        horizon = self.trading_horizon
        contexts = arrays["context_lengths"]
        minutes = arrays["minute_indices"]
        volatility = arrays["context_volatility"]
        median_volatility = float(np.median(volatility))
        selections = {
            "context_64_95": (contexts >= 64) & (contexts <= 95),
            "context_96_127": (contexts >= 96) & (contexts <= 127),
            "context_128_191": (contexts >= 128) & (contexts <= 191),
            "context_192_plus": contexts >= 192,
            "session_open": minutes < 90,
            "session_midday": (minutes >= 90) & (minutes < 300),
            "session_close": minutes >= 300,
            "volatility_low": volatility <= median_volatility,
            "volatility_high": volatility > median_volatility,
        }
        rows: list[dict[str, Any]] = []
        for name, selection in selections.items():
            row = self._cumulative_row(arrays, horizon, selection)
            row["slice"] = name
            rows.append(row)
        return rows
