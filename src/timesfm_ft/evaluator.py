"""Inference and evaluation for fine-tuned TimesFM 3 adapters."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import ExperimentConfig
from timesfm_ft.data import NpzWindowDataset, WindowBatch
from timesfm_ft.trainer import resolve_device

LOGGER = logging.getLogger(__name__)


class EvaluationAccumulator:
    """Streaming point, baseline, quantile, and per-horizon metrics."""

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
        self.baseline_squared_error_sum = np.zeros(horizon, dtype=np.float64)
        self.pinball_sum = np.zeros(horizon, dtype=np.float64)
        self.coverage_count = np.zeros(len(quantiles), dtype=np.int64)
        self.crossing_count = 0
        self.crossing_total = 0
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
        median_error = prediction_ticks[:, :, self.median_index] - target_ticks
        self.count += valid.sum(axis=0)
        self.absolute_error_sum += np.where(valid, np.abs(median_error), 0.0).sum(axis=0)
        self.squared_error_sum += np.where(valid, median_error**2, 0.0).sum(axis=0)
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
        ).sum(axis=(0, 1))
        crossings = prediction_ticks[:, :, :-1] > prediction_ticks[:, :, 1:]
        self.crossing_count += int((crossings & valid[:, :, None]).sum())
        self.crossing_total += int(valid.sum()) * max(len(self.quantiles) - 1, 0)
        self.sample_count += predictions.shape[0]

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return numerator / denominator

    def results(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        total_count = int(self.count.sum())
        squared_error = float(self.squared_error_sum.sum())
        baseline_squared_error = float(self.baseline_squared_error_sum.sum())
        mse = self._safe_ratio(squared_error, total_count)
        baseline_mse = self._safe_ratio(baseline_squared_error, total_count)
        coverage = {
            f"{quantile:.1f}": self._safe_ratio(int(count), total_count)
            for quantile, count in zip(
                self.quantiles, self.coverage_count, strict=True
            )
        }
        coverage_errors = [
            abs(value - quantile)
            for quantile, value in zip(
                self.quantiles, coverage.values(), strict=True
            )
            if value is not None
        ]
        summary = {
            "samples": self.sample_count,
            "valid_points": total_count,
            "mae_ticks": self._safe_ratio(
                float(self.absolute_error_sum.sum()), total_count
            ),
            "rmse_ticks": math.sqrt(mse) if mse is not None else None,
            "persistence_rmse_ticks": (
                math.sqrt(baseline_mse) if baseline_mse is not None else None
            ),
            "oos_r2_vs_persistence": (
                1.0 - squared_error / baseline_squared_error
                if baseline_squared_error > 0
                else None
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
                self.crossing_count, self.crossing_total
            ),
        }

        rows: list[dict[str, Any]] = []
        for index in range(self.horizon):
            count = int(self.count[index])
            horizon_mse = self._safe_ratio(self.squared_error_sum[index], count)
            horizon_baseline_sse = self.baseline_squared_error_sum[index]
            rows.append(
                {
                    "step": index + 1,
                    "horizon_seconds": (index + 1) * self.sampling_interval_seconds,
                    "valid_points": count,
                    "mae_ticks": self._safe_ratio(
                        self.absolute_error_sum[index], count
                    ),
                    "rmse_ticks": (
                        math.sqrt(horizon_mse) if horizon_mse is not None else None
                    ),
                    "oos_r2_vs_persistence": (
                        1.0 - self.squared_error_sum[index] / horizon_baseline_sse
                        if horizon_baseline_sse > 0
                        else None
                    ),
                    "mean_pinball_ticks": self._safe_ratio(
                        self.pinball_sum[index],
                        count * len(self.quantiles),
                    ),
                }
            )
        return summary, rows


def _move_batch(batch: WindowBatch, device: torch.device) -> WindowBatch:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def evaluate_experiment(
    config: ExperimentConfig,
    *,
    data_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
) -> Path:
    """Loads the official model, optionally applies an adapter, and evaluates."""

    config.validate()
    device = resolve_device(device_name or config.trainer.device)
    dataset = NpzWindowDataset(
        data_path or config.data.val_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=config.data.max_variates,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
    )
    if adapter_path is not None:
        model.load_adapter(adapter_path)
    model.eval()
    accumulator = EvaluationAccumulator(
        horizon=config.data.horizon_length,
        quantiles=model.quantiles,
        tick_size=config.objective.tick_size,
        sampling_interval_seconds=config.data.sampling_interval_seconds,
    )

    LOGGER.info(
        "eval_start checkpoint=%s adapter=%s data=%s samples=%d variates=%d "
        "context=%d horizon=%d device=%s",
        config.model.checkpoint,
        adapter_path or "zero-shot",
        data_path or config.data.val_path,
        len(dataset),
        dataset.num_variates,
        config.data.context_length,
        config.data.horizon_length,
        device,
    )
    for step, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        predictions = model.predict(
            batch["context_values"],
            horizon=config.data.horizon_length,
            context_mask=batch["context_mask"],
        )
        accumulator.update(
            predictions,
            batch["future_values"],
            batch["context_values"][:, 0, -1],
            batch["future_mask"],
        )
        if step % config.trainer.log_every_steps == 0 or step == len(loader):
            LOGGER.info("eval_progress step=%d/%d", step, len(loader))

    summary, horizon_rows = accumulator.results()
    destination = Path(output_dir or Path(config.trainer.output_dir) / "evaluation")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (destination / "per_horizon.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)

    LOGGER.info(
        "eval_end mae_ticks=%.6f rmse_ticks=%.6f oos_r2=%s "
        "pinball_ticks=%.6f crossing_rate=%.6f artifacts=%s",
        summary["mae_ticks"],
        summary["rmse_ticks"],
        (
            f"{summary['oos_r2_vs_persistence']:.6f}"
            if summary["oos_r2_vs_persistence"] is not None
            else "null"
        ),
        summary["mean_pinball_ticks"],
        summary["quantile_crossing_rate"],
        destination,
    )
    return destination
