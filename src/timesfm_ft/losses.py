"""Single-target multi-horizon probabilistic losses."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    pinball: torch.Tensor
    median_huber: torch.Tensor
    crossing: torch.Tensor


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype)
    denominator = weights.sum()
    return (values * weights).sum() / denominator


class ForecastLoss(nn.Module):
    """Pinball + median Huber + quantile crossing loss in tick space."""

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float],
        *,
        tick_size: float,
        target_mode: Literal["level", "delta_ticks"] = "level",
        pinball_weight: float = 1.0,
        include_median_in_pinball: bool = False,
        median_huber_weight: float = 0.5,
        crossing_weight: float = 0.05,
        huber_delta_ticks: float = 1.0,
    ) -> None:
        super().__init__()
        if target_mode not in {"level", "delta_ticks"}:
            raise ValueError(f"unsupported target_mode={target_mode!r}")
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if not math.isfinite(huber_delta_ticks) or huber_delta_ticks <= 0:
            raise ValueError("huber_delta_ticks must be positive")
        weights = (pinball_weight, median_huber_weight, crossing_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("loss weights must be finite and non-negative")
        quantile_tensor = torch.tensor(quantiles, dtype=torch.float32)
        if quantile_tensor.ndim != 1 or quantile_tensor.numel() == 0:
            raise ValueError("quantiles must be a non-empty sequence")
        if not torch.isfinite(quantile_tensor).all() or not torch.all(
            (quantile_tensor > 0) & (quantile_tensor < 1)
        ):
            raise ValueError("quantiles must be finite and in (0, 1)")
        if not torch.all(quantile_tensor[1:] > quantile_tensor[:-1]):
            raise ValueError("quantiles must be strictly increasing")
        self.register_buffer("quantiles_tensor", quantile_tensor, persistent=False)
        self.tick_size = tick_size
        self.target_mode = target_mode
        self.pinball_weight = pinball_weight
        self.include_median_in_pinball = include_median_in_pinball
        self.median_huber_weight = median_huber_weight
        self.crossing_weight = crossing_weight
        self.huber_delta_ticks = huber_delta_ticks
        self.median_index = int(torch.argmin(torch.abs(quantile_tensor - 0.5)).item())
        if abs(float(quantile_tensor[self.median_index]) - 0.5) > 1e-6:
            raise ValueError("balanced objective requires an explicit 0.5 quantile")
        pinball_indices = [
            index
            for index in range(len(quantiles))
            if include_median_in_pinball or index != self.median_index
        ]
        if not pinball_indices:
            raise ValueError("objective requires at least one pinball quantile")
        self.register_buffer(
            "pinball_indices",
            torch.tensor(pinball_indices, dtype=torch.long),
            persistent=False,
        )

    @property
    def pinball_quantile_count(self) -> int:
        return int(self.pinball_indices.numel())

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        current_price: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        if predictions.ndim != 3:
            raise ValueError("predictions must have shape (batch, horizon, quantiles)")
        if targets.shape != predictions.shape[:2]:
            raise ValueError("targets must match prediction batch and horizon")
        if self.target_mode == "level" and current_price.shape != (
            predictions.shape[0],
        ):
            raise ValueError("current_price must have shape (batch,)")
        if predictions.shape[-1] != self.quantiles_tensor.numel():
            raise ValueError("prediction quantile count does not match configured quantiles")
        if target_mask is not None and target_mask.shape != targets.shape:
            raise ValueError("target_mask must match targets")

        predictions = predictions.float()
        targets = targets.float()
        current_price = current_price.float()
        valid = (
            ~target_mask.bool()
            if target_mask is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        if not torch.any(valid).item():
            raise ValueError("loss batch has no valid target values")
        if self.target_mode == "delta_ticks":
            target_ticks = targets
            prediction_ticks = predictions
        else:
            origin = current_price[:, None]
            target_ticks = (targets - origin) / self.tick_size
            prediction_ticks = (predictions - origin[:, :, None]) / self.tick_size

        # Pinball supervises either all quantiles (pinball-only experiments) or
        # the tails only when P50 receives dedicated Huber supervision.
        errors = target_ticks[:, :, None] - prediction_ticks
        pinball_indices = self.pinball_indices.to(predictions.device)
        pinball_errors = errors.index_select(-1, pinball_indices)
        pinball_quantiles = self.quantiles_tensor.to(predictions.device).index_select(
            0, pinball_indices
        )[None, None, :]
        pinball_values = torch.maximum(
            pinball_quantiles * pinball_errors,
            (pinball_quantiles - 1.0) * pinball_errors,
        )
        pinball = _masked_mean(
            pinball_values,
            valid[:, :, None].expand_as(pinball_values),
        )

        # P50 gets smooth point-forecast supervision: quadratic below the
        # configured tick threshold and linear for larger errors.
        median_values = F.huber_loss(
            prediction_ticks[:, :, self.median_index],
            target_ticks,
            reduction="none",
            delta=self.huber_delta_ticks,
        )
        median_huber = _masked_mean(median_values, valid)

        # Adjacent ordering is sufficient to enforce global quantile ordering.
        crossing_values = F.relu(
            prediction_ticks[:, :, :-1] - prediction_ticks[:, :, 1:]
        )
        crossing = _masked_mean(
            crossing_values,
            valid[:, :, None].expand_as(crossing_values),
        )

        total = (
            self.pinball_weight * pinball
            + self.median_huber_weight * median_huber
            + self.crossing_weight * crossing
        )
        return LossOutput(
            total=total,
            pinball=pinball,
            median_huber=median_huber,
            crossing=crossing,
        )
