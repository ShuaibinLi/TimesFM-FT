"""Probabilistic objective for 1-minute return forecasts."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    pinball: torch.Tensor


class PinballLoss(nn.Module):
    """Mask-aware mean Pinball loss over every configured quantile."""

    def __init__(self, quantiles: tuple[float, ...] | list[float]) -> None:
        super().__init__()
        values = torch.tensor(quantiles, dtype=torch.float32)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("quantiles must be a non-empty sequence")
        if not torch.isfinite(values).all() or not torch.all((values > 0) & (values < 1)):
            raise ValueError("quantiles must be finite and in (0, 1)")
        if not torch.all(values[1:] > values[:-1]):
            raise ValueError("quantiles must be strictly increasing")
        self.register_buffer("quantiles_tensor", values, persistent=False)

    @property
    def quantile_count(self) -> int:
        return int(self.quantiles_tensor.numel())

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        if predictions.ndim != 3:
            raise ValueError("predictions must have shape (batch, horizon, quantiles)")
        if targets.shape != predictions.shape[:2]:
            raise ValueError("targets must match prediction batch and horizon")
        if predictions.shape[-1] != self.quantile_count:
            raise ValueError("prediction quantile count does not match configured quantiles")
        if target_mask is not None and target_mask.shape != targets.shape:
            raise ValueError("target_mask must match targets")

        predictions = predictions.float()
        targets = targets.float()
        valid = (
            ~target_mask.bool()
            if target_mask is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        if not torch.any(valid).item():
            raise ValueError("loss batch has no valid target values")
        if not torch.isfinite(targets[valid]).all().item():
            raise ValueError("valid targets contain non-finite values")
        expanded_valid = valid[:, :, None].expand_as(predictions)
        if not torch.isfinite(predictions[expanded_valid]).all().item():
            raise ValueError("valid predictions contain non-finite values")

        error = targets[:, :, None] - predictions
        quantiles = self.quantiles_tensor.to(predictions.device)[None, None, :]
        values = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
        weights = expanded_valid.to(values.dtype)
        pinball = torch.where(expanded_valid, values, 0.0).sum() / weights.sum()
        return LossOutput(total=pinball, pinball=pinball)
