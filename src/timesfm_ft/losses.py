"""Single-target multi-horizon probabilistic losses."""

from __future__ import annotations

import dataclasses

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
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


class ForecastLoss(nn.Module):
    """Pinball + median Huber + quantile crossing loss in tick space."""

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float],
        *,
        tick_size: float,
        pinball_weight: float = 1.0,
        median_huber_weight: float = 0.5,
        crossing_weight: float = 0.05,
        huber_delta_ticks: float = 1.0,
    ) -> None:
        super().__init__()
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        quantile_tensor = torch.tensor(quantiles, dtype=torch.float32)
        if quantile_tensor.ndim != 1 or quantile_tensor.numel() == 0:
            raise ValueError("quantiles must be a non-empty sequence")
        if not torch.all(quantile_tensor[1:] > quantile_tensor[:-1]):
            raise ValueError("quantiles must be strictly increasing")
        self.register_buffer("quantiles_tensor", quantile_tensor, persistent=False)
        self.tick_size = tick_size
        self.pinball_weight = pinball_weight
        self.median_huber_weight = median_huber_weight
        self.crossing_weight = crossing_weight
        self.huber_delta_ticks = huber_delta_ticks
        self.median_index = int(torch.argmin(torch.abs(quantile_tensor - 0.5)).item())

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
        if current_price.shape != (predictions.shape[0],):
            raise ValueError("current_price must have shape (batch,)")
        if predictions.shape[-1] != self.quantiles_tensor.numel():
            raise ValueError("prediction quantile count does not match configured quantiles")
        if target_mask is not None and target_mask.shape != targets.shape:
            raise ValueError("target_mask must match targets")

        valid = (
            ~target_mask.bool()
            if target_mask is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        origin = current_price[:, None]
        target_ticks = (targets - origin) / self.tick_size
        prediction_ticks = (predictions - origin[:, :, None]) / self.tick_size

        # 9 个分位数的概率预测。
        # 使用 Pinball loss 会让不同输出头学习不同条件分位数。而 mse 只能学习一个条件均值，是不够的。
        errors = target_ticks[:, :, None] - prediction_ticks
        quantiles = self.quantiles_tensor.to(predictions.dtype)[None, None, :]
        pinball_values = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        pinball = _masked_mean(pinball_values, valid[:, :, None].expand_as(pinball_values))

        # P50 点预测稳定性。主要是希望所有 quantile 都合理，P59 特别准确。
        # 所以 Pinball 负责整体概率分布，Huber 对 P50 额外加权。
        # Huber loss，误差小于 1 ticker 使用平方损失；误差大于 1 ticker 退化为线性损失
        # 为什么加入 Huber。单独使用 Pinball 时，P50 基本是 MAE（梯度在零点不平滑，对小误差没有精细校正，优化可能抖动）
        median_values = F.huber_loss(
            prediction_ticks[:, :, self.median_index],
            target_ticks,
            reduction="none",
            delta=self.huber_delta_ticks,
        )
        median_huber = _masked_mean(median_values, valid)

        # 分位数顺序错误
        # 只比较相邻 quantile，单步成立整体成立
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
