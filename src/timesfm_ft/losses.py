"""Business-aligned probabilistic objectives for intraday returns."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from timesfm_ft.config import ObjectiveConfig


@dataclasses.dataclass(frozen=True)
class ScaleEstimate:
    value: float
    estimator: str
    valid_count: int
    raw_value: float
    fallback: str | None = None

    def validate(self) -> None:
        if (
            not math.isfinite(self.value)
            or self.value <= 0
            or not math.isfinite(self.raw_value)
            or self.raw_value < 0
            or self.valid_count <= 0
        ):
            raise ValueError("invalid fitted scale estimate")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LossScaleState:
    """Robust scales fitted exclusively from the training split."""

    dataset_id: str
    date_file_sha256: str
    feature_schema_sha256: str | None
    manifest_sha256: str
    sampling_contract: dict[str, Any]
    cumulative_method: str
    auxiliary_method: str
    cumulative: dict[int, ScaleEstimate]
    auxiliary: dict[str, ScaleEstimate]
    source_split: str = "train"
    format_version: int = 2

    def validate(self, objective: ObjectiveConfig) -> None:
        if self.format_version != 2 or self.source_split != "train":
            raise ValueError("loss scales must be version 2 and fitted on train")
        if not self.dataset_id or not self.date_file_sha256 or not self.manifest_sha256:
            raise ValueError("loss scales require training data provenance")
        if self.sampling_contract.get("training_route") != objective.name:
            raise ValueError("loss-scale training route mismatch")
        if self.cumulative_method != objective.cumulative_scale_method:
            raise ValueError("cumulative scale method does not match objective")
        if self.auxiliary_method != objective.auxiliary_scale_method:
            raise ValueError("auxiliary scale method does not match objective")
        if set(self.cumulative) != set(objective.cumulative_horizons):
            raise ValueError("cumulative scale horizons do not match objective")
        if set(self.auxiliary) != set(objective.auxiliary_features):
            raise ValueError("auxiliary scales do not match objective features")
        for estimate in self.cumulative.values():
            if estimate.estimator != self.cumulative_method:
                raise ValueError("cumulative estimate method mismatch")
            estimate.validate()
        for estimate in self.auxiliary.values():
            if estimate.estimator != self.auxiliary_method:
                raise ValueError("auxiliary estimate method mismatch")
            estimate.validate()

    def _payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "source_split": self.source_split,
            "dataset_id": self.dataset_id,
            "date_file_sha256": self.date_file_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sampling_contract": self.sampling_contract,
            "cumulative_method": self.cumulative_method,
            "auxiliary_method": self.auxiliary_method,
            "cumulative": {
                str(horizon): estimate.to_dict()
                for horizon, estimate in sorted(self.cumulative.items())
            },
            "auxiliary": {
                feature: estimate.to_dict() for feature, estimate in sorted(self.auxiliary.items())
            },
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self._payload() | {"fingerprint": self.fingerprint}


@dataclasses.dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    return_pinball: torch.Tensor
    lead1_pinball: torch.Tensor
    correlation: torch.Tensor
    cumulative_huber: torch.Tensor
    auxiliary_pinball: torch.Tensor
    return_count: int
    lead1_count: int
    correlation_count: int
    cumulative_count: int
    auxiliary_count: int


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _masked_mean_or_zero(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    count = valid.sum()
    if not torch.any(valid).item():
        return _zero(values)
    return torch.where(valid, values, 0.0).sum() / count.to(values.dtype)


class BusinessForecastLoss(nn.Module):
    """F0-final/F0-all/F1/F1-MV route objective."""

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float],
        *,
        objective: ObjectiveConfig,
        scales: LossScaleState,
    ) -> None:
        super().__init__()
        values = torch.tensor(quantiles, dtype=torch.float32)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("quantiles must be a non-empty sequence")
        if not torch.isfinite(values).all() or not torch.all((values > 0) & (values < 1)):
            raise ValueError("quantiles must be finite and in (0, 1)")
        if not torch.all(values[1:] > values[:-1]):
            raise ValueError("quantiles must be strictly increasing")
        median_index = int(torch.argmin(torch.abs(values - 0.5)).item())
        if abs(float(values[median_index]) - 0.5) > 1e-6:
            raise ValueError("cumulative Huber requires an explicit 0.5 quantile")
        scales.validate(objective)

        self.objective = objective
        self.scales = scales
        self.median_index = median_index
        self.register_buffer("quantiles_tensor", values, persistent=False)
        horizons = tuple(objective.cumulative_horizons)
        self.register_buffer(
            "cumulative_horizons",
            torch.tensor(horizons, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "cumulative_scales",
            torch.tensor(
                [scales.cumulative[horizon].value for horizon in horizons],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "auxiliary_scales",
            torch.tensor(
                [scales.auxiliary[feature].value for feature in objective.auxiliary_features],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @property
    def quantile_count(self) -> int:
        return int(self.quantiles_tensor.numel())

    def _pinball(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction_valid = valid[..., None].expand_as(predictions)
        if not torch.isfinite(targets[valid]).all().item():
            raise ValueError("valid targets contain non-finite values")
        if not torch.isfinite(predictions[prediction_valid]).all().item():
            raise ValueError("valid predictions contain non-finite values")
        error = targets[..., None] - predictions
        if scale is not None:
            error = error / scale
        quantiles = self.quantiles_tensor.to(predictions.device)
        quantile_shape = (1,) * (error.ndim - 1) + (self.quantile_count,)
        quantiles = quantiles.reshape(quantile_shape)
        values = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
        return _masked_mean_or_zero(values, prediction_valid)

    def _correlation_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if int(valid.sum().item()) < 2:
            return _zero(prediction)
        prediction_values = prediction[valid]
        target_values = target[valid]
        prediction_centered = prediction_values - prediction_values.mean()
        target_centered = target_values - target_values.mean()
        denominator = torch.sqrt(
            prediction_centered.square().sum() * target_centered.square().sum()
            + self.objective.correlation_eps
        )
        correlation = (prediction_centered * target_centered).sum() / denominator
        return 1.0 - correlation

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        anchor_mask: torch.Tensor | None = None,
        final_predictions: torch.Tensor | None = None,
        final_targets: torch.Tensor | None = None,
        final_target_mask: torch.Tensor | None = None,
        auxiliary_predictions: torch.Tensor | None = None,
        auxiliary_targets: torch.Tensor | None = None,
        auxiliary_mask: torch.Tensor | None = None,
        auxiliary_anchor_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        if predictions.ndim not in {3, 4}:
            raise ValueError("target predictions must be (B,H,Q) or dense (B,N,H,Q)")
        if targets.shape != predictions.shape[:-1] or target_mask.shape != targets.shape:
            raise ValueError("target values/mask must match predictions")
        if predictions.shape[-1] != self.quantile_count:
            raise ValueError("prediction quantile count mismatch")
        valid_target = ~target_mask.bool()
        if anchor_mask is not None:
            if predictions.ndim != 4 or anchor_mask.shape != targets.shape[:2]:
                raise ValueError("dense anchor_mask must have shape (B,N)")
            valid_target &= anchor_mask[:, :, None]
        if not torch.any(valid_target).item():
            raise ValueError("loss batch has no valid return target")
        predictions = predictions.float()
        targets = targets.float()
        return_pinball = self._pinball(predictions, targets, valid_target)
        return_count = int(valid_target.sum().item()) * self.quantile_count

        lead1_pinball = _zero(predictions)
        lead1_count = 0
        correlation = _zero(predictions)
        correlation_count = 0
        if self.objective.lead1_pinball_weight > 0 or self.objective.correlation_weight > 0:
            if predictions.ndim != 3:
                raise ValueError("lead1 objectives require final-anchor predictions")
            lead1_valid = valid_target[:, 0]
            if lead1_valid.any().item():
                lead1_pinball = self._pinball(
                    predictions[:, 0],
                    targets[:, 0],
                    lead1_valid,
                )
                lead1_count = int(lead1_valid.sum().item()) * self.quantile_count
                if self.objective.correlation_weight > 0:
                    correlation = self._correlation_loss(
                        predictions[:, 0, self.median_index],
                        targets[:, 0],
                        lead1_valid,
                    )
                    correlation_count = int(lead1_valid.sum().item())

        cumulative_huber = _zero(predictions)
        cumulative_count = 0
        if self.objective.cumulative_huber_weight > 0:
            if final_predictions is None:
                if predictions.ndim != 3:
                    raise ValueError("dense cumulative loss requires final_predictions")
                final_predictions = predictions
                final_targets = targets
                final_target_mask = target_mask
            if (
                final_targets is None
                or final_target_mask is None
                or final_predictions.ndim != 3
                or final_targets.shape != final_predictions.shape[:2]
                or final_target_mask.shape != final_targets.shape
            ):
                raise ValueError("invalid final-anchor cumulative tensors")
            final_predictions = final_predictions.float()
            final_targets = final_targets.float()
            final_valid = ~final_target_mask.bool()
            median = final_predictions[:, :, self.median_index]
            horizon_losses: list[torch.Tensor] = []
            for index, horizon_value in enumerate(self.cumulative_horizons):
                horizon = int(horizon_value.item())
                path_valid = final_valid[:, :horizon].all(dim=1)
                prediction_sum = torch.where(
                    final_valid[:, :horizon],
                    median[:, :horizon],
                    0.0,
                ).sum(dim=1)
                target_sum = torch.where(
                    final_valid[:, :horizon],
                    final_targets[:, :horizon],
                    0.0,
                ).sum(dim=1)
                normalized_error = (prediction_sum - target_sum) / self.cumulative_scales[index]
                values = F.huber_loss(
                    normalized_error,
                    torch.zeros_like(normalized_error),
                    reduction="none",
                    delta=self.objective.cumulative_huber_delta,
                )
                if path_valid.any().item():
                    horizon_losses.append(_masked_mean_or_zero(values, path_valid))
                    cumulative_count += int(path_valid.sum().item())
            if horizon_losses:
                cumulative_huber = torch.stack(horizon_losses).mean()

        auxiliary_pinball = _zero(predictions)
        auxiliary_count = 0
        if self.objective.auxiliary_weight > 0:
            if auxiliary_predictions is None or auxiliary_targets is None or auxiliary_mask is None:
                raise ValueError("F1-MV requires auxiliary predictions, targets, and mask")
            if auxiliary_predictions.ndim not in {4, 5}:
                raise ValueError("auxiliary predictions must be (B,A,H,Q) or (B,A,N,H,Q)")
            if (
                auxiliary_predictions.shape[0] != predictions.shape[0]
                or auxiliary_predictions.shape[1] != len(self.objective.auxiliary_features)
                or auxiliary_predictions.shape[-2] != predictions.shape[-2]
                or auxiliary_predictions.shape[-1] != self.quantile_count
            ):
                raise ValueError("auxiliary prediction shape mismatch")
            expected_target_shape = auxiliary_predictions.shape[:-1]
            if (
                auxiliary_targets.shape != expected_target_shape
                or auxiliary_mask.shape != expected_target_shape
            ):
                raise ValueError("auxiliary targets/mask shape mismatch")
            feature_losses: list[torch.Tensor] = []
            for feature_index in range(len(self.objective.auxiliary_features)):
                feature_valid = ~auxiliary_mask[:, feature_index].bool()
                if auxiliary_anchor_mask is not None:
                    expected_anchor_shape = (
                        auxiliary_targets.shape[0],
                        auxiliary_targets.shape[2],
                    )
                    if (
                        auxiliary_predictions.ndim != 5
                        or auxiliary_anchor_mask.shape != expected_anchor_shape
                    ):
                        raise ValueError("auxiliary_anchor_mask must have shape (B,N)")
                    feature_valid &= auxiliary_anchor_mask[:, :, None]
                if feature_valid.any().item():
                    feature_losses.append(
                        self._pinball(
                            auxiliary_predictions[:, feature_index].float(),
                            auxiliary_targets[:, feature_index].float(),
                            feature_valid,
                            scale=self.auxiliary_scales[feature_index],
                        )
                    )
                    auxiliary_count += int(feature_valid.sum().item()) * self.quantile_count
            if feature_losses:
                auxiliary_pinball = torch.stack(feature_losses).mean()

        total = (
            self.objective.return_pinball_weight * return_pinball
            + self.objective.lead1_pinball_weight * lead1_pinball
            + self.objective.correlation_weight * correlation
            + self.objective.cumulative_huber_weight * cumulative_huber
            + self.objective.auxiliary_weight * auxiliary_pinball
        )
        return LossOutput(
            total=total,
            return_pinball=return_pinball,
            lead1_pinball=lead1_pinball,
            correlation=correlation,
            cumulative_huber=cumulative_huber,
            auxiliary_pinball=auxiliary_pinball,
            return_count=return_count,
            lead1_count=lead1_count,
            correlation_count=correlation_count,
            cumulative_count=cumulative_count,
            auxiliary_count=auxiliary_count,
        )
