from __future__ import annotations

import pytest
import torch

from timesfm_ft.config import ObjectiveConfig
from timesfm_ft.losses import (
    BusinessForecastLoss,
    LossScaleState,
    ScaleEstimate,
)

QUANTILES = (0.1, 0.5, 0.9)


def _estimate(value: float) -> ScaleEstimate:
    return ScaleEstimate(
        value=value,
        estimator="mad",
        valid_count=10,
        raw_value=value,
    )


def _loss(
    objective: ObjectiveConfig,
    *,
    cumulative: dict[int, float] | None = None,
    auxiliary: dict[str, float] | None = None,
) -> BusinessForecastLoss:
    return BusinessForecastLoss(
        QUANTILES,
        objective=objective,
        scales=LossScaleState(
            dataset_id="train",
            date_file_sha256="train-dates",
            feature_schema_sha256=None,
            manifest_sha256="train-manifest",
            sampling_contract={"training_route": objective.name},
            cumulative_method=objective.cumulative_scale_method,
            auxiliary_method=objective.auxiliary_scale_method,
            cumulative={horizon: _estimate(value) for horizon, value in (cumulative or {}).items()},
            auxiliary={feature: _estimate(value) for feature, value in (auxiliary or {}).items()},
        ),
    )


def test_f0_final_is_target_pinball_only_and_respects_masks():
    objective = ObjectiveConfig(name="f0_final")
    loss = _loss(objective)
    targets = torch.tensor([[0.5, 100.0]])
    predictions = targets[:, :, None].repeat(1, 1, 3)
    predictions[:, 1] = float("nan")
    result = loss(
        predictions.to(torch.bfloat16),
        targets,
        target_mask=torch.tensor([[False, True]]),
    )
    torch.testing.assert_close(result.total, torch.tensor(0.0))
    assert result.total.dtype == torch.float32
    assert result.cumulative_count == 0
    assert result.auxiliary_count == 0


def test_f0_all_pinball_uses_only_eligible_dense_anchors():
    objective = ObjectiveConfig(name="f0_all")
    loss = _loss(objective)
    predictions = torch.full((1, 3, 4, 3), float("nan"))
    predictions[:, 1] = 1.0
    targets = torch.zeros(1, 3, 4)
    result = loss(
        predictions,
        targets,
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
        anchor_mask=torch.tensor([[False, True, False]]),
    )
    assert float(result.return_pinball) == pytest.approx(0.5)
    assert result.return_count == 12


def test_dense_pinball_gradient_support_equals_supervision_mask():
    objective = ObjectiveConfig(name="f0_all")
    loss = _loss(objective)
    predictions = torch.zeros(1, 3, 4, 3, requires_grad=True)
    targets = torch.ones(1, 3, 4)
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    target_mask[:, 1, 2] = True
    result = loss(
        predictions,
        targets,
        target_mask=target_mask,
        anchor_mask=torch.tensor([[False, True, False]]),
    )
    gradient = torch.autograd.grad(result.return_pinball, predictions)[0]
    assert torch.count_nonzero(gradient[:, 0]) == 0
    assert torch.count_nonzero(gradient[:, 2]) == 0
    assert torch.count_nonzero(gradient[:, 1, 2]) == 0
    assert torch.count_nonzero(gradient[:, 1, [0, 1, 3]]) == 9


def test_f1_adds_train_scaled_final_cumulative_median_huber():
    objective = ObjectiveConfig(
        name="f1",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(5,),
    )
    loss = _loss(objective, cumulative={5: 5.0})
    targets = torch.zeros(2, 5)
    predictions = torch.zeros(2, 5, 3)
    predictions[:, :, 1] = 1.0
    result = loss(
        predictions,
        targets,
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
    )
    torch.testing.assert_close(result.cumulative_huber, torch.tensor(0.5))
    torch.testing.assert_close(
        result.total,
        result.return_pinball + 0.3 * result.cumulative_huber,
    )
    assert result.cumulative_count == 2


def test_cumulative_horizon_is_zero_weighted_when_path_contains_invalid_step():
    objective = ObjectiveConfig(
        name="f1",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(5,),
    )
    loss = _loss(objective, cumulative={5: 1.0})
    targets = torch.zeros(1, 5)
    mask = torch.tensor([[False, False, True, False, False]])
    predictions = torch.ones(1, 5, 3)
    result = loss(predictions, targets, target_mask=mask)
    assert result.cumulative_count == 0
    torch.testing.assert_close(result.cumulative_huber, torch.tensor(0.0))


def test_cumulative_huber_weights_each_horizon_equally():
    objective = ObjectiveConfig(
        name="f1",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2, 3),
    )
    loss = _loss(objective, cumulative={2: 1.0, 3: 1.0})
    targets = torch.zeros(2, 3)
    predictions = torch.zeros(2, 3, 3)
    predictions[:, :, 1] = 1.0
    mask = torch.tensor([[False, False, True], [False, False, False]])
    result = loss(predictions, targets, target_mask=mask)
    torch.testing.assert_close(result.cumulative_huber, torch.tensor(2.0))


def test_cumulative_huber_gradients_touch_only_p50_and_relevant_prefix():
    objective = ObjectiveConfig(
        name="f1",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2,),
    )
    loss = _loss(objective, cumulative={2: 1.0})
    predictions = torch.ones(1, 4, 3, requires_grad=True)
    result = loss(
        predictions,
        torch.zeros(1, 4),
        target_mask=torch.zeros(1, 4, dtype=torch.bool),
    )
    gradient = torch.autograd.grad(result.cumulative_huber, predictions)[0]
    assert torch.count_nonzero(gradient[:, :2, 1]) == 2
    assert torch.count_nonzero(gradient[:, 2:, 1]) == 0
    assert torch.count_nonzero(gradient[:, :, 0]) == 0
    assert torch.count_nonzero(gradient[:, :, 2]) == 0


def test_cumulative_huber_is_invariant_to_joint_unit_rescaling():
    objective = ObjectiveConfig(
        name="f1",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2,),
    )
    targets = torch.zeros(1, 2)
    predictions = torch.zeros(1, 2, 3)
    predictions[:, :, 1] = 0.5
    base = _loss(objective, cumulative={2: 2.0})(
        predictions,
        targets,
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
    )
    scaled = _loss(objective, cumulative={2: 20.0})(
        predictions * 10,
        targets * 10,
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
    )
    torch.testing.assert_close(base.cumulative_huber, scaled.cumulative_huber)


def test_f1_mv_supervises_only_explicit_selected_past_only_rows():
    objective = ObjectiveConfig(
        name="f1_mv",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2,),
        auxiliary_weight=0.05,
        auxiliary_features=("spread",),
    )
    loss = _loss(
        objective,
        cumulative={2: 1.0},
        auxiliary={"spread": 2.0},
    )
    targets = torch.zeros(1, 2)
    predictions = torch.zeros(1, 2, 3)
    auxiliary_targets = torch.zeros(1, 1, 2)
    auxiliary_predictions = torch.ones(1, 1, 2, 3)
    result = loss(
        predictions,
        targets,
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
        auxiliary_predictions=auxiliary_predictions,
        auxiliary_targets=auxiliary_targets,
        auxiliary_mask=torch.zeros_like(auxiliary_targets, dtype=torch.bool),
    )
    assert result.auxiliary_pinball == pytest.approx(0.25)
    torch.testing.assert_close(
        result.total,
        0.05 * result.auxiliary_pinball,
    )
    assert result.auxiliary_count == 6


def test_auxiliary_loss_weights_features_equally_despite_missingness():
    objective = ObjectiveConfig(
        name="f1_mv",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2,),
        auxiliary_weight=0.05,
        auxiliary_features=("spread", "volume"),
    )
    loss = _loss(
        objective,
        cumulative={2: 1.0},
        auxiliary={"spread": 1.0, "volume": 1.0},
    )
    targets = torch.zeros(1, 2)
    auxiliary_targets = torch.zeros(1, 2, 2)
    auxiliary_predictions = torch.zeros(1, 2, 2, 3)
    auxiliary_predictions[:, 0] = 1.0
    auxiliary_mask = torch.tensor([[[False, False], [False, True]]])
    result = loss(
        torch.zeros(1, 2, 3),
        targets,
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
        auxiliary_predictions=auxiliary_predictions,
        auxiliary_targets=auxiliary_targets,
        auxiliary_mask=auxiliary_mask,
    )
    assert float(result.auxiliary_pinball) == pytest.approx(0.25)


def test_loss_rejects_non_train_or_mismatched_scales():
    objective = ObjectiveConfig(
        name="f1",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(5,),
    )
    scales = LossScaleState(
        dataset_id="bad",
        date_file_sha256="val-dates",
        feature_schema_sha256=None,
        manifest_sha256="val-manifest",
        sampling_contract={"training_route": objective.name},
        cumulative_method="mad",
        auxiliary_method="mad",
        cumulative={5: _estimate(1.0)},
        auxiliary={},
        source_split="val",
    )
    with pytest.raises(ValueError, match="fitted on train"):
        BusinessForecastLoss(QUANTILES, objective=objective, scales=scales)
