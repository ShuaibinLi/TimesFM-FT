from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from timesfm_ft.config import (
    DataConfig,
    ExperimentConfig,
    ObjectiveConfig,
    TrainerConfig,
)
from timesfm_ft.evaluator import EvaluationAccumulator, evaluate_experiment
from timesfm_ft.metrics import reconstruct_wmp_paths


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

    summary, rows = accumulator.results()
    assert summary["valid_points"] == 1
    assert summary["mae_ticks"] == 0.0
    assert summary["quantile_crossing_rate"] == pytest.approx(0.5)
    assert rows[0]["quantile_crossing_rate"] == pytest.approx(0.5)
    assert "coverage_q10" in rows[0]
    assert rows[1]["valid_points"] == 0


def test_evaluation_accumulator_rejects_nonfinite_predictions():
    accumulator = EvaluationAccumulator(
        horizon=1,
        quantiles=(0.1, 0.5, 0.9),
        tick_size=1.0,
        sampling_interval_seconds=0.5,
    )
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.update(
            torch.tensor([[[0.0, float("nan"), 1.0]]]),
            torch.zeros(1, 1),
            current_price=torch.zeros(1),
            target_mask=torch.zeros(1, 1, dtype=torch.bool),
        )


def test_delta_metrics_use_zero_and_last_delta_baselines():
    accumulator = EvaluationAccumulator(
        horizon=2,
        quantiles=(0.1, 0.5, 0.9),
        tick_size=0.015625,
        sampling_interval_seconds=0.5,
        target_mode="delta_ticks",
    )
    targets = torch.tensor([[0.5, -0.25]])
    predictions = targets[:, :, None].repeat(1, 1, 3)
    accumulator.update(
        predictions,
        targets,
        current_price=torch.tensor([0.5]),
        target_mask=torch.zeros_like(targets, dtype=torch.bool),
    )
    summary, rows = accumulator.results()
    assert summary["rmse_ticks"] == 0.0
    assert summary["persistence_rmse_ticks"] == pytest.approx(
        (0.15625) ** 0.5
    )
    assert summary["last_delta_rmse_ticks"] == pytest.approx(
        (0.28125) ** 0.5
    )
    assert summary["directional_accuracy"] == 1.0
    assert rows[0]["oos_r2_vs_persistence"] == 1.0


def test_reconstructs_price_paths_from_delta_ticks():
    context_wmp, future_wmp = reconstruct_wmp_paths(
        np.array([1.0, -0.5]),
        np.array([[0.5, 1.0], [-1.0, 0.0]]),
        cutoff_wmp=100.0,
        tick_size=0.25,
    )
    np.testing.assert_allclose(context_wmp, [100.125, 100.0])
    np.testing.assert_allclose(
        future_wmp,
        [[100.125, 100.25], [99.875, 100.25]],
    )


class _PersistenceModel:
    quantiles = (0.1, 0.5, 0.9)

    def eval(self):
        return self

    def predict(self, context_values, *, horizon, context_mask=None):
        del context_mask
        origin = context_values[:, 0, -1, None, None]
        return origin.expand(len(context_values), horizon, 3)


def _write_eval_data(path, samples):
    np.savez(
        path,
        context_values=np.ones((samples, 4), dtype=np.float32),
        future_values=np.ones((samples, 2), dtype=np.float32),
    )


def test_evaluate_defaults_to_configured_test_split(monkeypatch, tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    test = tmp_path / "test.npz"
    _write_eval_data(train, 2)
    _write_eval_data(val, 3)
    _write_eval_data(test, 5)
    config = ExperimentConfig(
        data=DataConfig(
            train_path=str(train),
            val_path=str(val),
            test_path=str(test),
            context_length=4,
            horizon_length=2,
            max_variates=1,
        ),
        objective=ObjectiveConfig(tick_size=1.0),
        trainer=TrainerConfig(
            output_dir=str(tmp_path / "output"),
            batch_size=2,
            device="cpu",
            dtype="float32",
        ),
    )
    monkeypatch.setattr(
        "timesfm_ft.evaluator.TimesFM3Adapter.from_pretrained",
        lambda *args, **kwargs: _PersistenceModel(),
    )
    destination = evaluate_experiment(config)
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["samples"] == 5
    assert summary["evaluated_split"] == "test"
    with pytest.raises(ValueError, match="train_path"):
        evaluate_experiment(config, data_path=train)
    copied_train = tmp_path / "copied_train.npz"
    _write_eval_data(copied_train, 2)
    copied_train.with_suffix(".json").write_text(
        json.dumps({"split": "train"})
    )
    with pytest.raises(ValueError, match="declares split='train'"):
        evaluate_experiment(config, data_path=copied_train)
    unsafe = evaluate_experiment(
        config,
        data_path=copied_train,
        output_dir=tmp_path / "unsafe",
        allow_unsafe_data=True,
    )
    unsafe_summary = json.loads((unsafe / "summary.json").read_text())
    assert unsafe_summary["evaluated_split"] == "explicit"
