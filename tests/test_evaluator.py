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
