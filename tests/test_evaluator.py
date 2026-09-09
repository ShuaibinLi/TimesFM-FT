from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from timesfm_ft.config import DataConfig, EvaluationConfig, ExperimentConfig, TrainerConfig
from timesfm_ft.evaluator import evaluate_experiment


class _Model:
    quantiles = (0.1, 0.5, 0.9)
    backbone = SimpleNamespace(input_patch_len=4)

    def eval(self):
        return self

    def predict(
        self,
        context_values,
        *,
        horizon,
        context_mask=None,
        context_padding_mask=None,
        past_future_values=None,
        past_future_mask=None,
    ):
        del (
            context_mask,
            context_padding_mask,
            past_future_values,
            past_future_mask,
        )
        last = context_values[:, 0, -1, None, None]
        return last.expand(len(context_values), horizon, 3)


def _config(bundle_factory, tmp_path):
    train, train_dates = bundle_factory("train", split="train", start_date=20250102)
    val, val_dates = bundle_factory("val", split="val", start_date=20250202)
    test, test_dates = bundle_factory("test", split="test", start_date=20250302)
    return ExperimentConfig(
        data=DataConfig(
            train_path=str(train),
            val_path=str(val),
            test_path=str(test),
            dataset_id="test_intraday",
            product="TEST",
            target_unit="test",
            target_price_source="test",
            context_min=4,
            context_max=8,
            horizon_length=3,
            stride=1,
            session_minutes=12,
            past_only_features=("p1",),
            past_future_features=("tod",),
            train_dates_path=str(train_dates),
            val_dates_path=str(val_dates),
            test_dates_path=str(test_dates),
        ),
        trainer=TrainerConfig(
            output_dir=str(tmp_path / "outputs"),
            batch_size=2,
            num_workers=0,
            device="cpu",
            dtype="float32",
            checkpoint_metric="mean_pinball",
            checkpoint_horizons=(3,),
        ),
        evaluation=EvaluationConfig(
            report_horizons=(1, 3),
            trading_horizon=3,
            save_predictions=True,
        ),
    )


def test_evaluator_defaults_to_test_and_writes_full_report(monkeypatch, bundle_factory, tmp_path):
    config = _config(bundle_factory, tmp_path)
    monkeypatch.setattr(
        "timesfm_ft.evaluator.TimesFM3Adapter.from_pretrained",
        lambda *args, **kwargs: _Model(),
    )
    destination = evaluate_experiment(config)
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["evaluated_split"] == "test"
    assert summary["samples"] == 12
    assert (destination / "per_lead.csv").exists()
    assert (destination / "cumulative_horizons.csv").exists()
    assert (destination / "slices.csv").exists()
    assert (destination / "predictions.npz").exists()


def test_evaluator_refuses_training_data(monkeypatch, bundle_factory, tmp_path):
    config = _config(bundle_factory, tmp_path)
    monkeypatch.setattr(
        "timesfm_ft.evaluator.TimesFM3Adapter.from_pretrained",
        lambda *args, **kwargs: _Model(),
    )
    with pytest.raises(ValueError, match="train_path"):
        evaluate_experiment(config, data_path=config.data.train_path)
