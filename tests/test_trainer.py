from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from timesfm_ft import trainer
from timesfm_ft.config import (
    AdapterConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ObjectiveConfig,
    TrainerConfig,
)
from timesfm_ft.losses import (
    BusinessForecastLoss,
    LossScaleState,
    ScaleEstimate,
)


class _TinyForecast(nn.Module):
    quantiles = (0.1, 0.5, 0.9)

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.1))
        self.backbone = SimpleNamespace(
            input_patch_len=2,
            output_patch_len=2,
        )

    def forward(
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
        origin = context_values[:, 0, -1, None, None]
        return origin + self.offset.expand(len(context_values), horizon, 3)

    def forward_all(
        self,
        context_values,
        *,
        horizon,
        context_mask=None,
        context_padding_mask=None,
        past_future_values=None,
        past_future_mask=None,
    ):
        target = self.forward(
            context_values,
            horizon=horizon,
            context_mask=context_mask,
            context_padding_mask=context_padding_mask,
            past_future_values=past_future_values,
            past_future_mask=past_future_mask,
        )
        past_only = target[:, None]
        known_future = torch.full_like(past_only, float("nan"))
        return torch.cat((target[:, None], past_only, known_future), dim=1)

    def forward_unknown(self, context_values, **kwargs):
        raw = self.forward_all(context_values, **kwargs)
        return SimpleNamespace(target=raw[:, 0], past_only=raw[:, 1:2])

    def forward_dense(
        self,
        values,
        *,
        masks,
        patch_is_target,
        unknown_variates,
    ):
        del masks, patch_is_target
        base = values[:, 0, :, -1, None, None] + self.offset
        target = base.expand(-1, -1, 2, 3)
        past_only = target[:, None].expand(-1, unknown_variates - 1, -1, -1, -1)
        return SimpleNamespace(target=target, past_only=past_only)


class _CountingSgd(torch.optim.SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=0.01)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def _batch(samples: int = 2, horizon: int = 2):
    return {
        "context_values": torch.ones(samples, 2, 4),
        "context_mask": torch.zeros(samples, 2, 4, dtype=torch.bool),
        "context_padding_mask": torch.zeros(samples, 4, dtype=torch.bool),
        "past_future_values": torch.ones(samples, 1, 4 + horizon),
        "past_future_mask": torch.zeros(samples, 1, 4 + horizon, dtype=torch.bool),
        "unknown_future_values": torch.ones(samples, 2, horizon),
        "unknown_future_mask": torch.zeros(samples, 2, horizon, dtype=torch.bool),
        "context_lengths": torch.full((samples,), 4, dtype=torch.int16),
        "timestamps": torch.arange(samples, dtype=torch.int64) + 1,
        "dates": torch.full((samples,), 20250102, dtype=torch.int32),
        "minute_indices": torch.full((samples,), 3, dtype=torch.int16),
        "last_returns": torch.ones(samples),
        "context_volatility": torch.ones(samples),
    }


def test_gradient_accumulation_steps_partial_final_group():
    model = _TinyForecast()
    loader = [_batch(), _batch(), _batch(1)]
    optimizer = _CountingSgd(model.parameters())
    evaluated_steps: list[int] = []
    metrics = trainer._run_epoch(
        model,
        loader,
        BusinessForecastLoss(
            model.quantiles,
            objective=ObjectiveConfig(name="f0_final"),
            scales=LossScaleState(
                dataset_id="test",
                date_file_sha256="test-dates",
                feature_schema_sha256=None,
                manifest_sha256="test-manifest",
                sampling_contract={"training_route": "f0_final"},
                cumulative_method="mad",
                auxiliary_method="mad",
                cumulative={},
                auxiliary={},
            ),
        ),
        device=torch.device("cpu"),
        horizon=2,
        context_min=2,
        evaluation=EvaluationConfig(report_horizons=(1, 2), trading_horizon=2),
        auxiliary_indices=torch.empty(0, dtype=torch.long),
        optimizer=optimizer,
        scheduler=None,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        epoch=1,
        split="train",
        log_every_steps=10,
        step_eval_interval=2,
        step_callback=evaluated_steps.append,
    )
    assert optimizer.step_count == 2
    assert evaluated_steps == [2]
    assert np.isfinite(metrics["mean_pinball"])


def test_f1_mv_routes_only_selected_past_only_dense_output():
    model = _TinyForecast()
    optimizer = _CountingSgd(model.parameters())
    objective = ObjectiveConfig(
        name="f1_mv",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2,),
        auxiliary_weight=0.05,
        auxiliary_features=("p1",),
    )
    metrics = trainer._run_epoch(
        model,
        [_batch()],
        BusinessForecastLoss(
            model.quantiles,
            objective=objective,
            scales=LossScaleState(
                dataset_id="test",
                date_file_sha256="test-dates",
                feature_schema_sha256=None,
                manifest_sha256="test-manifest",
                sampling_contract={"training_route": "f1_mv"},
                cumulative_method="mad",
                auxiliary_method="mad",
                cumulative={2: ScaleEstimate(1.0, "mad", 10, 1.0)},
                auxiliary={"p1": ScaleEstimate(1.0, "mad", 10, 1.0)},
            ),
        ),
        device=torch.device("cpu"),
        horizon=2,
        context_min=2,
        evaluation=EvaluationConfig(report_horizons=(1, 2), trading_horizon=2),
        auxiliary_indices=torch.tensor([0]),
        optimizer=optimizer,
        scheduler=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        epoch=1,
        split="train",
        log_every_steps=10,
    )
    assert np.isfinite(metrics["loss"])
    assert metrics["auxiliary_pinball"] > 0


def test_business_checkpoint_metrics_are_maximized():
    metrics = {
        "mean_pinball": 1.0,
        "rmse": 2.0,
        "cumulative_horizons": [
            {"horizon_minutes": 5, "ic": 0.5, "rank_ic": 0.4, "mean_daily_rank_ic": 0.1},
            {"horizon_minutes": 15, "ic": 0.4, "rank_ic": 0.3, "mean_daily_rank_ic": 0.2},
            {"horizon_minutes": 30, "ic": 0.3, "rank_ic": 0.2, "mean_daily_rank_ic": 0.3},
            {"horizon_minutes": 60, "ic": 0.2, "rank_ic": 0.1, "mean_daily_rank_ic": 0.4},
        ],
        "trading_proxy": {"net_mean": 0.03},
    }
    assert trainer._checkpoint_value(
        metrics,
        metric="mean_daily_rank_ic",
        horizons=(5, 15, 30, 60),
    ) == (
        0.25,
        "max",
    )
    assert trainer._checkpoint_value(
        metrics,
        metric="ic",
        horizons=(5, 15, 30, 60),
    ) == (0.35, "max")
    rank_value, rank_mode = trainer._checkpoint_value(
        metrics,
        metric="rank_ic",
        horizons=(5, 15, 30, 60),
    )
    assert rank_value == pytest.approx(0.25)
    assert rank_mode == "max"
    assert trainer._checkpoint_value(metrics, metric="net_utility", horizons=(60,)) == (0.03, "max")
    assert trainer._checkpoint_value(metrics, metric="mean_pinball", horizons=(60,)) == (1.0, "min")


def test_scale_fallback_is_explicit_and_fingerprinted():
    estimate = trainer._robust_scale(np.ones(5), "mad")
    assert estimate.raw_value == 0.0
    assert estimate.value == 1.0
    assert estimate.fallback == "unit"
    assert estimate.valid_count == 5


class _CheckpointModel(_TinyForecast):
    compute_dtype = "float32"
    checkpoint = "tiny"
    tuning_mode = "head"
    trainable_names = ("offset",)

    @property
    def parameter_summary(self):
        return {"total": 1, "trainable": 1}

    def optimizer_parameter_groups(self, config):
        return [
            {
                "params": [self.offset],
                "lr": config.head_learning_rate,
                "weight_decay": 0.0,
                "group_name": "head",
            }
        ]

    def trainable_state_dict(self):
        return {"offset": self.offset.detach().cpu()}

    def load_trainable_state_dict(self, state):
        self.offset.data.copy_(state["offset"])

    def save_adapter(self, output_dir: Path, *, metadata=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state_dict(), output_dir / "adapter.pt")
        (output_dir / "adapter_config.json").write_text(json.dumps(metadata or {}))


def _experiment(bundle_factory, tmp_path, *, resume_from=None):
    train, train_dates = bundle_factory("train", split="train", start_date=20250102)
    val, val_dates = bundle_factory("val", split="val", start_date=20250202)
    return ExperimentConfig(
        data=DataConfig(
            train_path=str(train),
            val_path=str(val),
            dataset_id="test_intraday",
            product="TEST",
            target_unit="test",
            target_price_source="test",
            context_min=4,
            context_max=6,
            horizon_length=3,
            stride=2,
            session_minutes=12,
            past_only_features=("p1",),
            past_future_features=("tod",),
            train_dates_path=str(train_dates),
            val_dates_path=str(val_dates),
        ),
        adapter=AdapterConfig(type="head", last_n_layers=1),
        trainer=TrainerConfig(
            output_dir=str(tmp_path / "output"),
            epochs=3,
            batch_size=2,
            num_workers=0,
            log_every_steps=10,
            device="cpu",
            dtype="float32",
            deterministic=False,
            early_stopping_patience=1,
            checkpoint_metric="mean_pinball",
            checkpoint_horizons=(3,),
            resume_from=resume_from,
        ),
        evaluation=EvaluationConfig(
            report_horizons=(1, 3),
            trading_horizon=3,
            save_predictions=False,
        ),
    )


def test_loss_scales_are_fitted_from_declared_training_bundle(
    bundle_factory,
    tmp_path,
):
    config = _experiment(bundle_factory, tmp_path)
    dataset = trainer._dataset(
        config,
        path=config.data.train_path,
        split="train",
        dates_path=config.data.train_dates_path,
    )
    objective = ObjectiveConfig(
        name="f1_mv",
        cumulative_huber_weight=0.3,
        cumulative_horizons=(2, 3),
        auxiliary_weight=0.05,
        auxiliary_features=("p1",),
    )
    scales = trainer.fit_loss_scales(
        dataset,
        objective,
        input_patch_length=1,
        output_patch_length=3,
        context_min=4,
        batch_size=2,
    )
    assert scales.source_split == "train"
    assert set(scales.cumulative) == {2, 3}
    assert scales.auxiliary["p1"].value > 0
    assert scales.auxiliary["p1"].valid_count > 0
    assert len(scales.fingerprint) == 64
    val_dataset = trainer._dataset(
        config,
        path=config.data.val_path,
        split="val",
        dates_path=config.data.val_dates_path,
    )
    with pytest.raises(ValueError, match="split=train"):
        trainer.fit_loss_scales(val_dataset, objective)


def test_training_rejects_cross_split_feature_schema_drift(
    bundle_factory,
    tmp_path,
):
    config = _experiment(bundle_factory, tmp_path)
    manifest_path = Path(config.data.val_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_schema_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="feature_schema_sha256"):
        trainer.train_experiment(config)


def _epoch_metrics(*args, split, epoch, **kwargs):
    del args, kwargs
    value = 1.0 if epoch == 1 else 1.1
    result = {
        "loss": value,
        "mean_pinball": value,
        "return_pinball": value,
        "cumulative_huber": 0.0,
        "auxiliary_pinball": 0.0,
        "rmse": value,
        "samples_per_second": 1.0,
    }
    if split == "val":
        result.update(
            {
                "cumulative_horizons": [
                    {
                        "horizon_minutes": 1,
                        "ic": 0.0,
                        "mean_daily_rank_ic": 0.0,
                    },
                    {
                        "horizon_minutes": 3,
                        "ic": 0.0,
                        "mean_daily_rank_ic": 0.0,
                    },
                ],
                "slices": [],
                "trading_proxy": {"net_mean": 0.0},
                "mean_absolute_coverage_error": 0.0,
                "q10_q90_coverage": 0.8,
                "mean_q10_q90_width": 1.0,
                "quantile_crossing_rate": 0.0,
            }
        )
    return result


def test_training_writes_versioned_resumable_state(monkeypatch, bundle_factory, tmp_path):
    monkeypatch.setattr(
        trainer.TimesFM3Adapter,
        "from_pretrained",
        lambda *args, **kwargs: _CheckpointModel(),
    )
    monkeypatch.setattr(trainer, "_run_epoch", _epoch_metrics)
    output = trainer.train_experiment(_experiment(bundle_factory, tmp_path))
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(history) == 2
    state = torch.load(
        output / "last/training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["format_version"] == 4
    assert state["loss_scales"]["source_split"] == "train"
    scales_file = json.loads((output / "loss_scales.json").read_text())
    assert scales_file["fingerprint"] == state["loss_scales"]["fingerprint"]
    assert (output / "best/adapter.pt").exists()
    adapter_metadata = json.loads((output / "best/adapter_config.json").read_text())
    assert adapter_metadata["training_graph"] == "deployment_suffix_decode"
