from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from timesfm_ft import trainer
from timesfm_ft.config import (
    AdapterConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainerConfig,
)
from timesfm_ft.losses import PinballLoss


class _TinyForecast(nn.Module):
    quantiles = (0.1, 0.5, 0.9)

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.1))
        self.backbone = SimpleNamespace(input_patch_len=2)

    def forward(
        self,
        context_values,
        *,
        horizon,
        context_mask=None,
        past_future_values=None,
        past_future_mask=None,
    ):
        del context_mask, past_future_values, past_future_mask
        origin = context_values[:, 0, -1, None, None]
        return origin + self.offset.expand(len(context_values), horizon, 3)


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
        "past_future_values": torch.ones(samples, 1, 4 + horizon),
        "past_future_mask": torch.zeros(samples, 1, 4 + horizon, dtype=torch.bool),
        "future_values": torch.ones(samples, horizon),
        "future_mask": torch.zeros(samples, horizon, dtype=torch.bool),
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
    metrics = trainer._run_epoch(
        model,
        loader,
        PinballLoss(model.quantiles),
        device=torch.device("cpu"),
        horizon=2,
        evaluation=EvaluationConfig(report_horizons=(1, 2), trading_horizon=2),
        optimizer=optimizer,
        scheduler=None,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        epoch=1,
        split="train",
        log_every_steps=10,
    )
    assert optimizer.step_count == 2
    assert np.isfinite(metrics["mean_pinball"])


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
            resume_from=resume_from,
        ),
        evaluation=EvaluationConfig(
            report_horizons=(1, 3),
            trading_horizon=3,
            save_predictions=False,
        ),
    )


def _epoch_metrics(*args, split, epoch, **kwargs):
    del args, kwargs
    value = 1.0 if epoch == 1 else 1.1
    result = {
        "loss": value,
        "mean_pinball": value,
        "rmse": value,
        "samples_per_second": 1.0,
    }
    if split == "val":
        result.update(
            {
                "cumulative_horizons": [
                    {"horizon_minutes": 1, "ic": 0.0},
                    {"horizon_minutes": 3, "ic": 0.0},
                ],
                "slices": [],
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
    assert state["format_version"] == 2
    assert (output / "best/adapter.pt").exists()
