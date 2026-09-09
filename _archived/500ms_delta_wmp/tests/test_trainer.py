from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from timesfm_ft import trainer
from timesfm_ft.config import (
    AdapterConfig,
    DataConfig,
    ExperimentConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrainerConfig,
)
from timesfm_ft.losses import ForecastLoss


class _BatchDataset(Dataset):
    def __init__(self, samples: int, context: int, horizon: int) -> None:
        self.context = torch.ones(samples, 1, context)
        self.future = torch.ones(samples, horizon)

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "context_values": self.context[index],
            "context_mask": torch.zeros_like(self.context[index], dtype=torch.bool),
            "future_values": self.future[index],
            "future_mask": torch.zeros_like(self.future[index], dtype=torch.bool),
        }


class _TinyForecast(nn.Module):
    quantiles = (0.1, 0.5, 0.9)

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        context_values: torch.Tensor,
        *,
        horizon: int,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context_mask
        origin = context_values[:, 0, -1, None, None]
        return origin + self.offset.expand(len(context_values), horizon, 3)


class _CountingSgd(torch.optim.SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=0.01)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def test_run_epoch_handles_partial_final_accumulation_group():
    model = _TinyForecast()
    loader = DataLoader(_BatchDataset(5, 16, 6), batch_size=2)
    optimizer = _CountingSgd(model.parameters())
    loss = ForecastLoss(model.quantiles, tick_size=0.01)
    metrics = trainer._run_epoch(
        model,
        loader,
        loss,
        device=torch.device("cpu"),
        horizon=6,
        sampling_interval_seconds=0.5,
        optimizer=optimizer,
        scheduler=None,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        epoch=1,
        split="train",
        log_every_steps=10,
    )
    assert optimizer.step_count == 2
    assert np.isfinite(metrics["loss"])
    val_metrics = trainer._run_epoch(
        model,
        loader,
        loss,
        device=torch.device("cpu"),
        horizon=6,
        sampling_interval_seconds=0.5,
        optimizer=None,
        scheduler=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        epoch=1,
        split="val",
        log_every_steps=10,
    )
    assert val_metrics["valid_points"] == 30


class _CheckpointModel(_TinyForecast):
    compute_dtype = "float32"
    checkpoint = "tiny"
    tuning_mode = "head"
    trainable_names = ("offset",)

    @property
    def parameter_summary(self) -> dict[str, int]:
        return {"total": 1, "trainable": 1}

    def optimizer_parameter_groups(self, config: OptimizerConfig):
        return [
            {
                "params": [self.offset],
                "lr": config.head_learning_rate,
                "weight_decay": 0.0,
                "group_name": "head",
            }
        ]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {"offset": self.offset.detach().cpu()}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.offset.data.copy_(state["offset"])

    def save_adapter(self, output_dir: Path, *, metadata=None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state_dict(), output_dir / "adapter.pt")
        (output_dir / "adapter_config.json").write_text(json.dumps(metadata or {}))


def _experiment(tmp_path: Path, *, resume_from: str | None = None) -> ExperimentConfig:
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    for path in (train_path, val_path):
        np.savez(
            path,
            context_values=np.ones((4, 16), dtype=np.float32),
            future_values=np.ones((4, 6), dtype=np.float32),
        )
    return ExperimentConfig(
        data=DataConfig(
            train_path=str(train_path),
            val_path=str(val_path),
            context_length=16,
            horizon_length=6,
            stride=6,
            max_variates=1,
        ),
        objective=ObjectiveConfig(tick_size=0.01),
        adapter=AdapterConfig(type="head", last_n_layers=1),
        trainer=TrainerConfig(
            output_dir=str(tmp_path / "output"),
            epochs=5,
            batch_size=2,
            num_workers=0,
            gradient_accumulation_steps=1,
            log_every_steps=10,
            device="cpu",
            dtype="float32",
            deterministic=False,
            early_stopping_patience=1,
            checkpoint_metric="rmse_ticks",
            resume_from=resume_from,
        ),
    )


def _epoch_metrics(*args, split: str, epoch: int, **kwargs):
    del args, kwargs
    rmse = 1.0 if epoch == 1 else 1.1
    return {
        "loss": rmse,
        "pinball": rmse,
        "huber": rmse,
        "crossing": 0.0,
        "rmse_ticks": rmse,
        "oos_r2_vs_persistence": 0.0,
        "samples_per_second": 1.0,
        "split": split,
    }


def test_train_experiment_early_stops_and_writes_resumable_state(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        trainer.TimesFM3Adapter,
        "from_pretrained",
        lambda *args, **kwargs: _CheckpointModel(),
    )
    monkeypatch.setattr(trainer, "_run_epoch", _epoch_metrics)

    output = trainer.train_experiment(_experiment(tmp_path))
    history = [
        json.loads(line)
        for line in (output / "history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 2
    assert (output / "best" / "adapter.pt").exists()
    assert (output / "best" / "training_state.pt").exists()
    assert (output / "last" / "adapter.pt").exists()
    assert (output / "last" / "training_state.pt").exists()

    resumed = trainer.train_experiment(
        _experiment(
            tmp_path,
            resume_from=str(output / "last" / "training_state.pt"),
        )
    )
    resumed_history = [
        json.loads(line)
        for line in (resumed / "history.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in resumed_history] == [1, 2, 3]


def test_set_seed_reproduces_torch_and_numpy_draws():
    trainer.set_seed(123, deterministic=True)
    first = (torch.rand(3), np.random.rand(3))
    trainer.set_seed(123, deterministic=True)
    second = (torch.rand(3), np.random.rand(3))
    torch.testing.assert_close(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    trainer.set_seed(0, deterministic=False)
