"""Typed experiment configuration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class DataConfig:
    train_path: str
    val_path: str
    context_length: int = 512
    horizon_length: int = 60
    max_variates: int = 32


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = "google/timesfm-3.0-pytorch"
    tuning_mode: Literal["head", "lora", "partial", "full"] = "lora"
    last_n_layers: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    disable_linear_detrending: bool = False


@dataclasses.dataclass(frozen=True)
class LossConfig:
    tick_size: float
    pinball_weight: float = 1.0
    median_huber_weight: float = 0.5
    crossing_weight: float = 0.05
    huber_delta_ticks: float = 1.0


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "outputs/weighted-mid"
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    device: str = "auto"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    loss: LossConfig
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        return cls(
            data=DataConfig(**raw["data"]),
            loss=LossConfig(**raw["loss"]),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if self.data.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.data.horizon_length <= 0:
            raise ValueError("horizon_length must be positive")
        if self.data.max_variates < 1 or self.data.max_variates > 32:
            raise ValueError("max_variates must be in [1, 32]")
        if self.loss.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.model.last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        if self.model.tuning_mode == "lora" and self.model.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.train.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.train.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
