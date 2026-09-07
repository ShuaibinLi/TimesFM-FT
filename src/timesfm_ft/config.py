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
    disable_linear_detrending: bool = False


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    type: Literal["head", "lora", "partial", "full"] = "lora"
    last_n_layers: int = 4
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05


@dataclasses.dataclass(frozen=True)
class ObjectiveConfig:
    tick_size: float
    pinball_weight: float = 1.0
    median_huber_weight: float = 0.5
    crossing_weight: float = 0.05
    huber_delta_ticks: float = 1.0


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adamw"] = "adamw"
    adapter_learning_rate: float = 1e-4
    head_learning_rate: float = 3e-4
    pretrained_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    name: Literal["cosine"] = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    output_dir: str = "outputs/weighted-mid"
    epochs: int = 10
    batch_size: int = 16
    max_grad_norm: float = 1.0
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 10
    seed: int = 42
    device: str = "auto"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    objective: ObjectiveConfig
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    adapter: AdapterConfig = dataclasses.field(default_factory=AdapterConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        return cls(
            data=DataConfig(**raw["data"]),
            objective=ObjectiveConfig(**raw["objective"]),
            model=ModelConfig(**raw.get("model", {})),
            adapter=AdapterConfig(**raw.get("adapter", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            trainer=TrainerConfig(**raw.get("trainer", {})),
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
        if self.objective.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.adapter.last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        if self.adapter.type == "lora" and self.adapter.rank <= 0:
            raise ValueError("adapter rank must be positive")
        if self.adapter.alpha <= 0:
            raise ValueError("adapter alpha must be positive")
        if not 0 <= self.adapter.dropout < 1:
            raise ValueError("adapter dropout must be in [0, 1)")
        objective_weights = (
            self.objective.pinball_weight,
            self.objective.median_huber_weight,
            self.objective.crossing_weight,
        )
        if any(weight < 0 for weight in objective_weights):
            raise ValueError("objective weights must be non-negative")
        learning_rates = (
            self.optimizer.adapter_learning_rate,
            self.optimizer.head_learning_rate,
            self.optimizer.pretrained_learning_rate,
        )
        if any(learning_rate <= 0 for learning_rate in learning_rates):
            raise ValueError("all learning rates must be positive")
        if self.optimizer.name != "adamw":
            raise ValueError("only optimizer.name='adamw' is currently supported")
        if not 0 <= self.optimizer.weight_decay:
            raise ValueError("weight_decay must be non-negative")
        if not 0 < self.optimizer.beta1 < 1 or not 0 < self.optimizer.beta2 < 1:
            raise ValueError("optimizer beta1 and beta2 must be in (0, 1)")
        if self.optimizer.eps <= 0:
            raise ValueError("optimizer eps must be positive")
        if self.scheduler.name != "cosine":
            raise ValueError("only scheduler.name='cosine' is currently supported")
        if not 0 <= self.scheduler.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.scheduler.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.trainer.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.trainer.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.trainer.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.trainer.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")
        if self.trainer.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.trainer.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
