"""Typed configuration for the 1-minute intraday forecasting task."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class DataConfig:
    train_path: str
    val_path: str
    test_path: str | None = None
    dataset_id: str | None = None
    product: str = "ZN"
    target_name: str = "return_1m"
    target_unit: str = "ticks"
    target_price_source: str = "weighted_mid"
    target_return_type: Literal["simple", "log"] = "simple"
    target_timestamp_semantics: Literal["bar_start", "bar_end"] = "bar_end"
    target_availability_lag_minutes: int = 0
    target_missing_policy: Literal["mask"] = "mask"
    frequency_minutes: int = 1
    context_min: int = 64
    context_max: int = 192
    horizon_length: int = 64
    stride: int = 1
    session_minutes: int = 390
    max_variates: int = 32
    past_only_features: tuple[str, ...] = ()
    past_future_features: tuple[str, ...] = ()
    train_dates_path: str | None = None
    val_dates_path: str | None = None
    test_dates_path: str | None = None
    require_metadata: bool = True


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = "google/timesfm-3.0-pytorch"
    disable_linear_detrending: bool = False
    disable_iterative_cpm_revin: bool = True


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    type: Literal["head", "lora", "partial", "full"] = "head"
    last_n_layers: int = 4
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    lora_sequence_attention: bool = True
    lora_variate_attention: bool = True
    lora_feedforward: bool = True


@dataclasses.dataclass(frozen=True)
class ObjectiveConfig:
    name: Literal["f0_final", "f0_all", "f1", "f1_mv"] = "f0_final"
    return_pinball_weight: float = 1.0
    cumulative_huber_weight: float = 0.0
    cumulative_horizons: tuple[int, ...] = ()
    cumulative_scale_method: Literal["mad", "std"] = "mad"
    cumulative_huber_delta: float = 1.0
    auxiliary_weight: float = 0.0
    auxiliary_features: tuple[str, ...] = ()
    auxiliary_scale_method: Literal["mad", "std"] = "mad"

    @property
    def uses_dense_forward(self) -> bool:
        return self.name in {"f0_all", "f1", "f1_mv"}


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
    output_dir: str = "outputs/intraday-1min"
    epochs: int = 5
    batch_size: int = 32
    max_grad_norm: float = 1.0
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 10
    step_eval_interval: int | None = None
    seed: int = 42
    device: str = "auto"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    deterministic: bool = True
    early_stopping_patience: int | None = 2
    checkpoint_metric: Literal[
        "mean_pinball",
        "rmse",
        "ic",
        "rank_ic",
        "mean_daily_rank_ic",
        "net_utility",
    ] = "mean_daily_rank_ic"
    checkpoint_horizons: tuple[int, ...] = (5, 15, 30, 60)
    resume_from: str | None = None


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    report_horizons: tuple[int, ...] = (1, 5, 10, 15, 20, 30, 60)
    trading_horizon: int = 60
    cost_per_turnover: float = 0.0
    save_predictions: bool = True


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    adapter: AdapterConfig = dataclasses.field(default_factory=AdapterConfig)
    objective: ObjectiveConfig = dataclasses.field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path).resolve()

        def load_raw(current: Path, seen: frozenset[Path]) -> dict[str, Any]:
            if current in seen:
                raise ValueError(f"cyclic config inheritance at {current}")
            with current.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"config must be a JSON object: {current}")
            parent = value.pop("extends", None)
            if parent is None:
                return value
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = (current.parent / parent_path).resolve()
            base = load_raw(parent_path, seen | {current})
            for section, override in value.items():
                if isinstance(override, dict) and isinstance(base.get(section), dict):
                    base[section] = {**base[section], **override}
                else:
                    base[section] = override
            return base

        raw = load_raw(config_path, frozenset())
        base_dir = config_path.parent

        def resolve_path(value: str | None) -> str | None:
            if value is None or "://" in value or Path(value).is_absolute():
                return value
            return str((base_dir / value).resolve())

        data = dict(raw["data"])
        for key in ("past_only_features", "past_future_features"):
            data[key] = tuple(data.get(key, ()))
        for key in (
            "train_path",
            "val_path",
            "test_path",
            "train_dates_path",
            "val_dates_path",
            "test_dates_path",
        ):
            if key in data:
                data[key] = resolve_path(data[key])

        trainer = dict(raw.get("trainer", {}))
        if "checkpoint_horizons" in trainer:
            trainer["checkpoint_horizons"] = tuple(trainer["checkpoint_horizons"])
        if "output_dir" in trainer:
            trainer["output_dir"] = resolve_path(trainer["output_dir"])
        if "resume_from" in trainer:
            trainer["resume_from"] = resolve_path(trainer["resume_from"])

        model = dict(raw.get("model", {}))
        checkpoint = model.get("checkpoint")
        if isinstance(checkpoint, str) and checkpoint.startswith("."):
            model["checkpoint"] = resolve_path(checkpoint)

        evaluation = dict(raw.get("evaluation", {}))
        evaluation["report_horizons"] = tuple(
            evaluation.get("report_horizons", EvaluationConfig().report_horizons)
        )
        objective = dict(raw.get("objective", {}))
        for key in ("cumulative_horizons", "auxiliary_features"):
            if key in objective:
                objective[key] = tuple(objective[key])
        config = cls(
            data=DataConfig(**data),
            model=ModelConfig(**model),
            adapter=AdapterConfig(**raw.get("adapter", {})),
            objective=ObjectiveConfig(**objective),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            trainer=TrainerConfig(**trainer),
            evaluation=EvaluationConfig(**evaluation),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        data = self.data
        if data.frequency_minutes != 1:
            raise ValueError("v1 requires frequency_minutes=1")
        if data.target_availability_lag_minutes != 0:
            raise ValueError("v1 requires target_availability_lag_minutes=0")
        if data.target_timestamp_semantics != "bar_end":
            raise ValueError("v1.3 window alignment currently requires bar_end targets")
        if not 0 < data.context_min <= data.context_max:
            raise ValueError("context lengths must satisfy 0 < context_min <= context_max")
        if data.horizon_length <= 0 or data.stride <= 0:
            raise ValueError("horizon_length and stride must be positive")
        if data.session_minutes < data.context_min + data.horizon_length:
            raise ValueError("session is too short for context_min plus horizon")
        if not 1 <= data.max_variates <= 32:
            raise ValueError("max_variates must be in [1, 32]")
        if len(set(data.past_only_features)) != len(data.past_only_features):
            raise ValueError("past_only_features contains duplicates")
        if len(set(data.past_future_features)) != len(data.past_future_features):
            raise ValueError("past_future_features contains duplicates")
        overlap = set(data.past_only_features) & set(data.past_future_features)
        if overlap:
            raise ValueError(f"features cannot be both past-only and past-future: {overlap}")
        num_variates = 1 + len(data.past_only_features) + len(data.past_future_features)
        if num_variates > data.max_variates:
            raise ValueError(
                f"configured {num_variates} variates exceeds max_variates={data.max_variates}"
            )
        if data.train_path == data.val_path:
            raise ValueError("train_path and val_path must differ")
        if data.test_path is not None and data.test_path in {
            data.train_path,
            data.val_path,
        }:
            raise ValueError("test_path must differ from train_path and val_path")
        if data.require_metadata:
            required = {
                "dataset_id": data.dataset_id,
                "train_dates_path": data.train_dates_path,
                "val_dates_path": data.val_dates_path,
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise ValueError(f"metadata validation requires fields: {', '.join(missing)}")
            if data.test_path is not None and data.test_dates_path is None:
                raise ValueError("test_dates_path is required with test_path")

        objective = self.objective
        if objective.name not in {"f0_final", "f0_all", "f1", "f1_mv"}:
            raise ValueError(f"unsupported objective name={objective.name!r}")
        if objective.cumulative_scale_method not in {"mad", "std"}:
            raise ValueError("unsupported cumulative_scale_method")
        if objective.auxiliary_scale_method not in {"mad", "std"}:
            raise ValueError("unsupported auxiliary_scale_method")
        if (
            objective.cumulative_horizons
            and tuple(sorted(set(objective.cumulative_horizons))) != objective.cumulative_horizons
        ):
            raise ValueError("cumulative_horizons must be unique and sorted")
        if (
            objective.cumulative_horizons
            and objective.cumulative_horizons[-1] > data.horizon_length
        ):
            raise ValueError("cumulative_horizons cannot exceed horizon_length")
        weights = (
            objective.return_pinball_weight,
            objective.cumulative_huber_weight,
            objective.auxiliary_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("objective weights must be finite and non-negative")
        if objective.return_pinball_weight <= 0:
            raise ValueError("return_pinball_weight must be positive")
        if objective.cumulative_huber_weight > 0 and not objective.cumulative_horizons:
            raise ValueError("positive cumulative_huber_weight requires cumulative_horizons")
        if (
            not math.isfinite(objective.cumulative_huber_delta)
            or objective.cumulative_huber_delta <= 0
        ):
            raise ValueError("cumulative_huber_delta must be positive")
        if len(set(objective.auxiliary_features)) != len(objective.auxiliary_features):
            raise ValueError("auxiliary_features contains duplicates")
        missing_auxiliary = set(objective.auxiliary_features) - set(data.past_only_features)
        if missing_auxiliary:
            raise ValueError(
                "auxiliary_features must be selected past-only features: "
                f"{sorted(missing_auxiliary)}"
            )
        if objective.name in {"f0_final", "f0_all"} and (
            objective.cumulative_huber_weight != 0
            or objective.cumulative_horizons
            or objective.auxiliary_weight != 0
            or objective.auxiliary_features
        ):
            raise ValueError("F0-final/F0-all must use return Pinball only")
        if objective.name == "f1" and (
            objective.cumulative_huber_weight <= 0
            or objective.auxiliary_weight != 0
            or objective.auxiliary_features
        ):
            raise ValueError("F1 requires final cumulative Huber and no auxiliary loss")
        if objective.name == "f1_mv" and (
            objective.cumulative_huber_weight <= 0
            or objective.auxiliary_weight <= 0
            or not objective.auxiliary_features
        ):
            raise ValueError("F1-MV requires cumulative Huber plus selected auxiliary features")
        if objective.uses_dense_forward:
            if data.horizon_length != 64:
                raise ValueError("dense routes require the checkpoint output horizon of 64")
            if not self.model.disable_linear_detrending:
                raise ValueError("dense routes require disable_linear_detrending=true")
        if self.adapter.last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        if self.adapter.type == "lora" and self.adapter.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.adapter.type == "lora" and not any(
            (
                self.adapter.lora_sequence_attention,
                self.adapter.lora_variate_attention,
                self.adapter.lora_feedforward,
            )
        ):
            raise ValueError("LoRA requires at least one injection target")
        if self.adapter.alpha <= 0 or not 0 <= self.adapter.dropout < 1:
            raise ValueError("invalid LoRA alpha/dropout")

        learning_rates = (
            self.optimizer.adapter_learning_rate,
            self.optimizer.head_learning_rate,
            self.optimizer.pretrained_learning_rate,
        )
        if any(value <= 0 for value in learning_rates):
            raise ValueError("all learning rates must be positive")
        if self.optimizer.weight_decay < 0 or self.optimizer.eps <= 0:
            raise ValueError("invalid optimizer weight_decay/eps")
        if not 0 < self.optimizer.beta1 < 1 or not 0 < self.optimizer.beta2 < 1:
            raise ValueError("optimizer betas must be in (0, 1)")
        if not 0 <= self.scheduler.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.scheduler.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")

        trainer = self.trainer
        if (
            min(
                trainer.epochs,
                trainer.batch_size,
                trainer.gradient_accumulation_steps,
                trainer.log_every_steps,
            )
            <= 0
        ):
            raise ValueError("trainer counts must be positive")
        if trainer.max_grad_norm <= 0 or trainer.num_workers < 0:
            raise ValueError("invalid max_grad_norm/num_workers")
        if trainer.step_eval_interval is not None and trainer.step_eval_interval <= 0:
            raise ValueError("step_eval_interval must be positive or null")
        if trainer.early_stopping_patience is not None and trainer.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or null")
        if (
            not trainer.checkpoint_horizons
            or tuple(sorted(set(trainer.checkpoint_horizons))) != trainer.checkpoint_horizons
        ):
            raise ValueError("checkpoint_horizons must be non-empty, unique, and sorted")
        if trainer.checkpoint_horizons[-1] > data.horizon_length:
            raise ValueError("checkpoint_horizons cannot exceed horizon_length")

        evaluation = self.evaluation
        if (
            not evaluation.report_horizons
            or tuple(sorted(set(evaluation.report_horizons))) != evaluation.report_horizons
        ):
            raise ValueError("report_horizons must be non-empty, unique, and sorted")
        if evaluation.report_horizons[-1] > data.horizon_length:
            raise ValueError("report_horizons cannot exceed horizon_length")
        if not 1 <= evaluation.trading_horizon <= data.horizon_length:
            raise ValueError("trading_horizon must be within the forecast horizon")
        if evaluation.cost_per_turnover < 0:
            raise ValueError("cost_per_turnover must be non-negative")
        if trainer.checkpoint_metric in {
            "ic",
            "rank_ic",
            "mean_daily_rank_ic",
            "net_utility",
        } and not set(trainer.checkpoint_horizons).issubset(evaluation.report_horizons):
            raise ValueError("business checkpoint_horizons must be in evaluation.report_horizons")
        if trainer.checkpoint_metric == "net_utility" and trainer.checkpoint_horizons != (
            evaluation.trading_horizon,
        ):
            raise ValueError("net_utility requires only evaluation.trading_horizon")
