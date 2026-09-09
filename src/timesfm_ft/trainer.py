"""Deterministic trainer for dynamic-context 1-minute TimesFM experiments."""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import EvaluationConfig, ExperimentConfig
from timesfm_ft.data import (
    ContextBucketBatchSampler,
    IntradayWindowDataset,
    WindowBatch,
    collate_intraday_windows,
)
from timesfm_ft.losses import PinballLoss
from timesfm_ft.metrics import ForecastMetricsAccumulator

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR:
    warmup_steps = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _move_batch(batch: WindowBatch, device: torch.device) -> WindowBatch:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()
    }


def _make_loader(
    dataset: IntradayWindowDataset,
    *,
    batch_size: int,
    patch_length: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None,
) -> DataLoader[WindowBatch]:
    sampler = ContextBucketBatchSampler(
        dataset,
        batch_size=batch_size,
        patch_length=patch_length,
        shuffle=shuffle,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=functools.partial(collate_intraday_windows, patch_length=patch_length),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )


def _run_epoch(
    model: TimesFM3Adapter,
    loader: DataLoader[WindowBatch],
    loss_fn: PinballLoss,
    *,
    device: torch.device,
    horizon: int,
    evaluation: EvaluationConfig,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    split: str,
    log_every_steps: int,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    pinball_sum = torch.zeros((), device=device, dtype=torch.float64)
    pinball_count = 0
    sample_count = 0
    gradient_norm_total = 0.0
    optimizer_updates = 0
    started_at = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    accumulator = (
        None
        if training
        else ForecastMetricsAccumulator(
            horizon=horizon,
            quantiles=model.quantiles,
            report_horizons=evaluation.report_horizons,
            trading_horizon=evaluation.trading_horizon,
            cost_per_turnover=evaluation.cost_per_turnover,
        )
    )

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for step, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, device)
            predictions = model(
                batch["context_values"],
                horizon=horizon,
                context_mask=batch["context_mask"],
                past_future_values=batch["past_future_values"],
                past_future_mask=batch["past_future_mask"],
            )
            losses = loss_fn(
                predictions,
                batch["future_values"],
                target_mask=batch["future_mask"],
            )
            if not torch.isfinite(losses.total).item():
                raise FloatingPointError(
                    f"non-finite {split} loss at epoch={epoch} step={step + 1}"
                )
            if accumulator is not None:
                accumulator.update(
                    predictions,
                    batch["future_values"],
                    batch["future_mask"],
                    last_returns=batch["last_returns"],
                    context_lengths=batch["context_lengths"],
                    dates=batch["dates"],
                    timestamps=batch["timestamps"],
                    minute_indices=batch["minute_indices"],
                    context_volatility=batch["context_volatility"],
                )

            latest_gradient_norm: float | None = None
            if training:
                group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps,
                    len(loader) - group_start,
                )
                (losses.total / group_size).backward()
                should_step = (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(
                    loader
                )
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        max_grad_norm,
                    )
                    if not torch.isfinite(gradient_norm):
                        bad_gradients = [
                            name
                            for name, parameter in model.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ]
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"non-finite gradients at epoch={epoch} step={step + 1}; "
                            f"bad={bad_gradients[:10]}"
                        )
                    latest_gradient_norm = float(gradient_norm.detach())
                    gradient_norm_total += latest_gradient_norm
                    optimizer_updates += 1
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            valid_points = int((~batch["future_mask"]).sum().item())
            count = valid_points * loss_fn.quantile_count
            pinball_sum.add_(losses.pinball.detach().double() * count)
            pinball_count += count
            sample_count += len(batch["future_values"])
            if training and ((step + 1) % log_every_steps == 0 or step + 1 == len(loader)):
                lr_text = " ".join(
                    f"lr_{name}={value:.3e}" for name, value in _learning_rates(optimizer).items()
                )
                LOGGER.info(
                    "%s epoch=%d step=%d/%d batch_pinball=%.6f "
                    "running_pinball=%.6f context_width=%d gradient_norm=%s %s",
                    split,
                    epoch,
                    step + 1,
                    len(loader),
                    float(losses.pinball.detach()),
                    float(pinball_sum.item()) / max(pinball_count, 1),
                    batch["context_values"].shape[-1],
                    (
                        f"{latest_gradient_norm:.6f}"
                        if latest_gradient_norm is not None
                        else "pending"
                    ),
                    lr_text,
                )

    elapsed = time.perf_counter() - started_at
    metrics: dict[str, Any] = {
        "loss": float(pinball_sum.item()) / max(pinball_count, 1),
        "mean_pinball": float(pinball_sum.item()) / max(pinball_count, 1),
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / max(elapsed, 1e-9),
    }
    if accumulator is not None:
        summary, _, cumulative, slices = accumulator.results()
        metrics.update(summary)
        metrics["cumulative_horizons"] = cumulative
        metrics["slices"] = slices
    if training:
        metrics["mean_gradient_norm"] = gradient_norm_total / max(optimizer_updates, 1)
    return metrics


def _write_history(path: Path, history: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _training_state(
    *,
    model: TimesFM3Adapter,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    best_metric: float,
    stale_epochs: int,
    history: list[dict[str, Any]],
    config: ExperimentConfig,
    generator: torch.Generator,
    data_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "adapter": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "stale_epochs": stale_epochs,
        "history": history,
        "config": config.to_dict(),
        "rng": _rng_state(generator),
        "data_metadata": data_metadata,
    }


def _load_training_state(
    path: Path,
    *,
    model: TimesFM3Adapter,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: ExperimentConfig,
    generator: torch.Generator,
    data_metadata: dict[str, Any],
) -> tuple[int, float, int, list[dict[str, Any]]]:
    if path.is_dir():
        path = path / "training_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != 2:
        raise ValueError(f"unsupported training checkpoint: {path}")
    saved_config = state.get("config", {})
    current_config = config.to_dict()
    for section in (
        "data",
        "model",
        "adapter",
        "objective",
        "optimizer",
        "scheduler",
        "evaluation",
    ):
        if saved_config.get(section) != current_config.get(section):
            raise ValueError(f"resume config mismatch in section {section}")
    saved_trainer = dict(saved_config.get("trainer", {}))
    current_trainer = dict(current_config.get("trainer", {}))
    for key in ("resume_from", "log_every_steps"):
        saved_trainer.pop(key, None)
        current_trainer.pop(key, None)
    if saved_trainer != current_trainer:
        raise ValueError("resume config mismatch in section trainer")
    if state.get("data_metadata") != data_metadata:
        raise ValueError("resume data metadata mismatch")
    model.load_trainable_state_dict(state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng_state(state["rng"], generator)
    return (
        int(state["epoch"]) + 1,
        float(state["best_metric"]),
        int(state["stale_epochs"]),
        list(state["history"]),
    )


def _dataset(
    config: ExperimentConfig,
    *,
    path: str,
    split: str,
    dates_path: str | None,
) -> IntradayWindowDataset:
    return IntradayWindowDataset(
        path,
        context_min=config.data.context_min,
        context_max=config.data.context_max,
        horizon_length=config.data.horizon_length,
        stride=config.data.stride,
        past_only_features=config.data.past_only_features,
        past_future_features=config.data.past_future_features,
        max_variates=config.data.max_variates,
        expected_split=split,
        expected_dataset_id=config.data.dataset_id,
        expected_product=config.data.product,
        expected_target_name=config.data.target_name,
        expected_target_unit=config.data.target_unit,
        expected_target_price_source=config.data.target_price_source,
        expected_target_return_type=config.data.target_return_type,
        expected_target_timestamp_semantics=config.data.target_timestamp_semantics,
        expected_target_availability_lag_minutes=(config.data.target_availability_lag_minutes),
        expected_frequency_minutes=config.data.frequency_minutes,
        expected_session_minutes=config.data.session_minutes,
        expected_dates_path=dates_path,
        require_metadata=config.data.require_metadata,
    )


def train_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    set_seed(config.trainer.seed, deterministic=config.trainer.deterministic)
    device = resolve_device(config.trainer.device)
    generator = torch.Generator()
    generator.manual_seed(config.trainer.seed)
    train_data = _dataset(
        config,
        path=config.data.train_path,
        split="train",
        dates_path=config.data.train_dates_path,
    )
    val_data = _dataset(
        config,
        path=config.data.val_path,
        split="val",
        dates_path=config.data.val_dates_path,
    )
    if train_data.num_variates != val_data.num_variates:
        raise ValueError("train and validation variate counts differ")
    overlap = set(int(value) for value in train_data.dates) & set(
        int(value) for value in val_data.dates
    )
    if overlap:
        raise ValueError(f"train/validation date overlap: {sorted(overlap)[:5]}")

    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
    )
    patch_length = int(model.backbone.input_patch_len)
    train_loader = _make_loader(
        train_data,
        batch_size=config.trainer.batch_size,
        patch_length=patch_length,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = _make_loader(
        val_data,
        batch_size=config.trainer.batch_size,
        patch_length=patch_length,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
        generator=None,
    )

    loss_fn = PinballLoss(model.quantiles).to(device)
    parameter_groups = model.optimizer_parameter_groups(config.optimizer)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.trainer.gradient_accumulation_steps)
    scheduler = make_scheduler(
        optimizer,
        total_steps=max(updates_per_epoch * config.trainer.epochs, 1),
        warmup_ratio=config.scheduler.warmup_ratio,
        min_lr_ratio=config.scheduler.min_lr_ratio,
    )

    output_dir = Path(config.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    summary = model.parameter_summary
    LOGGER.info(
        "run_start checkpoint=%s adapter=%s device=%s dtype=%s "
        "train_samples=%d val_samples=%d variates=%d context=%d..%d horizon=%d "
        "batch=%d effective_batch=%d trainable=%d total=%d",
        config.model.checkpoint,
        config.adapter.type,
        device,
        config.trainer.dtype,
        len(train_data),
        len(val_data),
        train_data.num_variates,
        config.data.context_min,
        config.data.context_max,
        config.data.horizon_length,
        config.trainer.batch_size,
        config.trainer.batch_size * config.trainer.gradient_accumulation_steps,
        summary["trainable"],
        summary["total"],
    )
    for group in optimizer.param_groups:
        LOGGER.info(
            "optimizer_group name=%s parameters=%d lr=%.3e weight_decay=%.3e",
            group["group_name"],
            sum(parameter.numel() for parameter in group["params"]),
            group["lr"],
            group["weight_decay"],
        )

    data_metadata = {"train": train_data.metadata, "val": val_data.metadata}
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    stale_epochs = 0
    start_epoch = 1
    if config.trainer.resume_from is not None:
        start_epoch, best_metric, stale_epochs, history = _load_training_state(
            Path(config.trainer.resume_from),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            generator=generator,
            data_metadata=data_metadata,
        )
        LOGGER.info(
            "resumed checkpoint=%s start_epoch=%d best_%s=%.6f",
            config.trainer.resume_from,
            start_epoch,
            config.trainer.checkpoint_metric,
            best_metric,
        )
    if start_epoch > config.trainer.epochs:
        raise ValueError("resume epoch exceeds configured epochs")

    for epoch in range(start_epoch, config.trainer.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device=device,
            horizon=config.data.horizon_length,
            evaluation=config.evaluation,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
            max_grad_norm=config.trainer.max_grad_norm,
            epoch=epoch,
            split="train",
            log_every_steps=config.trainer.log_every_steps,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            loss_fn,
            device=device,
            horizon=config.data.horizon_length,
            evaluation=config.evaluation,
            optimizer=None,
            scheduler=None,
            gradient_accumulation_steps=1,
            max_grad_norm=config.trainer.max_grad_norm,
            epoch=epoch,
            split="val",
            log_every_steps=config.trainer.log_every_steps,
        )
        selected = val_metrics.get(config.trainer.checkpoint_metric)
        if selected is None or not math.isfinite(float(selected)):
            raise FloatingPointError("validation checkpoint metric is not finite")
        selected_metric = float(selected)
        improved = selected_metric < best_metric
        if improved:
            best_metric = selected_metric
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "learning_rates": _learning_rates(optimizer),
            "train": train_metrics,
            "val": val_metrics,
            "checkpoint": {
                "metric": config.trainer.checkpoint_metric,
                "value": selected_metric,
                "best": best_metric,
                "improved": improved,
                "stale_epochs": stale_epochs,
            },
        }
        history.append(record)
        _write_history(output_dir / "history.jsonl", history)
        LOGGER.info(
            "epoch_end epoch=%d train_pinball=%.6f val_pinball=%.6f "
            "val_rmse=%s val_ic60=%s train_sps=%.2f val_sps=%.2f",
            epoch,
            train_metrics["mean_pinball"],
            val_metrics["mean_pinball"],
            (f"{val_metrics['rmse']:.6f}" if val_metrics.get("rmse") is not None else "null"),
            next(
                (
                    f"{row['ic']:.6f}"
                    for row in val_metrics["cumulative_horizons"]
                    if row["horizon_minutes"] == 60 and row["ic"] is not None
                ),
                "null",
            ),
            train_metrics["samples_per_second"],
            val_metrics["samples_per_second"],
        )
        state = _training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            stale_epochs=stale_epochs,
            history=history,
            config=config,
            generator=generator,
            data_metadata=data_metadata,
        )
        checkpoint_metadata = {
            "num_variates": train_data.num_variates,
            "context_min": config.data.context_min,
            "context_max": config.data.context_max,
            "horizon_length": config.data.horizon_length,
            "past_only_features": config.data.past_only_features,
            "past_future_features": config.data.past_future_features,
            "checkpoint_metric": config.trainer.checkpoint_metric,
            "data_metadata": data_metadata,
        }
        model.save_adapter(
            output_dir / "last",
            metadata=checkpoint_metadata | {"metric": selected_metric, "epoch": epoch},
        )
        _atomic_torch_save(state, output_dir / "last" / "training_state.pt")
        if improved:
            model.save_adapter(
                output_dir / "best",
                metadata=checkpoint_metadata | {"best_metric": best_metric},
            )
            _atomic_torch_save(state, output_dir / "best" / "training_state.pt")
        patience = config.trainer.early_stopping_patience
        if patience is not None and stale_epochs >= patience:
            LOGGER.info("early_stop epoch=%d patience=%d", epoch, patience)
            break

    LOGGER.info(
        "run_end best_%s=%.6f artifacts=%s",
        config.trainer.checkpoint_metric,
        best_metric,
        output_dir,
    )
    return output_dir
