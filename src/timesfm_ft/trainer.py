"""Minimal, reproducible trainer for TimesFM 3 weighted-mid experiments."""

from __future__ import annotations

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
from timesfm_ft.config import ExperimentConfig
from timesfm_ft.data import NpzWindowDataset, WindowBatch
from timesfm_ft.losses import ForecastLoss
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
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def _read_expected_dates(path: str | None) -> set[int] | None:
    if path is None:
        return None
    date_path = Path(path)
    if not date_path.exists():
        raise FileNotFoundError(date_path)
    values = {
        int(line.strip())
        for line in date_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not values:
        raise ValueError(f"no dates in {date_path}")
    return values


def _run_epoch(
    model: TimesFM3Adapter,
    loader: DataLoader[WindowBatch],
    loss_fn: ForecastLoss,
    *,
    device: torch.device,
    horizon: int,
    sampling_interval_seconds: float,
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
    component_sums = {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in ("pinball", "huber", "crossing")
    }
    component_counts = {"pinball": 0, "huber": 0, "crossing": 0}
    sample_count = 0
    gradient_norm_total = 0.0
    optimizer_updates = 0
    latest_gradient_norm: float | None = None
    started_at = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    metric_accumulator = (
        None
        if training
        else ForecastMetricsAccumulator(
            horizon=horizon,
            quantiles=model.quantiles,
            tick_size=loss_fn.tick_size,
            sampling_interval_seconds=sampling_interval_seconds,
            target_mode=loss_fn.target_mode,
        )
    )

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for step, raw_batch in enumerate(loader):
            latest_gradient_norm = None
            valid_points = int((~raw_batch["future_mask"]).sum().item())
            batch = _move_batch(raw_batch, device)
            batch_size = batch["context_values"].shape[0]
            predictions = model(
                batch["context_values"],
                horizon=horizon,
                context_mask=batch["context_mask"],
            )
            losses = loss_fn(
                predictions.float(),
                batch["future_values"].float(),
                current_price=batch["context_values"][:, 0, -1].float(),
                target_mask=batch["future_mask"],
            )
            if not torch.isfinite(losses.total).item():
                raise FloatingPointError(
                    f"non-finite {split} loss at epoch={epoch} step={step + 1}"
                )
            if metric_accumulator is not None:
                metric_accumulator.update(
                    predictions,
                    batch["future_values"],
                    batch["context_values"][:, 0, -1],
                    batch["future_mask"],
                )

            if training:
                group_start = (
                    step // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps,
                    len(loader) - group_start,
                )
                (losses.total / group_size).backward()
                should_step = (
                    (step + 1) % gradient_accumulation_steps == 0
                    or step + 1 == len(loader)
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
                            f"non-finite gradient norm at epoch={epoch} "
                            f"step={step + 1}; bad_gradients={bad_gradients[:10]}"
                        )
                    gradient_norm_total += float(gradient_norm.detach())
                    latest_gradient_norm = float(gradient_norm.detach())
                    optimizer_updates += 1
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            counts = {
                "pinball": valid_points * loss_fn.pinball_quantile_count,
                "huber": valid_points,
                "crossing": valid_points * (len(model.quantiles) - 1),
            }
            values = {
                "pinball": losses.pinball.detach(),
                "huber": losses.median_huber.detach(),
                "crossing": losses.crossing.detach(),
            }
            for key in component_sums:
                component_sums[key].add_(values[key].double() * counts[key])
                component_counts[key] += counts[key]
            sample_count += batch_size

            if training and (
                (step + 1) % log_every_steps == 0 or step + 1 == len(loader)
            ):
                running = {
                    key: float(component_sums[key].item())
                    / max(component_counts[key], 1)
                    for key in component_sums
                }
                running["loss"] = (
                    loss_fn.pinball_weight * running["pinball"]
                    + loss_fn.median_huber_weight * running["huber"]
                    + loss_fn.crossing_weight * running["crossing"]
                )
                lr_text = " ".join(
                    f"lr_{name}={value:.3e}"
                    for name, value in _learning_rates(optimizer).items()
                )
                LOGGER.info(
                    "%s epoch=%d step=%d/%d batch_loss=%.6f "
                    "running_loss=%.6f pinball=%.6f huber=%.6f crossing=%.6f "
                    "optimizer_updates=%d gradient_norm=%s %s",
                    split,
                    epoch,
                    step + 1,
                    len(loader),
                    float(losses.total.detach()),
                    running["loss"],
                    running["pinball"],
                    running["huber"],
                    running["crossing"],
                    optimizer_updates,
                    (
                        f"{latest_gradient_norm:.6f}"
                        if latest_gradient_norm is not None
                        else "pending"
                    ),
                    lr_text,
                )

    elapsed_seconds = time.perf_counter() - started_at
    metrics = {
        key: float(component_sums[key].item()) / max(component_counts[key], 1)
        for key in component_sums
    }
    metrics["loss"] = (
        loss_fn.pinball_weight * metrics["pinball"]
        + loss_fn.median_huber_weight * metrics["huber"]
        + loss_fn.crossing_weight * metrics["crossing"]
    )
    if metric_accumulator is not None:
        point_metrics, _ = metric_accumulator.results()
        metrics.update(point_metrics)
    metrics["elapsed_seconds"] = elapsed_seconds
    metrics["samples_per_second"] = sample_count / max(elapsed_seconds, 1e-9)
    if training:
        metrics["mean_gradient_norm"] = gradient_norm_total / max(optimizer_updates, 1)
    return metrics


def _write_history(path: Path, history: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )
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
        "format_version": 1,
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
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError(f"unsupported training checkpoint: {path}")
    saved_config = state.get("config", {})
    current_config = config.to_dict()
    for section in ("data", "model", "adapter", "objective", "optimizer", "scheduler"):
        if saved_config.get(section) != current_config.get(section):
            raise ValueError(f"resume config mismatch in section {section}")
    saved_trainer = dict(saved_config.get("trainer", {}))
    current_trainer = dict(current_config.get("trainer", {}))
    for non_semantic_key in ("resume_from", "log_every_steps"):
        saved_trainer.pop(non_semantic_key, None)
        current_trainer.pop(non_semantic_key, None)
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


def train_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    if config.data.eval_only:
        raise ValueError("eval-only data config cannot be used for training")
    set_seed(config.trainer.seed, deterministic=config.trainer.deterministic)
    device = resolve_device(config.trainer.device)
    train_generator = torch.Generator()
    train_generator.manual_seed(config.trainer.seed)

    train_data = NpzWindowDataset(
        config.data.train_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=config.data.max_variates,
        sampling_interval_seconds=config.data.sampling_interval_seconds,
        expected_stride=config.data.stride,
        expected_product=config.data.product,
        expected_split="train",
        expected_target_mode=config.data.target_mode,
        expected_tick_size=config.objective.tick_size,
        expected_dates=_read_expected_dates(config.data.train_dates_path),
        expected_dates_path=config.data.train_dates_path,
        require_metadata=config.data.require_metadata,
    )
    val_data = NpzWindowDataset(
        config.data.val_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=config.data.max_variates,
        sampling_interval_seconds=config.data.sampling_interval_seconds,
        expected_stride=config.data.stride,
        expected_product=config.data.product,
        expected_split="val",
        expected_target_mode=config.data.target_mode,
        expected_tick_size=config.objective.tick_size,
        expected_dates=_read_expected_dates(config.data.val_dates_path),
        expected_dates_path=config.data.val_dates_path,
        require_metadata=config.data.require_metadata,
    )
    if train_data.num_variates != val_data.num_variates:
        raise ValueError("train and validation variate counts differ")
    if train_data.dates is not None and val_data.dates is not None:
        overlap = set(np.unique(train_data.dates)) & set(np.unique(val_data.dates))
        if overlap:
            raise ValueError(f"train/validation date overlap: {sorted(overlap)[:5]}")

    train_loader = DataLoader(
        train_data,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
        generator=train_generator,
        worker_init_fn=_seed_worker if config.trainer.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker if config.trainer.num_workers > 0 else None,
    )

    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
    )

    loss_fn = ForecastLoss(
        model.quantiles,
        tick_size=config.objective.tick_size,
        target_mode=config.data.target_mode,
        pinball_weight=config.objective.pinball_weight,
        include_median_in_pinball=config.objective.include_median_in_pinball,
        median_huber_weight=config.objective.median_huber_weight,
        crossing_weight=config.objective.crossing_weight,
        huber_delta_ticks=config.objective.huber_delta_ticks,
    ).to(device)
    parameter_groups = model.optimizer_parameter_groups(config.optimizer)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.trainer.gradient_accumulation_steps
    )
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
        "train_samples=%d val_samples=%d variates=%d context=%d horizon=%d "
        "micro_batch=%d effective_batch=%d deterministic=%s "
        "trainable=%d total=%d",
        config.model.checkpoint,
        config.adapter.type,
        device,
        config.trainer.dtype,
        len(train_data),
        len(val_data),
        train_data.num_variates,
        config.data.context_length,
        config.data.horizon_length,
        config.trainer.batch_size,
        config.trainer.batch_size * config.trainer.gradient_accumulation_steps,
        config.trainer.deterministic,
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

    data_metadata = {
        "train": train_data.metadata,
        "val": val_data.metadata,
    }
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
            generator=train_generator,
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
        raise ValueError(
            f"resume epoch {start_epoch} exceeds configured epochs "
            f"{config.trainer.epochs}"
        )
    for epoch in range(start_epoch, config.trainer.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device=device,
            horizon=config.data.horizon_length,
            sampling_interval_seconds=config.data.sampling_interval_seconds,
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
            sampling_interval_seconds=config.data.sampling_interval_seconds,
            optimizer=None,
            scheduler=None,
            gradient_accumulation_steps=1,
            max_grad_norm=config.trainer.max_grad_norm,
            epoch=epoch,
            split="val",
            log_every_steps=config.trainer.log_every_steps,
        )
        learning_rates = _learning_rates(optimizer)
        record = {
            "epoch": epoch,
            "learning_rates": learning_rates,
            "train": train_metrics,
            "val": val_metrics,
        }
        selected_value = val_metrics.get(config.trainer.checkpoint_metric)
        if selected_value is None or not math.isfinite(float(selected_value)):
            raise FloatingPointError(
                f"validation checkpoint metric "
                f"{config.trainer.checkpoint_metric} is not finite"
            )
        selected_metric = float(selected_value)
        improved = selected_metric < best_metric
        if improved:
            best_metric = selected_metric
            stale_epochs = 0
        else:
            stale_epochs += 1
        record["checkpoint"] = {
            "metric": config.trainer.checkpoint_metric,
            "value": selected_metric,
            "best": best_metric,
            "improved": improved,
            "stale_epochs": stale_epochs,
        }
        history.append(record)
        _write_history(output_dir / "history.jsonl", history)
        LOGGER.info(
            "epoch_end epoch=%d train_loss=%.6f val_loss=%.6f "
            "val_rmse_ticks=%s val_oos_r2=%s "
            "train_pinball=%.6f val_pinball=%.6f "
            "train_huber=%.6f val_huber=%.6f "
            "train_crossing=%.6f val_crossing=%.6f "
            "train_samples_per_second=%.2f val_samples_per_second=%.2f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            (
                f"{val_metrics['rmse_ticks']:.6f}"
                if val_metrics["rmse_ticks"] is not None
                else "null"
            ),
            (
                f"{val_metrics['oos_r2_vs_persistence']:.6f}"
                if val_metrics["oos_r2_vs_persistence"] is not None
                else "null"
            ),
            train_metrics["pinball"],
            val_metrics["pinball"],
            train_metrics["huber"],
            val_metrics["huber"],
            train_metrics["crossing"],
            val_metrics["crossing"],
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
            generator=train_generator,
            data_metadata=data_metadata,
        )
        model.save_adapter(
            output_dir / "last",
            metadata={
                "num_variates": train_data.num_variates,
                "context_length": config.data.context_length,
                "horizon_length": config.data.horizon_length,
                "checkpoint_metric": config.trainer.checkpoint_metric,
                "metric": selected_metric,
                "epoch": epoch,
                "data_metadata": data_metadata,
            },
        )
        _atomic_torch_save(state, output_dir / "last" / "training_state.pt")
        if improved:
            model.save_adapter(
                output_dir / "best",
                metadata={
                    "num_variates": train_data.num_variates,
                    "context_length": config.data.context_length,
                    "horizon_length": config.data.horizon_length,
                    "checkpoint_metric": config.trainer.checkpoint_metric,
                    "best_metric": best_metric,
                    "data_metadata": data_metadata,
                },
            )
            _atomic_torch_save(
                state,
                output_dir / "best" / "training_state.pt",
            )
            LOGGER.info(
                "checkpoint_saved epoch=%d %s=%.6f path=%s",
                epoch,
                config.trainer.checkpoint_metric,
                best_metric,
                output_dir / "best" / "adapter.pt",
            )
        patience = config.trainer.early_stopping_patience
        if patience is not None and stale_epochs >= patience:
            LOGGER.info(
                "early_stop epoch=%d patience=%d best_%s=%.6f",
                epoch,
                patience,
                config.trainer.checkpoint_metric,
                best_metric,
            )
            break

    LOGGER.info(
        "run_end best_%s=%.6f artifacts=%s",
        config.trainer.checkpoint_metric,
        best_metric,
        output_dir,
    )
    return output_dir
