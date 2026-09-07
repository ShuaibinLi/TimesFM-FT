"""Minimal, reproducible trainer for TimesFM 3 weighted-mid experiments."""

from __future__ import annotations

import json
import logging
import math
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

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _run_epoch(
    model: TimesFM3Adapter,
    loader: DataLoader[WindowBatch],
    loss_fn: ForecastLoss,
    *,
    device: torch.device,
    horizon: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    use_bfloat16: bool,
    epoch: int,
    split: str,
    log_every_steps: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "pinball": 0.0, "huber": 0.0, "crossing": 0.0}
    sample_count = 0
    gradient_norm_total = 0.0
    optimizer_updates = 0
    started_at = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for step, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, device)
            batch_size = batch["context_values"].shape[0]
            autocast_enabled = use_bfloat16 and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                predictions = model(
                    batch["context_values"],
                    horizon=horizon,
                    context_mask=batch["context_mask"],
                )
                losses = loss_fn(
                    predictions,
                    batch["future_values"],
                    current_price=batch["context_values"][:, 0, -1],
                    target_mask=batch["future_mask"],
                )

            if training:
                (losses.total / gradient_accumulation_steps).backward()
                should_step = (
                    (step + 1) % gradient_accumulation_steps == 0
                    or step + 1 == len(loader)
                )
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        max_grad_norm,
                    )
                    gradient_norm_total += float(gradient_norm.detach())
                    optimizer_updates += 1
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            totals["loss"] += float(losses.total.detach()) * batch_size
            totals["pinball"] += float(losses.pinball.detach()) * batch_size
            totals["huber"] += float(losses.median_huber.detach()) * batch_size
            totals["crossing"] += float(losses.crossing.detach()) * batch_size
            sample_count += batch_size

            if training and (
                (step + 1) % log_every_steps == 0 or step + 1 == len(loader)
            ):
                running = {
                    key: value / max(sample_count, 1) for key, value in totals.items()
                }
                lr_text = " ".join(
                    f"lr_{name}={value:.3e}"
                    for name, value in _learning_rates(optimizer).items()
                )
                LOGGER.info(
                    "%s epoch=%d step=%d/%d loss=%.6f pinball=%.6f "
                    "huber=%.6f crossing=%.6f %s",
                    split,
                    epoch,
                    step + 1,
                    len(loader),
                    running["loss"],
                    running["pinball"],
                    running["huber"],
                    running["crossing"],
                    lr_text,
                )

    elapsed_seconds = time.perf_counter() - started_at
    metrics = {key: value / max(sample_count, 1) for key, value in totals.items()}
    metrics["elapsed_seconds"] = elapsed_seconds
    metrics["samples_per_second"] = sample_count / max(elapsed_seconds, 1e-9)
    if training:
        metrics["mean_gradient_norm"] = gradient_norm_total / max(optimizer_updates, 1)
    return metrics


def _write_history(path: Path, history: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def train_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    set_seed(config.trainer.seed)
    device = resolve_device(config.trainer.device)

    train_data = NpzWindowDataset(
        config.data.train_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=config.data.max_variates,
    )
    val_data = NpzWindowDataset(
        config.data.val_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=config.data.max_variates,
    )
    if train_data.num_variates != val_data.num_variates:
        raise ValueError("train and validation variate counts differ")

    train_loader = DataLoader(
        train_data,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
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
        pinball_weight=config.objective.pinball_weight,
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

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(1, config.trainer.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device=device,
            horizon=config.data.horizon_length,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
            max_grad_norm=config.trainer.max_grad_norm,
            use_bfloat16=config.trainer.dtype == "bfloat16",
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
            optimizer=None,
            scheduler=None,
            gradient_accumulation_steps=1,
            max_grad_norm=config.trainer.max_grad_norm,
            use_bfloat16=config.trainer.dtype == "bfloat16",
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
        history.append(record)
        _write_history(output_dir / "history.jsonl", history)
        LOGGER.info(
            "epoch_end epoch=%d train_loss=%.6f val_loss=%.6f "
            "train_pinball=%.6f val_pinball=%.6f "
            "train_huber=%.6f val_huber=%.6f "
            "train_crossing=%.6f val_crossing=%.6f "
            "train_samples_per_second=%.2f val_samples_per_second=%.2f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["pinball"],
            val_metrics["pinball"],
            train_metrics["huber"],
            val_metrics["huber"],
            train_metrics["crossing"],
            val_metrics["crossing"],
            train_metrics["samples_per_second"],
            val_metrics["samples_per_second"],
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save_adapter(
                output_dir / "best",
                metadata={
                    "num_variates": train_data.num_variates,
                    "context_length": config.data.context_length,
                    "horizon_length": config.data.horizon_length,
                    "best_val_loss": best_val,
                },
            )
            LOGGER.info(
                "checkpoint_saved epoch=%d val_loss=%.6f path=%s",
                epoch,
                best_val,
                output_dir / "best" / "adapter.pt",
            )

    LOGGER.info("run_end best_val_loss=%.6f artifacts=%s", best_val, output_dir)
    return output_dir
