"""Minimal, reproducible trainer for TimesFM 3 weighted-mid experiments."""

from __future__ import annotations

import json
import math
import random
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
) -> LambdaLR:
    warmup_steps = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, multiplier)


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
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "pinball": 0.0, "huber": 0.0, "crossing": 0.0}
    sample_count = 0
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
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            totals["loss"] += float(losses.total.detach()) * batch_size
            totals["pinball"] += float(losses.pinball.detach()) * batch_size
            totals["huber"] += float(losses.median_huber.detach()) * batch_size
            totals["crossing"] += float(losses.crossing.detach()) * batch_size
            sample_count += batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def _write_history(path: Path, history: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def train_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

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
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TimesFM3Adapter.from_pretrained(
        config.model,
        device=device,
        dtype=config.train.dtype,
    )

    loss_fn = ForecastLoss(
        model.quantiles,
        tick_size=config.loss.tick_size,
        pinball_weight=config.loss.pinball_weight,
        median_huber_weight=config.loss.median_huber_weight,
        crossing_weight=config.loss.crossing_weight,
        huber_delta_ticks=config.loss.huber_delta_ticks,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.train.gradient_accumulation_steps
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=max(updates_per_epoch * config.train.epochs, 1),
        warmup_ratio=config.train.warmup_ratio,
    )

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    summary = model.parameter_summary
    print(
        f"device={device} variates={train_data.num_variates} "
        f"trainable={summary['trainable']:,}/{summary['total']:,}"
    )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device=device,
            horizon=config.data.horizon_length,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            max_grad_norm=config.train.max_grad_norm,
            use_bfloat16=config.train.dtype == "bfloat16",
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
            max_grad_norm=config.train.max_grad_norm,
            use_bfloat16=config.train.dtype == "bfloat16",
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        _write_history(output_dir / "history.jsonl", history)
        print(
            f"epoch={epoch} train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f}"
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

    return output_dir
