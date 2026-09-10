"""Deterministic trainer for dynamic-context 1-minute TimesFM experiments."""

from __future__ import annotations

import functools
import hashlib
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
from timesfm_ft.config import EvaluationConfig, ExperimentConfig, ObjectiveConfig
from timesfm_ft.data import (
    ContextBucketBatchSampler,
    IntradayWindowDataset,
    WindowBatch,
    collate_intraday_windows,
)
from timesfm_ft.dense import (
    build_dense_training_batch,
    gather_final_anchor,
)
from timesfm_ft.losses import (
    BusinessForecastLoss,
    LossScaleState,
    ScaleEstimate,
)
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


def _robust_scale(values: np.ndarray, method: str) -> ScaleEstimate:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot fit a loss scale without valid training values")
    if method == "mad":
        center = np.median(values)
        raw_scale = 1.4826 * float(np.median(np.abs(values - center)))
    elif method == "std":
        raw_scale = float(np.std(values))
    else:
        raise ValueError(f"unsupported scale method={method!r}")
    scale = raw_scale
    fallback_name: str | None = None
    if not math.isfinite(scale) or scale <= 1e-8:
        std_fallback = float(np.std(values))
        if math.isfinite(std_fallback) and std_fallback > 1e-8:
            scale = std_fallback
            fallback_name = "std"
        else:
            scale = 1.0
            fallback_name = "unit"
    return ScaleEstimate(
        value=scale,
        estimator=method,
        valid_count=int(values.size),
        raw_value=raw_scale,
        fallback=fallback_name,
    )


def fit_loss_scales(
    dataset: IntradayWindowDataset,
    objective: ObjectiveConfig,
    *,
    input_patch_length: int | None = None,
    output_patch_length: int | None = None,
    context_min: int | None = None,
    batch_size: int = 128,
) -> LossScaleState:
    """Fits route-specific normalization scales from the train split only."""

    if dataset.metadata.get("split") != "train":
        raise ValueError("loss scales may only be fitted from split=train")
    cumulative_values: dict[int, list[float]] = {
        horizon: [] for horizon in objective.cumulative_horizons
    }
    for day_value, anchor_value in zip(
        dataset._day_indices,
        dataset._anchor_indices,
        strict=True,
    ):
        day = int(day_value)
        anchor = int(anchor_value)
        start = anchor + 1
        for horizon in objective.cumulative_horizons:
            target = dataset.target_values[day, start : start + horizon]
            mask = dataset.target_mask[day, start : start + horizon]
            if not mask.any():
                cumulative_values[horizon].append(float(np.sum(target, dtype=np.float64)))
    cumulative_scales = {
        horizon: _robust_scale(
            np.asarray(values),
            objective.cumulative_scale_method,
        )
        for horizon, values in cumulative_values.items()
    }

    auxiliary_values: dict[str, list[np.ndarray]] = {
        feature: [] for feature in objective.auxiliary_features
    }
    if objective.auxiliary_features:
        if (
            not objective.uses_dense_forward
            or input_patch_length is None
            or output_patch_length is None
            or context_min is None
        ):
            raise ValueError("dense auxiliary scales require patch lengths and context_min")
        auxiliary_indices = [
            dataset.past_only_features.index(feature) for feature in objective.auxiliary_features
        ]
        scale_loader = _make_loader(
            dataset,
            batch_size=batch_size,
            patch_length=input_patch_length,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            generator=None,
        )
        for raw_batch in scale_loader:
            dense = build_dense_training_batch(
                raw_batch,
                context_min=context_min,
                input_patch_length=input_patch_length,
                output_patch_length=output_patch_length,
            )
            for feature, index in zip(
                objective.auxiliary_features,
                auxiliary_indices,
                strict=True,
            ):
                labels = dense.past_only_labels[:, index]
                valid = (
                    dense.eligible_anchor_mask[:, :, None] & ~dense.past_only_label_mask[:, index]
                )
                auxiliary_values[feature].append(labels[valid].numpy())
    auxiliary_scales = {
        feature: _robust_scale(
            np.concatenate(values) if values else np.asarray([]),
            objective.auxiliary_scale_method,
        )
        for feature, values in auxiliary_values.items()
    }
    manifest_payload = json.dumps(
        dataset.metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return LossScaleState(
        dataset_id=str(dataset.metadata["dataset_id"]),
        date_file_sha256=str(dataset.metadata["date_file_sha256"]),
        feature_schema_sha256=dataset.metadata.get("feature_schema_sha256"),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        sampling_contract={
            "training_route": objective.name,
            "context_min": dataset.context_min,
            "context_max": dataset.context_max,
            "horizon_length": dataset.horizon_length,
            "stride": dataset.stride,
            "input_patch_length": input_patch_length,
            "output_patch_length": output_patch_length,
        },
        cumulative_method=objective.cumulative_scale_method,
        auxiliary_method=objective.auxiliary_scale_method,
        cumulative=cumulative_scales,
        auxiliary=auxiliary_scales,
    )


def _checkpoint_value(
    metrics: dict[str, Any],
    *,
    metric: str,
    horizons: tuple[int, ...],
) -> tuple[float, str]:
    if metric in {"mean_pinball", "rmse"}:
        value = metrics.get(metric)
        mode = "min"
    elif metric in {"ic", "rank_ic", "mean_daily_rank_ic"}:
        rows = {
            int(item["horizon_minutes"]): item for item in metrics.get("cumulative_horizons", ())
        }
        values = [rows.get(horizon, {}).get(metric) for horizon in horizons]
        value = (
            float(np.mean(values)) if values and all(item is not None for item in values) else None
        )
        mode = "max"
    elif metric == "net_utility":
        value = metrics.get("trading_proxy", {}).get("net_mean")
        mode = "max"
    else:
        raise ValueError(f"unsupported checkpoint metric={metric!r}")
    if value is None or not math.isfinite(float(value)):
        raise FloatingPointError(f"validation checkpoint metric {metric}@{horizons} is not finite")
    return float(value), mode


def _run_epoch(
    model: TimesFM3Adapter,
    loader: DataLoader[WindowBatch],
    loss_fn: BusinessForecastLoss,
    *,
    device: torch.device,
    horizon: int,
    context_min: int,
    evaluation: EvaluationConfig,
    auxiliary_indices: torch.Tensor,
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
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in (
            "return_pinball",
            "cumulative_huber",
            "auxiliary_pinball",
        )
    }
    component_counts = {name: 0 for name in component_sums}
    sample_count = 0
    eligible_anchor_count = 0
    unique_dense_anchors: set[tuple[int, int]] = set()
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
            business_targets = batch["unknown_future_values"][:, 0]
            business_target_mask = batch["unknown_future_mask"][:, 0]
            model_kwargs = {
                "horizon": horizon,
                "context_mask": batch["context_mask"],
                "context_padding_mask": batch["context_padding_mask"],
                "past_future_values": batch["past_future_values"],
                "past_future_mask": batch["past_future_mask"],
            }
            auxiliary_predictions: torch.Tensor | None = None
            auxiliary_targets: torch.Tensor | None = None
            auxiliary_mask: torch.Tensor | None = None
            auxiliary_anchor_mask: torch.Tensor | None = None
            if loss_fn.objective.uses_dense_forward:
                dense_batch = build_dense_training_batch(
                    batch,
                    context_min=context_min,
                    input_patch_length=int(model.backbone.input_patch_len),
                    output_patch_length=int(model.backbone.output_patch_len),
                )
                dense_predictions = model.forward_dense(
                    dense_batch.values,
                    masks=dense_batch.masks,
                    patch_is_target=dense_batch.patch_is_target,
                    unknown_variates=batch["context_values"].shape[1],
                )
                predictions = dense_predictions.target
                targets = dense_batch.target_labels
                target_mask = dense_batch.target_label_mask
                anchor_mask = dense_batch.eligible_anchor_mask
                eligible_anchor_count += int(anchor_mask.sum().item())
                dense_dates = batch["dates"][:, None].expand_as(dense_batch.anchor_timestamps)
                for date, timestamp in zip(
                    dense_dates[anchor_mask].detach().cpu().tolist(),
                    dense_batch.anchor_timestamps[anchor_mask].detach().cpu().tolist(),
                    strict=True,
                ):
                    unique_dense_anchors.add((int(date), int(timestamp)))
                final_predictions = gather_final_anchor(
                    dense_predictions.target,
                    dense_batch.final_anchor_indices,
                )
                metric_predictions = final_predictions
                if auxiliary_indices.numel():
                    auxiliary_predictions = dense_predictions.past_only.index_select(
                        1, auxiliary_indices
                    )
                    auxiliary_targets = dense_batch.past_only_labels.index_select(
                        1, auxiliary_indices
                    )
                    auxiliary_mask = dense_batch.past_only_label_mask.index_select(
                        1, auxiliary_indices
                    )
                    auxiliary_anchor_mask = dense_batch.eligible_anchor_mask
            else:
                metric_predictions = model(
                    batch["context_values"],
                    **model_kwargs,
                )
                predictions = metric_predictions
                targets = business_targets
                target_mask = business_target_mask
                anchor_mask = None
                final_predictions = None
            losses = loss_fn(
                predictions,
                targets,
                target_mask=target_mask,
                anchor_mask=anchor_mask,
                final_predictions=final_predictions,
                final_targets=business_targets,
                final_target_mask=business_target_mask,
                auxiliary_predictions=auxiliary_predictions,
                auxiliary_targets=auxiliary_targets,
                auxiliary_mask=auxiliary_mask,
                auxiliary_anchor_mask=auxiliary_anchor_mask,
            )
            if not torch.isfinite(losses.total).item():
                raise FloatingPointError(
                    f"non-finite {split} loss at epoch={epoch} step={step + 1}"
                )
            if accumulator is not None:
                if loss_fn.objective.uses_dense_forward:
                    deployment_predictions = model.predict(
                        batch["context_values"],
                        **model_kwargs,
                    )
                    if not torch.allclose(
                        metric_predictions.float(),
                        deployment_predictions.float(),
                        rtol=1e-4,
                        atol=1e-5,
                    ):
                        difference = (
                            (metric_predictions.float() - deployment_predictions.float())
                            .abs()
                            .max()
                        )
                        raise FloatingPointError(
                            "dense final-token/deployment parity failed; "
                            f"max_abs_error={float(difference):.6g}"
                        )
                    metric_predictions = deployment_predictions
                accumulator.update(
                    metric_predictions,
                    business_targets,
                    business_target_mask,
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

            component_values = {
                "return_pinball": losses.return_pinball,
                "cumulative_huber": losses.cumulative_huber,
                "auxiliary_pinball": losses.auxiliary_pinball,
            }
            batch_counts = {
                "return_pinball": losses.return_count,
                "cumulative_huber": losses.cumulative_count,
                "auxiliary_pinball": losses.auxiliary_count,
            }
            for name, value in component_values.items():
                count = batch_counts[name]
                component_sums[name].add_(value.detach().double() * count)
                component_counts[name] += count
            sample_count += len(business_targets)
            if training and ((step + 1) % log_every_steps == 0 or step + 1 == len(loader)):
                running = {
                    name: float(component_sums[name].item()) / max(component_counts[name], 1)
                    for name in component_sums
                }
                running_total = (
                    loss_fn.objective.return_pinball_weight * running["return_pinball"]
                    + loss_fn.objective.cumulative_huber_weight * running["cumulative_huber"]
                    + loss_fn.objective.auxiliary_weight * running["auxiliary_pinball"]
                )
                lr_text = " ".join(
                    f"lr_{name}={value:.3e}" for name, value in _learning_rates(optimizer).items()
                )
                LOGGER.info(
                    "%s epoch=%d step=%d/%d batch_loss=%.6f "
                    "running_loss=%.6f return_pinball=%.6f cumulative_huber=%.6f "
                    "auxiliary_pinball=%.6f context_width=%d gradient_norm=%s %s",
                    split,
                    epoch,
                    step + 1,
                    len(loader),
                    float(losses.total.detach()),
                    running_total,
                    running["return_pinball"],
                    running["cumulative_huber"],
                    running["auxiliary_pinball"],
                    batch["context_values"].shape[-1],
                    (
                        f"{latest_gradient_norm:.6f}"
                        if latest_gradient_norm is not None
                        else "pending"
                    ),
                    lr_text,
                )

    elapsed = time.perf_counter() - started_at
    components = {
        name: float(component_sums[name].item()) / max(component_counts[name], 1)
        for name in component_sums
    }
    total_loss = (
        loss_fn.objective.return_pinball_weight * components["return_pinball"]
        + loss_fn.objective.cumulative_huber_weight * components["cumulative_huber"]
        + loss_fn.objective.auxiliary_weight * components["auxiliary_pinball"]
    )
    metrics: dict[str, Any] = {
        "loss": total_loss,
        "return_pinball": components["return_pinball"],
        "cumulative_huber": components["cumulative_huber"],
        "auxiliary_pinball": components["auxiliary_pinball"],
        "mean_pinball": components["return_pinball"],
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / max(elapsed, 1e-9),
        "eligible_dense_anchors": (
            eligible_anchor_count if loss_fn.objective.uses_dense_forward else None
        ),
        "unique_dense_anchors": (
            len(unique_dense_anchors) if loss_fn.objective.uses_dense_forward else None
        ),
        "dense_anchor_repeat_factor": (
            eligible_anchor_count / max(len(unique_dense_anchors), 1)
            if loss_fn.objective.uses_dense_forward
            else None
        ),
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
    loss_scales: LossScaleState,
) -> dict[str, Any]:
    return {
        "format_version": 4,
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
        "loss_scales": loss_scales.to_dict(),
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
    loss_scales: LossScaleState,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    if path.is_dir():
        path = path / "training_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != 4:
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
    if state.get("loss_scales") != loss_scales.to_dict():
        raise ValueError("resume loss-scale provenance mismatch")
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
    for_training: bool = False,
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
        require_complete_future=(for_training and config.objective.uses_dense_forward),
        expected_split=split,
        expected_dataset_id=config.data.dataset_id,
        expected_product=config.data.product,
        expected_target_name=config.data.target_name,
        expected_target_unit=config.data.target_unit,
        expected_target_price_source=config.data.target_price_source,
        expected_target_return_type=config.data.target_return_type,
        expected_target_timestamp_semantics=config.data.target_timestamp_semantics,
        expected_target_availability_lag_minutes=(config.data.target_availability_lag_minutes),
        expected_target_missing_policy=config.data.target_missing_policy,
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
        for_training=True,
    )
    val_data = _dataset(
        config,
        path=config.data.val_path,
        split="val",
        dates_path=config.data.val_dates_path,
        for_training=True,
    )
    if train_data.num_variates != val_data.num_variates:
        raise ValueError("train and validation variate counts differ")
    for key in (
        "dataset_id",
        "product",
        "target",
        "frequency_minutes",
        "session_minutes",
        "past_only_features",
        "past_only_availability_lag_minutes",
        "past_future_features",
        "feature_schema_sha256",
    ):
        if train_data.metadata.get(key) != val_data.metadata.get(key):
            raise ValueError(f"train/validation bundle contract mismatch: {key}")
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

    loss_scales = fit_loss_scales(
        train_data,
        config.objective,
        input_patch_length=int(model.backbone.input_patch_len),
        output_patch_length=int(model.backbone.output_patch_len),
        context_min=config.data.context_min,
        batch_size=max(config.trainer.batch_size, 1),
    )
    loss_fn = BusinessForecastLoss(
        model.quantiles,
        objective=config.objective,
        scales=loss_scales,
    ).to(device)
    auxiliary_indices = torch.tensor(
        [
            config.data.past_only_features.index(feature)
            for feature in config.objective.auxiliary_features
        ],
        dtype=torch.long,
        device=device,
    )
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
    with (output_dir / "loss_scales.json").open("w", encoding="utf-8") as handle:
        json.dump(loss_scales.to_dict(), handle, indent=2, sort_keys=True)
    summary = model.parameter_summary
    LOGGER.info(
        "run_start checkpoint=%s adapter=%s device=%s dtype=%s "
        "train_samples=%d val_samples=%d variates=%d context=%d..%d horizon=%d "
        "batch=%d effective_batch=%d objective=%s "
        "return_weight=%.3f cumulative_weight=%.3f auxiliary_weight=%.3f "
        "trainable=%d total=%d",
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
        config.objective.name,
        config.objective.return_pinball_weight,
        config.objective.cumulative_huber_weight,
        config.objective.auxiliary_weight,
        summary["trainable"],
        summary["total"],
    )
    LOGGER.info(
        "loss_scales source=train cumulative=%s auxiliary=%s provenance=%s",
        loss_scales.cumulative,
        loss_scales.auxiliary,
        loss_scales.date_file_sha256,
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
    checkpoint_mode = (
        "min" if config.trainer.checkpoint_metric in {"mean_pinball", "rmse"} else "max"
    )
    best_metric = float("inf") if checkpoint_mode == "min" else float("-inf")
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
            loss_scales=loss_scales,
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
            context_min=config.data.context_min,
            evaluation=config.evaluation,
            auxiliary_indices=auxiliary_indices,
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
            context_min=config.data.context_min,
            evaluation=config.evaluation,
            auxiliary_indices=auxiliary_indices,
            optimizer=None,
            scheduler=None,
            gradient_accumulation_steps=1,
            max_grad_norm=config.trainer.max_grad_norm,
            epoch=epoch,
            split="val",
            log_every_steps=config.trainer.log_every_steps,
        )
        selected_metric, selected_mode = _checkpoint_value(
            val_metrics,
            metric=config.trainer.checkpoint_metric,
            horizons=config.trainer.checkpoint_horizons,
        )
        if selected_mode != checkpoint_mode:
            raise RuntimeError("checkpoint metric mode changed during training")
        improved = (
            selected_metric < best_metric
            if checkpoint_mode == "min"
            else selected_metric > best_metric
        )
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
                "horizons": config.trainer.checkpoint_horizons,
                "mode": checkpoint_mode,
                "value": selected_metric,
                "best": best_metric,
                "improved": improved,
                "stale_epochs": stale_epochs,
            },
        }
        history.append(record)
        _write_history(output_dir / "history.jsonl", history)
        checkpoint_rows = [
            row
            for row in val_metrics["cumulative_horizons"]
            if row["horizon_minutes"] in config.trainer.checkpoint_horizons
        ]
        rank_values = [
            row["mean_daily_rank_ic"]
            for row in checkpoint_rows
            if row.get("mean_daily_rank_ic") is not None
        ]
        LOGGER.info(
            "epoch_end epoch=%d train_loss=%.6f val_loss=%.6f "
            "train_return_pinball=%.6f val_return_pinball=%.6f "
            "train_cumulative_huber=%.6f val_cumulative_huber=%.6f "
            "train_auxiliary_pinball=%.6f val_auxiliary_pinball=%.6f "
            "checkpoint_%s_h%s=%.6f mean_daily_rank_ic=%s net_utility=%s "
            "train_sps=%.2f val_sps=%.2f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["return_pinball"],
            val_metrics["return_pinball"],
            train_metrics["cumulative_huber"],
            val_metrics["cumulative_huber"],
            train_metrics["auxiliary_pinball"],
            val_metrics["auxiliary_pinball"],
            config.trainer.checkpoint_metric,
            ",".join(str(value) for value in config.trainer.checkpoint_horizons),
            selected_metric,
            (f"{np.mean(rank_values):.6f}" if rank_values else "null"),
            (
                f"{val_metrics['trading_proxy']['net_mean']:.6f}"
                if val_metrics["trading_proxy"]["net_mean"] is not None
                else "null"
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
            loss_scales=loss_scales,
        )
        validation_scorecard = {
            "loss": val_metrics["loss"],
            "return_pinball": val_metrics["return_pinball"],
            "cumulative_huber": val_metrics["cumulative_huber"],
            "auxiliary_pinball": val_metrics["auxiliary_pinball"],
            "cumulative_horizons": val_metrics["cumulative_horizons"],
            "mean_absolute_coverage_error": val_metrics["mean_absolute_coverage_error"],
            "q10_q90_coverage": val_metrics["q10_q90_coverage"],
            "mean_q10_q90_width": val_metrics["mean_q10_q90_width"],
            "quantile_crossing_rate": val_metrics["quantile_crossing_rate"],
            "trading_proxy": val_metrics["trading_proxy"],
        }
        checkpoint_metadata = {
            "num_variates": train_data.num_variates,
            "context_min": config.data.context_min,
            "context_max": config.data.context_max,
            "horizon_length": config.data.horizon_length,
            "past_only_features": config.data.past_only_features,
            "past_future_features": config.data.past_future_features,
            "checkpoint_metric": config.trainer.checkpoint_metric,
            "checkpoint_horizons": config.trainer.checkpoint_horizons,
            "checkpoint_mode": checkpoint_mode,
            "objective": config.objective.name,
            "training_graph": (
                "public_torch_full_sequence_forward"
                if config.objective.uses_dense_forward
                else "deployment_suffix_decode"
            ),
            "training_semantics_status": (
                "pretraining-like-engineering-route-not-official-recipe"
                if config.objective.uses_dense_forward
                else "deployment-consistent"
            ),
            "loss_scales": loss_scales.to_dict(),
            "validation_scorecard": validation_scorecard,
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
