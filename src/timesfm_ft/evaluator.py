"""Inference and evaluation for fine-tuned TimesFM 3 adapters."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import ExperimentConfig
from timesfm_ft.data import NpzWindowDataset, WindowBatch
from timesfm_ft.metrics import ForecastMetricsAccumulator
from timesfm_ft.trainer import resolve_device

LOGGER = logging.getLogger(__name__)

# Backward-compatible import surface for existing callers and tests.
EvaluationAccumulator = ForecastMetricsAccumulator


def _read_expected_dates(path: str | None) -> set[int] | None:
    if path is None:
        return None
    date_path = Path(path)
    return {
        int(line.strip())
        for line in date_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _move_batch(batch: WindowBatch, device: torch.device) -> WindowBatch:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def _read_dataset_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path / "manifest.json" if path.is_dir() else path.with_suffix(".json")
    if not metadata_path.exists():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"dataset metadata must be an object: {metadata_path}")
    return metadata


def evaluate_experiment(
    config: ExperimentConfig,
    *,
    data_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
    split: Literal["val", "test"] | None = None,
    allow_unsafe_data: bool = False,
) -> Path:
    """Loads the official model, optionally applies an adapter, and evaluates."""

    config.validate()
    device = resolve_device(device_name or config.trainer.device)
    if data_path is not None and split is not None:
        raise ValueError("data_path and split are mutually exclusive")
    selected_split: Literal["val", "test"] | None
    if data_path is not None:
        selected_path = Path(data_path)
        if selected_path.resolve() == Path(config.data.train_path).resolve():
            raise ValueError("evaluation on data.train_path is not allowed")
        metadata = _read_dataset_metadata(selected_path)
        if allow_unsafe_data:
            selected_split = None
            expected_dates = None
            expected_dates_path = None
        else:
            if metadata is None:
                raise ValueError(
                    "explicit evaluation data requires metadata; "
                    "pass --unsafe-data to bypass provenance checks"
                )
            declared_split = metadata.get("split")
            if declared_split not in {"val", "test"}:
                raise ValueError(
                    f"explicit evaluation data declares split={declared_split!r}; "
                    "only val/test are allowed"
                )
            selected_split = declared_split
            if selected_split == "test":
                expected_dates_path = config.data.test_dates_path
            else:
                expected_dates_path = config.data.val_dates_path
            expected_dates = _read_expected_dates(expected_dates_path)
    else:
        selected_split = split or ("test" if config.data.test_path else "val")
        if selected_split == "test":
            if config.data.test_path is None:
                raise ValueError("data.test_path is required for --split test")
            selected_path = Path(config.data.test_path)
            expected_dates = _read_expected_dates(config.data.test_dates_path)
            expected_dates_path = config.data.test_dates_path
        else:
            selected_path = Path(config.data.val_path)
            expected_dates = _read_expected_dates(config.data.val_dates_path)
            expected_dates_path = config.data.val_dates_path
    dataset = NpzWindowDataset(
        selected_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=config.data.max_variates,
        sampling_interval_seconds=(
            None
            if data_path is not None and allow_unsafe_data
            else config.data.sampling_interval_seconds
        ),
        expected_stride=(
            None
            if data_path is not None and allow_unsafe_data
            else config.data.stride
        ),
        expected_product=(
            None
            if data_path is not None and allow_unsafe_data
            else config.data.product
        ),
        expected_split=selected_split,
        expected_dates=expected_dates,
        expected_dates_path=expected_dates_path,
        require_metadata=(
            config.data.require_metadata
            or (data_path is not None and not allow_unsafe_data)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
        configure_for_training=adapter_path is not None,
    )
    if adapter_path is not None:
        model.load_adapter(adapter_path)
    model.eval()
    accumulator = ForecastMetricsAccumulator(
        horizon=config.data.horizon_length,
        quantiles=model.quantiles,
        tick_size=config.objective.tick_size,
        sampling_interval_seconds=config.data.sampling_interval_seconds,
    )

    LOGGER.info(
        "eval_start checkpoint=%s adapter=%s data=%s samples=%d variates=%d "
        "context=%d horizon=%d device=%s",
        config.model.checkpoint,
        adapter_path or "zero-shot",
        selected_path,
        len(dataset),
        dataset.num_variates,
        config.data.context_length,
        config.data.horizon_length,
        device,
    )
    for step, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        predictions = model.predict(
            batch["context_values"],
            horizon=config.data.horizon_length,
            context_mask=batch["context_mask"],
        )
        accumulator.update(
            predictions,
            batch["future_values"],
            batch["context_values"][:, 0, -1],
            batch["future_mask"],
        )
        if step % config.trainer.log_every_steps == 0 or step == len(loader):
            LOGGER.info("eval_progress step=%d/%d", step, len(loader))

    summary, horizon_rows = accumulator.results()
    summary["evaluated_split"] = selected_split or "explicit"
    summary["data_path"] = str(selected_path)
    destination = Path(output_dir or Path(config.trainer.output_dir) / "evaluation")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    with (destination / "per_horizon.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)

    metric_text = {
        key: f"{value:.6f}" if value is not None else "null"
        for key, value in {
            "mae": summary["mae_ticks"],
            "rmse": summary["rmse_ticks"],
            "r2": summary["oos_r2_vs_persistence"],
            "pinball": summary["mean_pinball_ticks"],
            "crossing": summary["quantile_crossing_rate"],
        }.items()
    }
    LOGGER.info(
        "eval_end mae_ticks=%s rmse_ticks=%s oos_r2=%s "
        "pinball_ticks=%s crossing_rate=%s artifacts=%s",
        metric_text["mae"],
        metric_text["rmse"],
        metric_text["r2"],
        metric_text["pinball"],
        metric_text["crossing"],
        destination,
    )
    return destination
