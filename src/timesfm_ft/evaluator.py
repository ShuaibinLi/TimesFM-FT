"""Zero-shot and adapted evaluation for 1-minute intraday forecasts."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import ExperimentConfig
from timesfm_ft.metrics import ForecastMetricsAccumulator
from timesfm_ft.trainer import _dataset, _make_loader, _move_batch, resolve_device

LOGGER = logging.getLogger(__name__)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_experiment(
    config: ExperimentConfig,
    *,
    data_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
    split: Literal["val", "test"] | None = None,
) -> Path:
    """Evaluates a chronological validation/test bundle and writes all reports."""

    config.validate()
    if data_path is not None and split is not None:
        raise ValueError("data_path and split are mutually exclusive")
    selected_split: Literal["val", "test"]
    if data_path is not None:
        selected_path = Path(data_path)
        if selected_path.resolve() == Path(config.data.train_path).resolve():
            raise ValueError("evaluation on data.train_path is not allowed")
        metadata_path = selected_path / "manifest.json"
        if not metadata_path.exists():
            raise ValueError("explicit evaluation data requires manifest.json")
        with metadata_path.open(encoding="utf-8") as handle:
            declared_split = json.load(handle).get("split")
        if declared_split not in {"val", "test"}:
            raise ValueError("explicit evaluation bundle must declare val or test")
        selected_split = declared_split
        dates_path = (
            config.data.test_dates_path if selected_split == "test" else config.data.val_dates_path
        )
    else:
        selected_split = split or ("test" if config.data.test_path else "val")
        if selected_split == "test":
            if config.data.test_path is None:
                raise ValueError("data.test_path is required for test evaluation")
            selected_path = Path(config.data.test_path)
            dates_path = config.data.test_dates_path
        else:
            selected_path = Path(config.data.val_path)
            dates_path = config.data.val_dates_path
    dataset = _dataset(
        config,
        path=str(selected_path),
        split=selected_split,
        dates_path=dates_path,
    )
    device = resolve_device(device_name or config.trainer.device)
    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
        configure_for_training=adapter_path is not None,
    )
    if adapter_path is not None:
        model.load_adapter(
            adapter_path,
            expected_metadata={
                "num_variates": dataset.num_variates,
                "context_min": config.data.context_min,
                "context_max": config.data.context_max,
                "horizon_length": config.data.horizon_length,
                "past_only_features": list(config.data.past_only_features),
                "past_future_features": list(config.data.past_future_features),
            },
        )
    model.eval()
    loader = _make_loader(
        dataset,
        batch_size=batch_size or config.trainer.batch_size,
        patch_length=int(model.backbone.input_patch_len),
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
        generator=None,
    )
    accumulator = ForecastMetricsAccumulator(
        horizon=config.data.horizon_length,
        quantiles=model.quantiles,
        report_horizons=config.evaluation.report_horizons,
        trading_horizon=config.evaluation.trading_horizon,
        cost_per_turnover=config.evaluation.cost_per_turnover,
    )
    LOGGER.info(
        "eval_start checkpoint=%s adapter=%s split=%s samples=%d variates=%d "
        "context=%d..%d horizon=%d",
        config.model.checkpoint,
        adapter_path or "zero-shot",
        selected_split,
        len(dataset),
        dataset.num_variates,
        config.data.context_min,
        config.data.context_max,
        config.data.horizon_length,
    )
    for step, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        predictions = model.predict(
            batch["context_values"],
            horizon=config.data.horizon_length,
            context_mask=batch["context_mask"],
            context_padding_mask=batch["context_padding_mask"],
            past_future_values=batch["past_future_values"],
            past_future_mask=batch["past_future_mask"],
        )
        accumulator.update(
            predictions,
            batch["unknown_future_values"][:, 0],
            batch["unknown_future_mask"][:, 0],
            last_returns=batch["last_returns"],
            context_lengths=batch["context_lengths"],
            dates=batch["dates"],
            timestamps=batch["timestamps"],
            minute_indices=batch["minute_indices"],
            context_volatility=batch["context_volatility"],
        )
        if step % config.trainer.log_every_steps == 0 or step == len(loader):
            LOGGER.info("eval_progress step=%d/%d", step, len(loader))

    summary, lead_rows, cumulative_rows, slice_rows = accumulator.results()
    summary.update(
        {
            "evaluated_split": selected_split,
            "data_path": str(selected_path),
            "dataset_id": config.data.dataset_id,
            "target_name": config.data.target_name,
            "target_unit": config.data.target_unit,
            "past_only_features": config.data.past_only_features,
            "past_future_features": config.data.past_future_features,
        }
    )
    destination = Path(
        output_dir or Path(config.trainer.output_dir) / f"evaluation-{selected_split}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    _write_rows(destination / "per_lead.csv", lead_rows)
    _write_rows(destination / "cumulative_horizons.csv", cumulative_rows)
    _write_rows(destination / "slices.csv", slice_rows)
    if config.evaluation.save_predictions:
        np.savez(
            destination / "predictions.npz",
            **accumulator.prediction_arrays(),
        )

    cumulative_lookup = {row["horizon_minutes"]: row for row in cumulative_rows}
    trading_row = cumulative_lookup.get(config.evaluation.trading_horizon, {})
    LOGGER.info(
        "eval_end pinball=%s rank_ic_%dm=%s net_utility=%s artifacts=%s",
        (f"{summary['mean_pinball']:.6f}" if summary["mean_pinball"] is not None else "null"),
        config.evaluation.trading_horizon,
        (f"{trading_row['rank_ic']:.6f}" if trading_row.get("rank_ic") is not None else "null"),
        (
            f"{summary['trading_proxy']['net_mean']:.6f}"
            if summary["trading_proxy"]["net_mean"] is not None
            else "null"
        ),
        destination,
    )
    return destination
