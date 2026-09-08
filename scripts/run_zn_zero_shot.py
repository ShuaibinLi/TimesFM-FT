#!/usr/bin/env python3
"""Prepare June 2025 ZN WMP windows and evaluate TimesFM 3 zero-shot."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

import fsspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import ExperimentConfig
from timesfm_ft.data import NpzWindowDataset
from timesfm_ft.evaluator import EvaluationAccumulator
from timesfm_ft.trainer import resolve_device

LOGGER = logging.getLogger("zn_zero_shot")
plt.switch_backend("Agg")
DEFAULT_SOURCE = (
    "gs://vatic-tmp-30d/shuaibin.li/timesfm_ft/"
    "wmp_500ms_202506_v2/ZN"
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_windows(
    source_root: str,
    output_path: Path,
    *,
    context_length: int,
    horizon_length: int,
    stride: int,
    interval_ns: int,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        LOGGER.info("reuse_windows path=%s", output_path)
        return output_path

    fs, source_path = fsspec.core.url_to_fs(source_root.rstrip("/"))
    files = sorted(path for path in fs.find(source_path) if path.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files below {source_root}")

    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    rows_by_day: dict[str, int] = {}
    samples_by_day: dict[str, int] = {}

    for source in files:
        day = next(
            part.removeprefix("date=")
            for part in PurePosixPath(source).parts
            if part.startswith("date=")
        )
        with fs.open(source, "rb") as stream:
            table = pq.read_table(stream, columns=["timestamp_ns", "wmp"])
        timestamp = table.column("timestamp_ns").to_numpy(zero_copy_only=False)
        wmp = table.column("wmp").to_numpy(zero_copy_only=False).astype(np.float32)
        if len(timestamp) < context_length + horizon_length:
            raise ValueError(f"{source} has only {len(timestamp)} rows")
        if not np.all(np.diff(timestamp) == interval_ns):
            raise ValueError(f"{source} is not a {interval_ns} ns regular grid")
        if not np.all(np.isfinite(wmp)):
            raise ValueError(f"{source} contains non-finite WMP values")

        cutoffs = np.arange(
            context_length - 1,
            len(wmp) - horizon_length,
            stride,
            dtype=np.int64,
        )
        context_starts = cutoffs - context_length + 1
        future_starts = cutoffs + 1
        contexts.append(
            np.stack([wmp[start : start + context_length] for start in context_starts])
        )
        futures.append(
            np.stack([wmp[start : start + horizon_length] for start in future_starts])
        )
        timestamps.append(timestamp[cutoffs].astype(np.int64))
        dates.append(np.full(len(cutoffs), int(day), dtype=np.int32))
        rows_by_day[day] = len(wmp)
        samples_by_day[day] = len(cutoffs)

    context_values = np.concatenate(contexts)
    future_values = np.concatenate(futures)
    cutoff_timestamps = np.concatenate(timestamps)
    sample_dates = np.concatenate(dates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        context_values=context_values,
        future_values=future_values,
        timestamps=cutoff_timestamps,
        dates=sample_dates,
    )
    metadata = {
        "source_root": source_root,
        "source_days": len(files),
        "context_length": context_length,
        "horizon_length": horizon_length,
        "stride": stride,
        "sampling_interval_ns": interval_ns,
        "samples": len(context_values),
        "rows_by_day": rows_by_day,
        "samples_by_day": samples_by_day,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "prepared_windows days=%d samples=%d context=%d horizon=%d stride=%d path=%s",
        len(files),
        len(context_values),
        context_length,
        horizon_length,
        stride,
        output_path,
    )
    return output_path


def _metric_rows(
    predictions: np.ndarray,
    targets: np.ndarray,
    origins: np.ndarray,
    target_mask: np.ndarray,
    *,
    tick_size: float,
    interval_seconds: float,
) -> list[dict[str, object]]:
    prediction_ticks = (predictions - origins[:, None]) / tick_size
    target_ticks = (targets - origins[:, None]) / tick_size
    rows: list[dict[str, object]] = []
    for index in range(predictions.shape[1]):
        valid = ~target_mask[:, index]
        prediction_error = (
            prediction_ticks[valid, index] - target_ticks[valid, index]
        )
        persistence_error = target_ticks[valid, index]
        nonzero = persistence_error != 0.0
        baseline_sse = np.sum(persistence_error**2)
        rows.append(
            {
                "step": index + 1,
                "horizon_seconds": (index + 1) * interval_seconds,
                "mae_ticks": float(np.mean(np.abs(prediction_error))),
                "persistence_mae_ticks": float(np.mean(np.abs(persistence_error))),
                "rmse_ticks": float(np.sqrt(np.mean(prediction_error**2))),
                "persistence_rmse_ticks": float(
                    np.sqrt(np.mean(persistence_error**2))
                ),
                "oos_r2_vs_persistence": (
                    float(1.0 - np.sum(prediction_error**2) / baseline_sse)
                    if baseline_sse > 0
                    else None
                ),
                "directional_accuracy": float(
                    np.mean(
                        np.sign(prediction_ticks[valid, index][nonzero])
                        == np.sign(persistence_error[nonzero])
                    )
                )
                if np.any(nonzero)
                else None,
                "nonzero_targets": int(np.sum(nonzero)),
            }
        )
    return rows


def _daily_rows(
    predictions: np.ndarray,
    targets: np.ndarray,
    origins: np.ndarray,
    dates: np.ndarray,
    target_mask: np.ndarray,
    *,
    tick_size: float,
    horizon_seconds: float,
) -> list[dict[str, object]]:
    prediction_ticks = (predictions[:, -1] - origins) / tick_size
    target_ticks = (targets[:, -1] - origins) / tick_size
    rows: list[dict[str, object]] = []
    for day in sorted(np.unique(dates)):
        selected = (dates == day) & ~target_mask[:, -1]
        error = prediction_ticks[selected] - target_ticks[selected]
        baseline = target_ticks[selected]
        nonzero = baseline != 0.0
        rows.append(
            {
                "date": int(day),
                "samples": int(np.sum(selected)),
                "horizon_seconds": horizon_seconds,
                "mae_ticks": float(np.mean(np.abs(error))),
                "persistence_mae_ticks": float(np.mean(np.abs(baseline))),
                "directional_accuracy": (
                    float(
                    np.mean(
                        np.sign(prediction_ticks[selected][nonzero])
                        == np.sign(baseline[nonzero])
                    )
                    )
                    if np.any(nonzero)
                    else None
                ),
            }
        )
    return rows


def _plot_horizons(rows: list[dict[str, object]], destination: Path) -> None:
    seconds = np.asarray([row["horizon_seconds"] for row in rows])
    mae = np.asarray([row["mae_ticks"] for row in rows])
    baseline = np.asarray([row["persistence_mae_ticks"] for row in rows])
    r2 = np.asarray([row["oos_r2_vs_persistence"] for row in rows])
    direction = np.asarray([row["directional_accuracy"] for row in rows])

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(seconds, mae, label="TimesFM P50")
    axes[0].plot(seconds, baseline, label="Persistence")
    axes[0].set(xlabel="Forecast horizon (seconds)", ylabel="MAE (ZN ticks)")
    axes[0].legend()
    axes[1].plot(seconds, r2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Forecast horizon (seconds)", ylabel="OOS R² vs persistence")
    axes[2].plot(seconds, direction)
    axes[2].axhline(0.5, color="black", linewidth=0.8)
    axes[2].set(
        xlabel="Forecast horizon (seconds)",
        ylabel="Directional accuracy",
        ylim=(0.0, 1.0),
    )
    figure.suptitle("ZN WMP zero-shot TimesFM 3 · June 2025")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_daily(rows: list[dict[str, object]], destination: Path) -> None:
    labels = [str(row["date"])[4:] for row in rows]
    x = np.arange(len(rows))
    horizon_seconds = float(rows[0]["horizon_seconds"])
    mae = [row["mae_ticks"] for row in rows]
    baseline = [row["persistence_mae_ticks"] for row in rows]
    direction = [row["directional_accuracy"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, mae, marker="o", label="TimesFM P50")
    axes[0].plot(x, baseline, marker="o", label="Persistence")
    axes[0].set(ylabel=f"{horizon_seconds:g}s MAE (ZN ticks)")
    axes[0].legend()
    axes[1].plot(x, direction, marker="o")
    axes[1].axhline(0.5, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="June 2025 trade date (MMDD)",
        ylabel=f"{horizon_seconds:g}s directional accuracy",
        ylim=(0.0, 1.0),
        xticks=x,
        xticklabels=labels,
    )
    figure.suptitle("Daily ZN WMP zero-shot performance")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_quantile_fan(
    axis: plt.Axes,
    x_values: np.ndarray,
    predictions: np.ndarray,
    quantiles: np.ndarray,
    *,
    horizon_seconds: float,
) -> None:
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    for lower_index in range(median_index):
        upper_index = len(quantiles) - lower_index - 1
        axis.fill_between(
            x_values,
            predictions[:, lower_index],
            predictions[:, upper_index],
            color="tab:orange",
            alpha=0.07 + lower_index * 0.04,
            label="P10–P90 prediction region" if lower_index == 0 else None,
            linewidth=0,
        )
    for quantile_index in range(len(quantiles)):
        if quantile_index == median_index:
            continue
        axis.plot(
            x_values,
            predictions[:, quantile_index],
            color="tab:orange",
            alpha=0.35,
            linewidth=0.55,
        )
    axis.plot(
        x_values,
        predictions[:, median_index],
        color="tab:orange",
        label=f"TimesFM P50 point forecast ({horizon_seconds:g}s)",
        linewidth=2.0,
        zorder=5,
    )


def _plot_example(
    context: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    quantiles: np.ndarray,
    destination: Path,
    *,
    interval_seconds: float,
    tick_size: float,
) -> None:
    shown_context = min(240, len(context))
    context_x = np.arange(-shown_context + 1, 1) * interval_seconds
    future_x = np.arange(1, len(target) + 1) * interval_seconds
    origin = context[-1]
    context_ticks = (context - origin) / tick_size
    target_ticks = (target - origin) / tick_size
    prediction_ticks = (predictions - origin) / tick_size
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        context_x,
        context_ticks[-shown_context:],
        label="Observed context WMP",
        linewidth=1.4,
    )
    _plot_quantile_fan(
        axis,
        future_x,
        prediction_ticks,
        quantiles,
        horizon_seconds=len(target) * interval_seconds,
    )
    axis.plot(
        future_x,
        target_ticks,
        label="Actual future WMP",
        color="black",
        linewidth=2.0,
        zorder=6,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Representative ZN WMP forecast",
        xlabel="Seconds from forecast cutoff",
        ylabel="WMP displacement from cutoff (ZN ticks)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_daily_price_curves(
    dataset: NpzWindowDataset,
    predictions: np.ndarray,
    quantiles: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    timestamps: np.ndarray,
    destination: Path,
    *,
    interval_seconds: float,
    tick_size: float,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    interval_ns = int(interval_seconds * 1_000_000_000)
    horizon = targets.shape[1]
    context_length = dataset.context_values.shape[-1]
    eastern = ZoneInfo("America/New_York")

    for day in sorted(np.unique(dates)):
        selected = np.flatnonzero(dates == day)
        time_formatter = mdates.DateFormatter("%H:%M", tz=eastern)

        figure = plt.figure(figsize=(20, 23), constrained_layout=True)
        grid = figure.add_gridspec(
            6,
            4,
            height_ratios=[0.16, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        legend_axis = figure.add_subplot(grid[0, :])
        legend_axis.axis("off")
        detail_positions = np.linspace(
            0,
            len(selected) - 1,
            20,
            dtype=np.int64,
        )
        for panel, position in enumerate(detail_positions):
            sample_index = int(selected[position])
            cutoff = int(timestamps[sample_index])
            context_times = cutoff - np.arange(
                context_length - 1, -1, -1, dtype=np.int64
            ) * interval_ns
            future_times = cutoff + np.arange(
                1, horizon + 1, dtype=np.int64
            ) * interval_ns
            context_x = mdates.date2num(context_times.astype("datetime64[ns]"))
            future_x = mdates.date2num(future_times.astype("datetime64[ns]"))
            context_values = dataset.context_values[sample_index, 0]
            origin = context_values[-1]
            context_ticks = (context_values - origin) / tick_size
            target_ticks = (targets[sample_index] - origin) / tick_size
            prediction_ticks = (predictions[sample_index] - origin) / tick_size
            axis = figure.add_subplot(grid[1 + panel // 4, panel % 4])
            axis.plot(
                context_x,
                context_ticks,
                label=f"Context ({context_length * interval_seconds:g}s)",
                linewidth=1.3,
            )
            _plot_quantile_fan(
                axis,
                future_x,
                prediction_ticks,
                quantiles,
                horizon_seconds=horizon * interval_seconds,
            )
            axis.plot(
                future_x,
                target_ticks,
                label=f"Actual future ({horizon * interval_seconds:g}s)",
                color="black",
                linewidth=2.0,
                zorder=6,
            )
            axis.axvline(
                mdates.date2num(np.datetime64(cutoff, "ns")),
                color="gray",
                linestyle="--",
                linewidth=0.8,
            )
            cutoff_label = datetime.fromtimestamp(cutoff / 1e9, eastern).strftime(
                "%H:%M:%S ET"
            )
            axis.set_title(f"Forecast cutoff {cutoff_label}", fontsize=10)
            axis.xaxis.set_major_formatter(time_formatter)
            axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            axis.tick_params(axis="x", labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            axis.grid(alpha=0.2)
            if panel % 4 == 0:
                axis.set_ylabel("Δ WMP (ticks)")
            if panel >= 16:
                axis.set_xlabel("Actual market time (ET)")

        handles, labels = figure.axes[1].get_legend_handles_labels()
        legend_axis.legend(
            handles,
            labels,
            loc="center",
            ncol=4,
        )
        figure.suptitle(
            f"ZN WMP displacement from each cutoff · {int(day)} · "
            f"20 representative windows of {len(selected)}",
            fontsize=16,
        )
        figure.savefig(
            destination / f"{int(day)}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)


def evaluate(
    config: ExperimentConfig,
    data_path: Path,
    output_dir: Path,
    *,
    stride: int,
    max_samples: int | None,
) -> None:
    config.validate()
    dataset = NpzWindowDataset(
        data_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=1,
    )
    archive = np.load(data_path, allow_pickle=False)
    dates = np.asarray(archive["dates"])
    timestamps = np.asarray(archive["timestamps"])
    indices = np.arange(len(dataset))
    if max_samples is not None and max_samples < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, max_samples, dtype=np.int64)
        loader_dataset = Subset(dataset, indices.tolist())
        dates = dates[indices]
        timestamps = timestamps[indices]
    else:
        loader_dataset = dataset

    device = resolve_device(config.trainer.device)
    loader = DataLoader(
        loader_dataset,
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
        configure_for_training=False,
    )
    model.eval()
    accumulator = EvaluationAccumulator(
        horizon=config.data.horizon_length,
        quantiles=model.quantiles,
        tick_size=config.objective.tick_size,
        sampling_interval_seconds=config.data.sampling_interval_seconds,
    )
    quantile_values = np.asarray(model.quantiles, dtype=np.float32)
    median_index = int(np.argmin(np.abs(quantile_values - 0.5)))
    quantile_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    started = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        context = batch["context_values"].to(device)
        context_mask = batch["context_mask"].to(device)
        future = batch["future_values"].to(device)
        future_mask = batch["future_mask"].to(device)
        prediction = model.predict(
            context,
            horizon=config.data.horizon_length,
            context_mask=context_mask,
        )
        origin = context[:, 0, -1]
        accumulator.update(prediction, future, origin, future_mask)
        quantile_predictions.append(prediction.float().cpu().numpy())
        targets.append(future.float().cpu().numpy())
        target_masks.append(future_mask.cpu().numpy())
        origins.append(origin.float().cpu().numpy())
        if step % 25 == 0 or step == len(loader):
            LOGGER.info("inference_progress step=%d/%d", step, len(loader))

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    quantile_prediction_array = np.concatenate(quantile_predictions)
    prediction_array = quantile_prediction_array[:, :, median_index]
    target_array = np.concatenate(targets)
    target_mask_array = np.concatenate(target_masks)
    origin_array = np.concatenate(origins)
    summary, probabilistic_rows = accumulator.results()
    horizon_rows = _metric_rows(
        prediction_array,
        target_array,
        origin_array,
        target_mask_array,
        tick_size=config.objective.tick_size,
        interval_seconds=config.data.sampling_interval_seconds,
    )
    for row, probabilistic in zip(horizon_rows, probabilistic_rows, strict=True):
        row["mean_pinball_ticks"] = probabilistic["mean_pinball_ticks"]
    daily_rows = _daily_rows(
        prediction_array,
        target_array,
        origin_array,
        dates,
        target_mask_array,
        tick_size=config.objective.tick_size,
        horizon_seconds=(
            config.data.horizon_length * config.data.sampling_interval_seconds
        ),
    )

    summary.update(
        {
            "model": "google/timesfm-3.0-pytorch",
            "product": "ZN",
            "mode": "zero-shot",
            "tick_size": config.objective.tick_size,
            "context_points": config.data.context_length,
            "context_seconds": (
                config.data.context_length * config.data.sampling_interval_seconds
            ),
            "horizon_points": config.data.horizon_length,
            "horizon_seconds": (
                config.data.horizon_length * config.data.sampling_interval_seconds
            ),
            "window_stride_points": stride,
            "inference_seconds": elapsed,
            "samples_per_second": len(loader_dataset) / elapsed,
            "endpoint": horizon_rows[-1],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "per_horizon.csv", horizon_rows)
    _write_csv(output_dir / "daily.csv", daily_rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        timestamps=timestamps,
        dates=dates,
        origins=origin_array,
        actual=target_array,
        future_mask=target_mask_array,
        prediction_p50=prediction_array,
        prediction_quantiles=quantile_prediction_array,
        quantiles=quantile_values,
    )
    _plot_horizons(horizon_rows, output_dir / "horizon_metrics.png")
    _plot_daily(daily_rows, output_dir / "daily_metrics.png")
    example = len(prediction_array) // 2
    source_index = int(indices[example])
    _plot_example(
        dataset.context_values[source_index, 0],
        target_array[example],
        quantile_prediction_array[example],
        quantile_values,
        output_dir / "forecast_example.png",
        interval_seconds=config.data.sampling_interval_seconds,
        tick_size=config.objective.tick_size,
    )
    if max_samples is None:
        _plot_daily_price_curves(
            dataset,
            quantile_prediction_array,
            quantile_values,
            target_array,
            dates,
            timestamps,
            output_dir / "daily_price_curves",
            interval_seconds=config.data.sampling_interval_seconds,
            tick_size=config.objective.tick_size,
        )
    LOGGER.info(
        "complete samples=%d elapsed=%.1fs mae_ticks=%.4f endpoint_direction=%.4f output=%s",
        len(loader_dataset),
        elapsed,
        summary["mae_ticks"],
        horizon_rows[-1]["directional_accuracy"],
        output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/zn_zero_shot.json")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--stride",
        type=int,
        help="Forecast-cutoff stride in points (default: horizon length).",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = ExperimentConfig.from_json(args.config)
    stride = args.stride or config.data.horizon_length
    if stride <= 0:
        parser.error("--stride must be positive")
    data_path = Path(args.data or config.data.val_path)
    output_dir = Path(
        args.output_dir or Path(config.trainer.output_dir) / "evaluation"
    )
    data_path = prepare_windows(
        args.source_root,
        data_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        stride=stride,
        interval_ns=int(config.data.sampling_interval_seconds * 1_000_000_000),
        force=args.force_data,
    )
    evaluate(
        config,
        data_path,
        output_dir,
        stride=stride,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
