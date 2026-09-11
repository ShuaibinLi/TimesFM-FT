#!/usr/bin/env python3
"""Extract one non-overlapping forecast per minute from predictions.npz."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_ft.metrics import _pearson, _rank_ic, binary_direction_metrics

MINUTE_NS = 60_000_000_000


def _fixed_lead_points(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    dates: np.ndarray,
    anchor_timestamps: np.ndarray,
    minute_indices: np.ndarray,
    context_lengths: np.ndarray,
    quantile_index: int,
    lead: int,
) -> dict[str, np.ndarray]:
    lead_index = lead - 1
    valid = ~target_mask[:, lead_index]
    return {
        "prediction": predictions[valid, lead_index, quantile_index],
        "target": targets[valid, lead_index],
        "date": dates[valid],
        "anchor_timestamp": anchor_timestamps[valid],
        "target_timestamp": anchor_timestamps[valid] + lead * MINUTE_NS,
        "minute_index": minute_indices[valid],
        "context_length": context_lengths[valid],
        "lead": np.full(int(valid.sum()), lead, dtype=np.int16),
    }


def _block_points(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    dates: np.ndarray,
    anchor_timestamps: np.ndarray,
    minute_indices: np.ndarray,
    context_lengths: np.ndarray,
    quantile_index: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    if not 1 <= block_size <= predictions.shape[1]:
        raise ValueError(f"block_size must be within 1..{predictions.shape[1]}")
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "prediction",
            "target",
            "date",
            "anchor_timestamp",
            "target_timestamp",
            "minute_index",
            "context_length",
            "lead",
        )
    }
    leads = np.arange(1, block_size + 1, dtype=np.int16)
    for date in np.unique(dates):
        day_indices = np.flatnonzero(dates == date)
        day_indices = day_indices[np.argsort(minute_indices[day_indices], kind="stable")]
        first_minute = int(minute_indices[day_indices[0]])
        origins = day_indices[
            (minute_indices[day_indices].astype(np.int64) - first_minute) % block_size == 0
        ]
        for origin in origins:
            valid_leads = ~target_mask[origin, :block_size]
            selected_leads = leads[valid_leads]
            rows["prediction"].append(predictions[origin, :block_size, quantile_index][valid_leads])
            rows["target"].append(targets[origin, :block_size][valid_leads])
            rows["date"].append(np.full(len(selected_leads), date, dtype=np.int32))
            rows["anchor_timestamp"].append(
                np.full(len(selected_leads), anchor_timestamps[origin], dtype=np.int64)
            )
            rows["target_timestamp"].append(
                anchor_timestamps[origin] + selected_leads.astype(np.int64) * MINUTE_NS
            )
            rows["minute_index"].append(
                np.full(len(selected_leads), minute_indices[origin], dtype=np.int16)
            )
            rows["context_length"].append(
                np.full(len(selected_leads), context_lengths[origin], dtype=np.int16)
            )
            rows["lead"].append(selected_leads)
    return {
        name: np.concatenate(values) if values else np.asarray([]) for name, values in rows.items()
    }


def build_daily_vectors(
    source: Path,
    *,
    output_dir: Path,
    lead: int = 1,
    quantile: float = 0.5,
    mode: str = "fixed-lead",
    block_size: int = 64,
) -> dict[str, Any]:
    payload = np.load(source)
    predictions = payload["predictions"]
    targets = payload["targets"]
    target_mask = payload["target_mask"].astype(np.bool_, copy=False)
    quantiles = payload["quantiles"]
    dates = payload["dates"].astype(np.int32, copy=False)
    anchor_timestamps = payload["timestamps"].astype(np.int64, copy=False)
    minute_indices = payload["minute_indices"].astype(np.int16, copy=False)
    context_lengths = payload["context_lengths"].astype(np.int16, copy=False)
    if predictions.ndim != 3 or targets.shape != predictions.shape[:2]:
        raise ValueError("predictions/targets have incompatible shapes")
    if target_mask.shape != targets.shape:
        raise ValueError("target_mask must match targets")
    quantile_index = int(np.argmin(np.abs(quantiles - quantile)))
    if abs(float(quantiles[quantile_index]) - quantile) > 1e-8:
        raise ValueError(f"requested quantile={quantile} is not present")

    if mode == "fixed-lead":
        if not 1 <= lead <= predictions.shape[1]:
            raise ValueError(f"lead must be within 1..{predictions.shape[1]}")
        points = _fixed_lead_points(
            predictions=predictions,
            targets=targets,
            target_mask=target_mask,
            dates=dates,
            anchor_timestamps=anchor_timestamps,
            minute_indices=minute_indices,
            context_lengths=context_lengths,
            quantile_index=quantile_index,
            lead=lead,
        )
        definition = "one fixed-lead P50 prediction per unique target minute"
    elif mode == "blocks":
        points = _block_points(
            predictions=predictions,
            targets=targets,
            target_mask=target_mask,
            dates=dates,
            anchor_timestamps=anchor_timestamps,
            minute_indices=minute_indices,
            context_lengths=context_lengths,
            quantile_index=quantile_index,
            block_size=block_size,
        )
        definition = "non-overlapping forecast blocks tiled from the first eligible origin"
    else:
        raise ValueError(f"unsupported mode={mode!r}")
    point_prediction = points["prediction"].astype(np.float32, copy=False)
    point_target = points["target"].astype(np.float32, copy=False)
    point_dates = points["date"].astype(np.int32, copy=False)
    point_anchor_timestamps = points["anchor_timestamp"].astype(np.int64, copy=False)
    point_target_timestamps = points["target_timestamp"].astype(np.int64, copy=False)
    point_minute_indices = points["minute_index"].astype(np.int16, copy=False)
    point_context_lengths = points["context_length"].astype(np.int16, copy=False)
    point_leads = points["lead"].astype(np.int16, copy=False)
    order = np.lexsort((point_target_timestamps, point_dates))
    arrays = [
        point_prediction,
        point_target,
        point_dates,
        point_anchor_timestamps,
        point_target_timestamps,
        point_minute_indices,
        point_context_lengths,
        point_leads,
    ]
    (
        point_prediction,
        point_target,
        point_dates,
        point_anchor_timestamps,
        point_target_timestamps,
        point_minute_indices,
        point_context_lengths,
        point_leads,
    ) = (array[order] for array in arrays)

    keys = np.rec.fromarrays(
        [point_dates, point_target_timestamps],
        names=("date", "target_timestamp"),
    )
    if len(np.unique(keys)) != len(keys):
        raise ValueError("daily series contains duplicate target minutes")
    if not np.isfinite(point_prediction).all() or not np.isfinite(point_target).all():
        raise ValueError("daily series contains non-finite valid values")

    unique_dates = np.unique(point_dates)
    lengths = np.asarray(
        [(point_dates == date).sum() for date in unique_dates],
        dtype=np.int16,
    )
    width = int(lengths.max())
    daily_prediction = np.full((len(unique_dates), width), np.nan, dtype=np.float32)
    daily_target = np.full_like(daily_prediction, np.nan)
    daily_target_timestamp = np.zeros((len(unique_dates), width), dtype=np.int64)
    daily_lead = np.zeros((len(unique_dates), width), dtype=np.int16)
    daily_mask = np.ones((len(unique_dates), width), dtype=np.bool_)
    daily_rows: list[dict[str, Any]] = []
    for day_index, date in enumerate(unique_dates):
        selected = point_dates == date
        count = int(selected.sum())
        daily_prediction[day_index, :count] = point_prediction[selected]
        daily_target[day_index, :count] = point_target[selected]
        daily_target_timestamp[day_index, :count] = point_target_timestamps[selected]
        daily_lead[day_index, :count] = point_leads[selected]
        daily_mask[day_index, :count] = False
        daily_rows.append(
            {
                "date": int(date),
                "points": count,
                "ic": _pearson(point_prediction[selected], point_target[selected]),
                "rank_ic": _rank_ic(point_prediction[selected], point_target[selected]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "daily_vectors.npz",
        predictions=daily_prediction,
        targets=daily_target,
        target_mask=daily_mask,
        target_timestamps=daily_target_timestamp,
        leads=daily_lead,
        dates=unique_dates,
        lengths=lengths,
        mode=np.asarray(mode),
        lead=np.asarray(lead if mode == "fixed-lead" else 0, dtype=np.int16),
        block_size=np.asarray(block_size if mode == "blocks" else 0, dtype=np.int16),
        quantile=np.asarray(quantile, dtype=np.float64),
    )
    with (output_dir / "daily_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "date",
                "anchor_timestamp",
                "target_timestamp",
                "anchor_minute_index",
                "context_length",
                "lead",
                "prediction",
                "target",
                "error",
            )
        )
        for values in zip(
            point_dates,
            point_anchor_timestamps,
            point_target_timestamps,
            point_minute_indices,
            point_context_lengths,
            point_leads,
            point_prediction,
            point_target,
            strict=True,
        ):
            (
                date,
                anchor_ts,
                target_ts,
                minute_index,
                context_length,
                point_lead,
                prediction,
                target,
            ) = values
            writer.writerow(
                (
                    int(date),
                    int(anchor_ts),
                    int(target_ts),
                    int(minute_index),
                    int(context_length),
                    int(point_lead),
                    float(prediction),
                    float(target),
                    float(prediction - target),
                )
            )
    summary = {
        "source": str(source),
        "definition": definition,
        "mode": mode,
        "lead": lead if mode == "fixed-lead" else None,
        "block_size": block_size if mode == "blocks" else None,
        "quantile": quantile,
        "days": len(unique_dates),
        "points": len(point_target),
        "points_per_day": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        },
        "overall_ic": _pearson(point_prediction, point_target),
        "overall_rank_ic": _rank_ic(point_prediction, point_target),
        "mean_daily_ic": float(np.mean([row["ic"] for row in daily_rows if row["ic"] is not None])),
        "mean_daily_rank_ic": float(
            np.mean([row["rank_ic"] for row in daily_rows if row["rank_ic"] is not None])
        ),
        "daily": daily_rows,
    }
    summary.update(binary_direction_metrics(point_prediction, point_target))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed-lead", "blocks"), default="fixed-lead")
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--quantile", type=float, default=0.5)
    args = parser.parse_args()
    summary = build_daily_vectors(
        args.predictions,
        output_dir=args.output_dir,
        lead=args.lead,
        quantile=args.quantile,
        mode=args.mode,
        block_size=args.block_size,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "daily"}, indent=2))


if __name__ == "__main__":
    main()
