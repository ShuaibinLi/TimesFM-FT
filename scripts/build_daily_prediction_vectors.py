#!/usr/bin/env python3
"""Extract one non-overlapping forecast per minute from predictions.npz."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_ft.metrics import _pearson, _rank_ic

MINUTE_NS = 60_000_000_000


def build_daily_vectors(
    source: Path,
    *,
    output_dir: Path,
    lead: int = 1,
    quantile: float = 0.5,
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
    if not 1 <= lead <= predictions.shape[1]:
        raise ValueError(f"lead must be within 1..{predictions.shape[1]}")
    quantile_index = int(np.argmin(np.abs(quantiles - quantile)))
    if abs(float(quantiles[quantile_index]) - quantile) > 1e-8:
        raise ValueError(f"requested quantile={quantile} is not present")

    lead_index = lead - 1
    valid = ~target_mask[:, lead_index]
    point_prediction = predictions[valid, lead_index, quantile_index].astype(
        np.float32,
        copy=False,
    )
    point_target = targets[valid, lead_index].astype(np.float32, copy=False)
    point_dates = dates[valid]
    point_anchor_timestamps = anchor_timestamps[valid]
    point_target_timestamps = point_anchor_timestamps + lead * MINUTE_NS
    point_minute_indices = minute_indices[valid]
    point_context_lengths = context_lengths[valid]
    order = np.lexsort((point_target_timestamps, point_dates))
    arrays = [
        point_prediction,
        point_target,
        point_dates,
        point_anchor_timestamps,
        point_target_timestamps,
        point_minute_indices,
        point_context_lengths,
    ]
    (
        point_prediction,
        point_target,
        point_dates,
        point_anchor_timestamps,
        point_target_timestamps,
        point_minute_indices,
        point_context_lengths,
    ) = (array[order] for array in arrays)

    keys = np.rec.fromarrays(
        [point_dates, point_target_timestamps],
        names=("date", "target_timestamp"),
    )
    if len(np.unique(keys)) != len(keys):
        raise ValueError("fixed-lead daily series contains duplicate target minutes")
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
    daily_mask = np.ones((len(unique_dates), width), dtype=np.bool_)
    daily_rows: list[dict[str, Any]] = []
    for day_index, date in enumerate(unique_dates):
        selected = point_dates == date
        count = int(selected.sum())
        daily_prediction[day_index, :count] = point_prediction[selected]
        daily_target[day_index, :count] = point_target[selected]
        daily_target_timestamp[day_index, :count] = point_target_timestamps[selected]
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
        dates=unique_dates,
        lengths=lengths,
        lead=np.asarray(lead, dtype=np.int16),
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
            point_prediction,
            point_target,
            strict=True,
        ):
            date, anchor_ts, target_ts, minute_index, context_length, prediction, target = values
            writer.writerow(
                (
                    int(date),
                    int(anchor_ts),
                    int(target_ts),
                    int(minute_index),
                    int(context_length),
                    float(prediction),
                    float(target),
                    float(prediction - target),
                )
            )
    summary = {
        "source": str(source),
        "definition": "one fixed-lead P50 prediction per unique target minute",
        "lead": lead,
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
        "mean_daily_ic": float(
            np.mean([row["ic"] for row in daily_rows if row["ic"] is not None])
        ),
        "mean_daily_rank_ic": float(
            np.mean([row["rank_ic"] for row in daily_rows if row["rank_ic"] is not None])
        ),
        "daily": daily_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.5)
    args = parser.parse_args()
    summary = build_daily_vectors(
        args.predictions,
        output_dir=args.output_dir,
        lead=args.lead,
        quantile=args.quantile,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "daily"}, indent=2))


if __name__ == "__main__":
    main()
