#!/usr/bin/env python3
"""Evaluate weighted-mid-price up/down forecasts from saved lead1 points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from timesfm_ft.metrics import binary_direction_metrics


def _load_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[float] = []
    targets: list[float] = []
    dates: list[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"prediction", "target", "date"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            predictions.append(float(row["prediction"]))
            targets.append(float(row["target"]))
            dates.append(int(row["date"]))
    if not predictions:
        raise ValueError(f"{path} contains no prediction rows")
    return (
        np.asarray(predictions, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(dates, dtype=np.int32),
    )


def evaluate_direction_points(
    source: Path,
    *,
    output_dir: Path,
    coverage_levels: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05),
) -> dict[str, Any]:
    prediction, target, dates = _load_points(source)
    if np.any(target == 0.0):
        zero_target_points = int(np.sum(target == 0.0))
    else:
        zero_target_points = 0
    overall = binary_direction_metrics(prediction, target)
    daily = []
    for date in np.unique(dates):
        selected = dates == date
        metrics = binary_direction_metrics(prediction[selected], target[selected])
        daily.append({"date": int(date), **metrics})

    order = np.argsort(-np.abs(prediction), kind="stable")
    coverage_rows: list[dict[str, Any]] = []
    for requested_coverage in coverage_levels:
        if not 0 < requested_coverage <= 1:
            raise ValueError("coverage levels must be in (0, 1]")
        count = max(1, int(np.ceil(len(order) * requested_coverage)))
        selected = order[:count]
        threshold = float(np.min(np.abs(prediction[selected])))
        coverage_rows.append(
            {
                "requested_coverage": requested_coverage,
                "points": count,
                "actual_coverage": count / len(order),
                "absolute_prediction_threshold": threshold,
                **binary_direction_metrics(prediction[selected], target[selected]),
            }
        )

    daily_accuracy = [
        row["direction_accuracy"] for row in daily if row["direction_accuracy"] is not None
    ]
    daily_balanced = [
        row["direction_balanced_accuracy"]
        for row in daily
        if row["direction_balanced_accuracy"] is not None
    ]
    summary = {
        "source": str(source),
        "definition": (
            "sign(P50 predicted 1min return) versus sign(realized 1min weighted-mid-price delta)"
        ),
        "target_scale_note": (
            "return_1m is weighted-mid-price delta divided by positive tick size, "
            "so its sign is unchanged"
        ),
        "zero_target_policy": "exclude unchanged target points from binary up/down metrics",
        "points": int(len(target)),
        "days": int(len(daily)),
        "zero_target_points": zero_target_points,
        **overall,
        "mean_daily_direction_accuracy": (
            float(np.mean(daily_accuracy)) if daily_accuracy else None
        ),
        "mean_daily_direction_balanced_accuracy": (
            float(np.mean(daily_balanced)) if daily_balanced else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for filename, rows in (
        ("daily_direction.csv", daily),
        ("direction_by_coverage.csv", coverage_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_direction_points(args.points, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
