#!/usr/bin/env python3
"""Collect all available weighted-mid up/down evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DIRECTION_KEYS = (
    "directional_accuracy",
    "direction_accuracy",
    "direction_balanced_accuracy",
    "direction_roc_auc",
    "direction_up_precision",
    "direction_up_accuracy",
    "direction_up_recall",
    "direction_down_precision",
    "direction_down_accuracy",
    "direction_down_recall",
    "direction_realized_signed_delta_mean",
)


def _direction_row(
    *,
    artifact_type: str,
    path: Path,
    metrics: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    row = {
        "artifact_type": artifact_type,
        "path": str(path),
        "horizon_minutes": horizon,
        "points": metrics.get("direction_nonzero_points", metrics.get("samples")),
        "overall_ic": metrics.get("ic", metrics.get("overall_ic")),
        "overall_rank_ic": metrics.get("rank_ic", metrics.get("overall_rank_ic")),
    }
    row.update({key: metrics.get(key) for key in DIRECTION_KEYS})
    if row["direction_accuracy"] is None:
        row["direction_accuracy"] = row["directional_accuracy"]
    return row


def collect_results(root: Path, *, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        horizon = next(
            (
                item
                for item in payload.get("cumulative_horizons", [])
                if item.get("horizon_minutes") == 1
            ),
            None,
        )
        if horizon is not None:
            rows.append(
                _direction_row(
                    artifact_type="checkpoint_h1_aggregate",
                    path=path,
                    metrics=horizon,
                    horizon=1,
                )
            )

    for path in sorted(root.glob("**/adapter_config.json")):
        if path.parent.name not in {"best", "last"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scorecard = payload.get("validation_scorecard", {})
        horizon = next(
            (
                item
                for item in scorecard.get("cumulative_horizons", [])
                if item.get("horizon_minutes") == 1
            ),
            None,
        )
        if horizon is not None:
            rows.append(
                _direction_row(
                    artifact_type=f"checkpoint_epoch_{path.parent.name}",
                    path=path,
                    metrics=horizon,
                    horizon=1,
                )
            )

    for path in sorted(root.glob("**/wmid-direction*/summary.json")):
        if output_dir.resolve() in path.resolve().parents:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            _direction_row(
                artifact_type="saved_points_full_direction",
                path=path,
                metrics=payload,
                horizon=1,
            )
        )

    for path in sorted(root.glob("**/ridge-test/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        horizon = next(
            (item for item in payload.get("horizons", []) if item.get("horizon_minutes") == 1),
            None,
        )
        if horizon is not None:
            rows.append(
                _direction_row(
                    artifact_type="ridge_h1_aggregate",
                    path=path,
                    metrics=horizon,
                    horizon=1,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "artifact_type",
        "path",
        "horizon_minutes",
        "points",
        "overall_ic",
        "overall_rank_ic",
        *DIRECTION_KEYS,
    )
    with (output_dir / "all_direction_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "root": str(root),
        "results": len(rows),
        "full_direction_results": sum(
            row["artifact_type"] == "saved_points_full_direction" for row in rows
        ),
        "aggregate_only_results": sum(
            row["artifact_type"] != "saved_points_full_direction" for row in rows
        ),
        "note": (
            "aggregate-only historical checkpoints retain accuracy but require inference "
            "to recover confusion matrices, AUC, and confidence-coverage curves"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wmid-direction-index"),
    )
    args = parser.parse_args()
    rows = collect_results(args.root, output_dir=args.output_dir)
    print(f"collected {len(rows)} direction results into {args.output_dir}")


if __name__ == "__main__":
    main()
