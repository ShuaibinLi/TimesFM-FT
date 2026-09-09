#!/usr/bin/env python3
"""Select low-redundancy past-only features using training dates only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    left = left.astype(np.float64, copy=True)
    right = right.astype(np.float64, copy=True)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _rank_ic(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _load_training_rows(
    bundle: Path,
    *,
    context_min: int,
    horizons: tuple[int, ...],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("split") != "train":
        raise ValueError("feature selection is allowed only on split=train")
    target = np.load(bundle / "target_values.npy", mmap_mode="r")
    target_mask = np.load(bundle / "target_mask.npy", mmap_mode="r")
    features = np.load(bundle / "past_only_values.npy", mmap_mode="r")
    masks = np.load(bundle / "past_only_mask.npy", mmap_mode="r")
    lengths = np.load(bundle / "session_lengths.npy", mmap_mode="r")
    rows: list[np.ndarray] = []
    row_masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    max_horizon = max(horizons)
    for day, length_value in enumerate(lengths):
        length = int(length_value)
        for anchor in range(context_min - 1, length - max_horizon, stride):
            future_slice = slice(anchor + 1, anchor + 1 + max_horizon)
            if target_mask[day, anchor] or target_mask[day, future_slice].any():
                continue
            rows.append(features[day, :, anchor])
            row_masks.append(masks[day, :, anchor])
            future = target[day, future_slice]
            cumulative = np.cumsum(future, dtype=np.float64)
            labels.append(np.asarray([cumulative[horizon - 1] for horizon in horizons]))
    if not rows:
        raise ValueError("training bundle has no fully valid feature-selection rows")
    return np.stack(rows), np.stack(row_masks), np.stack(labels), manifest


def select_features(
    bundle: Path,
    *,
    output: Path,
    context_min: int,
    horizons: tuple[int, ...],
    stride: int,
    limit: int,
    correlation_limit: float,
    family_limit: int,
) -> None:
    values, masks, labels, manifest = _load_training_rows(
        bundle,
        context_min=context_min,
        horizons=horizons,
        stride=stride,
    )
    names = tuple(manifest["past_only_features"])
    families = manifest.get("past_only_families", {})
    reports: list[dict] = []
    for index, name in enumerate(names):
        valid = ~masks[:, index] & np.isfinite(values[:, index])
        horizon_metrics = {}
        scores: list[float] = []
        for horizon_index, horizon in enumerate(horizons):
            ic = _pearson(values[valid, index], labels[valid, horizon_index])
            rank_ic = _rank_ic(values[valid, index], labels[valid, horizon_index])
            horizon_metrics[str(horizon)] = {"ic": ic, "rank_ic": rank_ic}
            if rank_ic is not None:
                scores.append(abs(rank_ic))
        reports.append(
            {
                "name": name,
                "family": families.get(name, "unknown"),
                "valid_rows": int(valid.sum()),
                "missing_rate": float(1.0 - valid.mean()),
                "score": max(scores, default=0.0),
                "horizons": horizon_metrics,
                "index": index,
            }
        )

    selected: list[dict] = []
    family_counts: dict[str, int] = {}
    for candidate in sorted(reports, key=lambda row: (-row["score"], row["name"])):
        family = candidate["family"]
        if family_counts.get(family, 0) >= family_limit:
            continue
        candidate_index = candidate["index"]
        redundant = False
        for existing in selected:
            existing_index = existing["index"]
            valid = (
                ~masks[:, candidate_index]
                & ~masks[:, existing_index]
                & np.isfinite(values[:, candidate_index])
                & np.isfinite(values[:, existing_index])
            )
            correlation = _pearson(
                values[valid, candidate_index],
                values[valid, existing_index],
            )
            if correlation is not None and abs(correlation) >= correlation_limit:
                redundant = True
                break
        if redundant:
            continue
        selected.append(candidate)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) == limit:
            break
    for report in reports:
        report.pop("index")
    result = {
        "source_bundle": str(bundle),
        "split": "train",
        "context_min": context_min,
        "horizons": horizons,
        "stride": stride,
        "correlation_limit": correlation_limit,
        "family_limit": family_limit,
        "selected_features": [row["name"] for row in selected],
        "selected_detail": selected,
        "all_features": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-min", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 15, 30, 60))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--correlation-limit", type=float, default=0.9)
    parser.add_argument("--family-limit", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.correlation_limit <= 1:
        parser.error("--correlation-limit must be in (0, 1]")
    if min(args.context_min, args.stride, args.limit, args.family_limit) <= 0:
        parser.error("count arguments must be positive")
    select_features(
        args.bundle,
        output=args.output,
        context_min=args.context_min,
        horizons=tuple(sorted(set(args.horizons))),
        stride=args.stride,
        limit=args.limit,
        correlation_limit=args.correlation_limit,
        family_limit=args.family_limit,
    )


if __name__ == "__main__":
    main()
