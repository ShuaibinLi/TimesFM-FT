#!/usr/bin/env python3
"""Freeze chronological train/validation/test dates for selected100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent / "spine" / "configs" / "wmp_500ms_dates_20221101_20260130.txt"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "splits" / "zn-rank-selected100"
BOUNDARIES = {
    "train": (20221101, 20240930),
    "val": (20241001, 20250731),
    "test": (20250801, 20260130),
}
EXPECTED_COUNTS = {"train": 476, "val": 209, "test": 128}


def _dates(path: Path) -> list[int]:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(values) != len(set(values)) or values != sorted(values):
        raise ValueError("source dates must be unique and chronological")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_dates = _dates(args.source)
    source_set = set(source_dates)
    splits = {
        name: [date_value for date_value in source_dates if start <= date_value <= end]
        for name, (start, end) in BOUNDARIES.items()
    }
    actual_counts = {name: len(values) for name, values in splits.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"split count drift: actual={actual_counts}, expected={EXPECTED_COUNTS}")
    assigned = set().union(*(set(values) for values in splits.values()))
    if assigned != source_set:
        raise ValueError("fixed-boundary splits must partition source dates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (args.output_dir / f"dates-{name}.txt").write_text(
            "".join(f"{value}\n" for value in values),
            encoding="utf-8",
        )
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_dates": len(source_dates),
        "excluded_dates": [],
        "policy": "fixed_date_boundaries_without_event_day_exclusions",
        "splits": {
            name: {
                "dates": len(values),
                "first": values[0],
                "last": values[-1],
                "boundary_start": BOUNDARIES[name][0],
                "boundary_end": BOUNDARIES[name][1],
            }
            for name, values in splits.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
