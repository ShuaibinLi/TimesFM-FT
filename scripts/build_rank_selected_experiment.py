#!/usr/bin/env python3
"""Build a TimesFM experiment config from train-only feature selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "outputs" / "feature-selection" / "zn-rank-selected100.json"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "experiments" / "zn_rank_e2_selected20_tod.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("split") != "train":
        raise ValueError("experiment features must come from split=train")
    features = selection["selected_features"]
    if not features or len(features) > 28 or len(features) != len(set(features)):
        raise ValueError("selected past-only features must be 1..28 unique names")

    config = {
        "extends": "_base.json",
        "data": {
            "train_path": "../../data/zn-rank-selected100-1min/train",
            "val_path": "../../data/zn-rank-selected100-1min/val",
            "test_path": "../../data/zn-rank-selected100-1min/test",
            "dataset_id": "zn_rank_selected100_1min_wmid_ticks_v1",
            "product": "ZN",
            "target_name": "return_1m",
            "target_unit": "ZN_ticks",
            "target_price_source": "WMid",
            "target_return_type": "tick_displacement",
            "target_timestamp_semantics": "bar_end",
            "target_availability_lag_minutes": 0,
            "target_missing_policy": "mask",
            "past_only_features": features,
            "past_future_features": [
                "sin_time_of_day",
                "cos_time_of_day",
                "time_to_close",
            ],
            "train_dates_path": "../splits/zn-rank-selected100/dates-train.txt",
            "val_dates_path": "../splits/zn-rank-selected100/dates-val.txt",
            "test_dates_path": "../splits/zn-rank-selected100/dates-test.txt",
        },
        "trainer": {
            "output_dir": "../../outputs/zn-rank-selected100-e2",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}: {len(features)} past-only + 3 past-future")


if __name__ == "__main__":
    main()
