#!/usr/bin/env python3
"""Generate the TimesFM dataset schema from the frozen selected100 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT.parent / "datas" / "wenyao_features" / "zn_rank_selected100_1min_manifest.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "datasets" / "zn_rank_selected100_1min.json"


def build_schema(manifest_path: Path) -> dict:
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    features = manifest["selection"]["features"]
    names = [entry["column"] for entry in features]
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError("selected100 manifest must contain 100 unique features")
    target = manifest["target_contract"]
    if target["unit"] != "ZN ticks":
        raise ValueError(f"unexpected target unit={target['unit']!r}")

    return {
        "dataset_id": "zn_rank_selected100_1min_wmid_ticks_v1",
        "product": "ZN",
        "path_template": "date={date}/*.parquet",
        "timestamp_column": "timestamp_ns",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "target": {
            "name": "return_1m",
            "column": "return_1m",
            "unit": "ZN_ticks",
            "price_source": "WMid",
            "return_type": "tick_displacement",
            "timestamp_semantics": "bar_end",
            "availability_lag_minutes": 0,
            "missing_policy": "mask",
            "derivation": {
                "kind": "trailing_price_difference_ticks",
                "price_column": "wmid",
                "tick_size": 0.015625,
                "timestamp_source_column": "hwts",
                "interval_minutes": 1,
            },
        },
        "session": {
            "timezone": "America/New_York",
            "first_bar_time": "09:31:00",
            "minutes": 390,
        },
        "past_only_features": [
            {
                "name": entry["column"],
                "column": entry["column"],
                "family": entry["family"],
                "availability_lag_minutes": 0,
            }
            for entry in features
        ],
        "past_future_features": [
            {"name": "sin_time_of_day", "kind": "sin_time_of_day"},
            {"name": "cos_time_of_day", "kind": "cos_time_of_day"},
            {"name": "time_to_close", "kind": "time_to_close"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    schema = build_schema(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}: {len(schema['past_only_features'])} candidates")


if __name__ == "__main__":
    main()
