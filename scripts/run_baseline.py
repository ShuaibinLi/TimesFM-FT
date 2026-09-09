#!/usr/bin/env python3
"""Fit Ridge or LightGBM on the same features and chronological split."""

from __future__ import annotations

import argparse
from pathlib import Path

from timesfm_ft.baselines import run_baseline
from timesfm_ft.config import ExperimentConfig
from timesfm_ft.trainer import _dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("ridge", "lightgbm"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    train = _dataset(
        config,
        path=config.data.train_path,
        split="train",
        dates_path=config.data.train_dates_path,
    )
    if args.split == "test":
        if config.data.test_path is None:
            parser.error("config has no test_path")
        selected_path = config.data.test_path
        dates_path = config.data.test_dates_path
    else:
        selected_path = config.data.val_path
        dates_path = config.data.val_dates_path
    selected = _dataset(
        config,
        path=selected_path,
        split=args.split,
        dates_path=dates_path,
    )
    destination = args.output_dir or (
        Path(config.trainer.output_dir) / f"{args.model}-{args.split}"
    )
    result = run_baseline(
        train,
        selected,
        model_name=args.model,
        output_dir=destination,
        horizons=config.evaluation.report_horizons,
        ridge_alpha=args.ridge_alpha,
    )
    print(result)


if __name__ == "__main__":
    main()
