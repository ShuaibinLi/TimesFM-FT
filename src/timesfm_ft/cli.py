"""Command-line entry point."""

from __future__ import annotations

import argparse

from timesfm_ft.config import ExperimentConfig
from timesfm_ft.trainer import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timesfm-ft",
        description="Fine-tune TimesFM 3 on pre-windowed weighted-mid data.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment JSON file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_json(args.config)
    output_dir = train_experiment(config)
    print(f"artifacts={output_dir}")


if __name__ == "__main__":
    main()
