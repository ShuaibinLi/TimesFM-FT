"""Command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

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
    parser.add_argument(
        "--resume",
        help="Resume from a best/last directory or training_state.pt.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = ExperimentConfig.from_json(args.config)
    if args.resume is not None:
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(
                config.trainer,
                resume_from=str(Path(args.resume).resolve()),
            ),
        )
    output_dir = train_experiment(config)
    logging.getLogger(__name__).info("artifacts=%s", output_dir)


if __name__ == "__main__":
    main()
