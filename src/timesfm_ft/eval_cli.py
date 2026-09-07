"""Evaluation command-line entry point."""

from __future__ import annotations

import argparse
import logging

from timesfm_ft.config import ExperimentConfig
from timesfm_ft.evaluator import evaluate_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timesfm-eval",
        description="Evaluate zero-shot or fine-tuned TimesFM 3 forecasts.",
    )
    parser.add_argument("--config", required=True, help="Experiment JSON path.")
    parser.add_argument(
        "--adapter",
        help="Path to adapter.pt. Omit for official zero-shot evaluation.",
    )
    parser.add_argument(
        "--data",
        help="Evaluation NPZ path. Defaults to data.val_path from the config.",
    )
    parser.add_argument(
        "--output-dir",
        help="Metric output directory. Defaults to trainer.output_dir/evaluation.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", help="Override trainer.device.")
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
    destination = evaluate_experiment(
        config,
        data_path=args.data,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    logging.getLogger(__name__).info("artifacts=%s", destination)


if __name__ == "__main__":
    main()
