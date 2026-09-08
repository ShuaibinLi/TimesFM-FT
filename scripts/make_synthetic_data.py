#!/usr/bin/env python3
"""Creates aligned single-input and multi-input NPZ files for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_samples(
    *,
    num_samples: int,
    context_length: int,
    horizon_length: int,
    num_variates: int,
    tick_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total_length = context_length + horizon_length
    contexts = np.zeros(
        (num_samples, num_variates, context_length), dtype=np.float32
    )
    futures = np.zeros((num_samples, horizon_length), dtype=np.float32)

    for sample_index in range(num_samples):
        innovations = rng.normal(0.0, 0.08, total_length)
        returns = np.zeros(total_length, dtype=np.float64)
        for step in range(1, total_length):
            returns[step] = 0.2 * returns[step - 1] + innovations[step]
        price = 100.0 + tick_size * np.cumsum(returns)
        context_returns = returns[:context_length]

        contexts[sample_index, 0] = price[:context_length]
        feature_bank = [
            context_returns,
            np.tanh(context_returns * 4.0),
            np.abs(context_returns),
            rng.lognormal(4.0, 0.4, context_length),
            np.sign(context_returns) * rng.lognormal(3.0, 0.5, context_length),
            rng.poisson(8.0, context_length),
            1.0 + np.abs(context_returns) * 2.0,
        ]
        for channel in range(1, num_variates):
            contexts[sample_index, channel] = feature_bank[(channel - 1) % len(feature_bank)]
        futures[sample_index] = price[context_length:]

    return contexts, futures


def save_split(
    root: Path,
    contexts: np.ndarray,
    futures: np.ndarray,
    train_count: int,
    val_count: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "train.npz",
        context_values=contexts[:train_count],
        future_values=futures[:train_count],
    )
    np.savez_compressed(
        root / "val.npz",
        context_values=contexts[train_count : train_count + val_count],
        future_values=futures[train_count : train_count + val_count],
    )
    np.savez_compressed(
        root / "test.npz",
        context_values=contexts[train_count + val_count :],
        future_values=futures[train_count + val_count :],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon-length", type=int, default=64)
    parser.add_argument("--num-variates", type=int, default=8)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_samples < 4:
        raise ValueError("num-samples must be at least 4")
    if not 2 <= args.num_variates <= 32:
        raise ValueError("num-variates must be in [2, 32]")

    multi_contexts, futures = make_samples(
        num_samples=args.num_samples,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        num_variates=args.num_variates,
        tick_size=args.tick_size,
        seed=args.seed,
    )
    train_count = max(1, round(args.num_samples * 0.8))
    val_count = max(1, round(args.num_samples * 0.1))
    train_count = min(train_count, args.num_samples - 2)
    val_count = min(val_count, args.num_samples - train_count - 1)
    test_count = args.num_samples - train_count - val_count
    output_root = Path(args.output_root)
    save_split(output_root / "multi", multi_contexts, futures, train_count, val_count)
    save_split(
        output_root / "single",
        multi_contexts[:, :1, :],
        futures,
        train_count,
        val_count,
    )
    print(
        f"saved {train_count} train, {val_count} validation, and {test_count} test "
        f"samples under {output_root}"
    )


if __name__ == "__main__":
    main()
