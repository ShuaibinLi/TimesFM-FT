#!/usr/bin/env python3
"""Run one real-checkpoint multivariate forward/backward/save/load gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path

import torch

from timesfm_ft import trainer
from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import ExperimentConfig, ModelConfig
from timesfm_ft.losses import PinballLoss

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument(
        "--checkpoint",
        default="../timesfm-3.0-pytorch",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    config = ExperimentConfig.from_json(config_path)
    config = dataclasses.replace(
        config,
        model=ModelConfig(
            checkpoint=str(checkpoint_path.resolve()),
            disable_linear_detrending=config.model.disable_linear_detrending,
            disable_iterative_cpm_revin=config.model.disable_iterative_cpm_revin,
        ),
        trainer=dataclasses.replace(config.trainer, dtype=args.dtype),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = trainer._dataset(
        config,
        path=config.data.train_path,
        split="train",
        dates_path=config.data.train_dates_path,
    )
    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
    )
    loader = trainer._make_loader(
        dataset,
        batch_size=args.batch_size,
        patch_length=int(model.backbone.input_patch_len),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=None,
    )
    batch = trainer._move_batch(next(iter(loader)), device)
    prediction = model(
        batch["context_values"],
        horizon=config.data.horizon_length,
        context_mask=batch["context_mask"],
        past_future_values=batch["past_future_values"],
        past_future_mask=batch["past_future_mask"],
    )
    loss = PinballLoss(model.quantiles).to(device)(
        prediction,
        batch["future_values"],
        target_mask=batch["future_mask"],
    )
    loss.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        config.trainer.max_grad_norm,
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("real checkpoint produced non-finite gradients")
    expected = model.predict(
        batch["context_values"],
        horizon=config.data.horizon_length,
        context_mask=batch["context_mask"],
        past_future_values=batch["past_future_values"],
        past_future_mask=batch["past_future_mask"],
    )
    with tempfile.TemporaryDirectory(prefix="timesfm_1min_smoke_") as temp:
        destination = Path(temp)
        model.save_adapter(destination)
        restored = TimesFM3Adapter.from_pretrained(
            config.model,
            config.adapter,
            device=device,
            dtype=config.trainer.dtype,
        )
        restored.load_adapter(destination / "adapter.pt")
        actual = restored.predict(
            batch["context_values"],
            horizon=config.data.horizon_length,
            context_mask=batch["context_mask"],
            past_future_values=batch["past_future_values"],
            past_future_mask=batch["past_future_mask"],
        )
        torch.testing.assert_close(actual, expected)
    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": model.compute_dtype,
                "context_shape": list(batch["context_values"].shape),
                "past_future_shape": list(batch["past_future_values"].shape),
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.total.detach()),
                "gradient_norm": float(gradient_norm),
                "save_load_parity": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
