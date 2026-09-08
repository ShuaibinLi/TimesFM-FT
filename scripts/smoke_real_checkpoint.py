#!/usr/bin/env python3
"""Run one real TimesFM precision, train, checkpoint, and resume gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from timesfm_ft import trainer
from timesfm_ft.adapter import TimesFM3Adapter
from timesfm_ft.config import ExperimentConfig, ModelConfig
from timesfm_ft.data import NpzWindowDataset
from timesfm_ft.losses import ForecastLoss

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/zn_single_input.json")
    parser.add_argument("--checkpoint", default="../timesfm-3.0-pytorch")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    checkpoint_path = Path(args.checkpoint)
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
        trainer=dataclasses.replace(
            config.trainer,
            dtype=args.dtype or config.trainer.dtype,
            deterministic=(
                config.trainer.deterministic
                if args.deterministic is None
                else args.deterministic
            ),
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.set_seed(config.trainer.seed, deterministic=config.trainer.deterministic)
    expected_dates = trainer._read_expected_dates(config.data.train_dates_path)
    dataset = NpzWindowDataset(
        config.data.train_path,
        context_length=config.data.context_length,
        horizon_length=config.data.horizon_length,
        max_variates=1,
        sampling_interval_seconds=config.data.sampling_interval_seconds,
        expected_stride=config.data.stride,
        expected_product=config.data.product,
        expected_split="train",
        expected_dates=expected_dates,
        expected_dates_path=config.data.train_dates_path,
        require_metadata=True,
    )
    batch_size = args.batch_size or config.trainer.batch_size
    if batch_size <= 0:
        parser.error("--batch-size must be positive")
    indices = (
        torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(config.trainer.seed),
        )[:batch_size].tolist()
        if args.shuffle
        else list(range(batch_size))
    )
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
            )
        )
    )
    batch = trainer._move_batch(batch, device)
    model = TimesFM3Adapter.from_pretrained(
        config.model,
        config.adapter,
        device=device,
        dtype=config.trainer.dtype,
    )
    if {parameter.dtype for parameter in model.parameters()} != {torch.float32}:
        raise AssertionError("model does not retain FP32 master parameters")
    loss_fn = ForecastLoss(
        model.quantiles,
        tick_size=config.objective.tick_size,
        pinball_weight=config.objective.pinball_weight,
        median_huber_weight=config.objective.median_huber_weight,
        crossing_weight=config.objective.crossing_weight,
        huber_delta_ticks=config.objective.huber_delta_ticks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(config.optimizer)
    )
    scheduler = trainer.make_scheduler(
        optimizer,
        total_steps=2,
        warmup_ratio=0.0,
        min_lr_ratio=0.1,
    )
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    prediction = model(
        batch["context_values"],
        horizon=config.data.horizon_length,
        context_mask=batch["context_mask"],
    )
    losses = loss_fn(
        prediction,
        batch["future_values"],
        current_price=batch["context_values"][:, 0, -1],
        target_mask=batch["future_mask"],
    )
    losses.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        config.trainer.max_grad_norm,
    )
    if not torch.isfinite(gradient_norm):
        bad_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        raise FloatingPointError(
            "real-checkpoint smoke produced non-finite gradients: "
            f"{bad_gradients[:10]}"
        )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    state_dtypes = {
        value.dtype
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if state_dtypes != {torch.float32}:
        raise AssertionError(f"optimizer state is not FP32: {state_dtypes}")

    model.eval()
    expected = model.predict(
        batch["context_values"],
        horizon=config.data.horizon_length,
        context_mask=batch["context_mask"],
    )
    data_metadata = {"train": dataset.metadata, "val": dataset.metadata}
    generator = torch.Generator().manual_seed(config.trainer.seed)
    with tempfile.TemporaryDirectory(prefix="timesfm_real_smoke_") as temp:
        root = Path(temp)
        model.save_adapter(root / "adapter")
        training_state = trainer._training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_metric=1.0,
            stale_epochs=0,
            history=[],
            config=config,
            generator=generator,
            data_metadata=data_metadata,
        )
        trainer._atomic_torch_save(
            training_state,
            root / "last" / "training_state.pt",
        )

        restored = TimesFM3Adapter.from_pretrained(
            config.model,
            config.adapter,
            device=device,
            dtype=config.trainer.dtype,
        )
        restored.load_adapter(root / "adapter" / "adapter.pt")
        restored_optimizer = torch.optim.AdamW(
            restored.optimizer_parameter_groups(config.optimizer)
        )
        restored_scheduler = trainer.make_scheduler(
            restored_optimizer,
            total_steps=2,
            warmup_ratio=0.0,
            min_lr_ratio=0.1,
        )
        restored_generator = torch.Generator().manual_seed(0)
        start_epoch, best_metric, _, _ = trainer._load_training_state(
            root / "last",
            model=restored,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            config=config,
            generator=restored_generator,
            data_metadata=data_metadata,
        )
        restored.eval()
        actual = restored.predict(
            batch["context_values"],
            horizon=config.data.horizon_length,
            context_mask=batch["context_mask"],
        )
        torch.testing.assert_close(actual, expected)
        if start_epoch != 2 or best_metric != 1.0:
            raise AssertionError("resume metadata was not restored")

    print(
        json.dumps(
            {
                "device": str(device),
                "compute_dtype": model.compute_dtype,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "loss_dtype": str(losses.total.dtype),
                "loss": float(losses.total.detach()),
                "pre_clip_gradient_norm": float(gradient_norm),
                "optimizer_state_dtypes": sorted(str(value) for value in state_dtypes),
                "prediction_shape": list(expected.shape),
                "resume_parity": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
