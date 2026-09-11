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
from timesfm_ft.dense import build_dense_training_batch, gather_final_anchor
from timesfm_ft.losses import BusinessForecastLoss

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/zn_rank_e2_pilot.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="../timesfm-3.0-pytorch",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--minimum-context", type=int, default=0)
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
        for_training=True,
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
    raw_batch = next(
        (
            candidate
            for candidate in loader
            if int(candidate["context_lengths"].max()) >= args.minimum_context
        ),
        None,
    )
    if raw_batch is None:
        raise ValueError(f"no smoke batch has context >= {args.minimum_context}")
    batch = trainer._move_batch(raw_batch, device)
    model_kwargs = {
        "horizon": config.data.horizon_length,
        "context_mask": batch["context_mask"],
        "context_padding_mask": batch["context_padding_mask"],
        "past_future_values": batch["past_future_values"],
        "past_future_mask": batch["past_future_mask"],
    }
    auxiliary_indices = torch.tensor(
        [
            config.data.past_only_features.index(feature)
            for feature in config.objective.auxiliary_features
        ],
        dtype=torch.long,
        device=device,
    )
    auxiliary_prediction = None
    auxiliary_target = None
    auxiliary_mask = None
    auxiliary_anchor_mask = None
    anchor_mask = None
    final_prediction = None
    business_target = batch["unknown_future_values"][:, 0]
    business_target_mask = batch["unknown_future_mask"][:, 0]
    eligible_dense_anchors = None
    if config.objective.uses_dense_forward:
        dense_batch = build_dense_training_batch(
            batch,
            context_min=config.data.context_min,
            input_patch_length=int(model.backbone.input_patch_len),
            output_patch_length=int(model.backbone.output_patch_len),
        )
        dense_prediction = model.forward_dense(
            dense_batch.values,
            masks=dense_batch.masks,
            patch_is_target=dense_batch.patch_is_target,
            unknown_variates=batch["context_values"].shape[1],
        )
        prediction = dense_prediction.target
        target = dense_batch.target_labels
        target_mask = dense_batch.target_label_mask
        anchor_mask = dense_batch.eligible_anchor_mask
        eligible_dense_anchors = int(anchor_mask.sum().item())
        final_prediction = gather_final_anchor(
            dense_prediction.target,
            dense_batch.final_anchor_indices,
        )
        if auxiliary_indices.numel():
            auxiliary_prediction = dense_prediction.past_only.index_select(1, auxiliary_indices)
            auxiliary_target = dense_batch.past_only_labels.index_select(1, auxiliary_indices)
            auxiliary_mask = dense_batch.past_only_label_mask.index_select(1, auxiliary_indices)
            auxiliary_anchor_mask = dense_batch.eligible_anchor_mask
    elif auxiliary_indices.numel():
        unknown_prediction = model.forward_unknown(batch["context_values"], **model_kwargs)
        prediction = unknown_prediction.target
        target = business_target
        target_mask = business_target_mask
        auxiliary_prediction = unknown_prediction.past_only.index_select(1, auxiliary_indices)
        auxiliary_target = batch["unknown_future_values"][:, 1:].index_select(1, auxiliary_indices)
        auxiliary_mask = batch["unknown_future_mask"][:, 1:].index_select(1, auxiliary_indices)
    else:
        prediction = model(batch["context_values"], **model_kwargs)
        target = business_target
        target_mask = business_target_mask
    loss_scales = trainer.fit_loss_scales(
        dataset,
        config.objective,
        input_patch_length=int(model.backbone.input_patch_len),
        output_patch_length=int(model.backbone.output_patch_len),
        context_min=config.data.context_min,
        batch_size=args.batch_size,
    )
    loss = BusinessForecastLoss(
        model.quantiles,
        objective=config.objective,
        scales=loss_scales,
    ).to(device)(
        prediction,
        target,
        target_mask=target_mask,
        anchor_mask=anchor_mask,
        final_predictions=final_prediction,
        final_targets=business_target,
        final_target_mask=business_target_mask,
        auxiliary_predictions=auxiliary_prediction,
        auxiliary_targets=auxiliary_target,
        auxiliary_mask=auxiliary_mask,
        auxiliary_anchor_mask=auxiliary_anchor_mask,
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
        context_padding_mask=batch["context_padding_mask"],
        past_future_values=batch["past_future_values"],
        past_future_mask=batch["past_future_mask"],
    )
    if final_prediction is not None:
        torch.testing.assert_close(final_prediction, expected)
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
            context_padding_mask=batch["context_padding_mask"],
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
                "context_lengths": batch["context_lengths"].tolist(),
                "eligible_dense_anchors": eligible_dense_anchors,
                "past_future_shape": list(batch["past_future_values"].shape),
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.total.detach()),
                "return_pinball": float(loss.return_pinball.detach()),
                "lead1_pinball": float(loss.lead1_pinball.detach()),
                "correlation": float(loss.correlation.detach()),
                "cumulative_huber": float(loss.cumulative_huber.detach()),
                "auxiliary_pinball": float(loss.auxiliary_pinball.detach()),
                "loss_scale_fingerprint": loss_scales.fingerprint,
                "gradient_norm": float(gradient_norm),
                "save_load_parity": True,
                "dense_final_decode_parity": (True if final_prediction is not None else None),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
