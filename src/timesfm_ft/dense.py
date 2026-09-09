"""Explicit full-sequence tensors and shifted labels for dense-anchor training."""

from __future__ import annotations

import dataclasses

import torch

from timesfm_ft.data import WindowBatch


@dataclasses.dataclass(frozen=True)
class DenseTrainingBatch:
    values: torch.Tensor
    masks: torch.Tensor
    patch_is_target: torch.Tensor
    target_labels: torch.Tensor
    target_label_mask: torch.Tensor
    past_only_labels: torch.Tensor
    past_only_label_mask: torch.Tensor
    eligible_anchor_mask: torch.Tensor
    final_anchor_indices: torch.Tensor
    anchor_timestamps: torch.Tensor
    context_patch_count: int


def shifted_output_patches(
    values: torch.Tensor,
    rolls: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns token-j labels from patches j+1..j+rolls plus wrap mask."""

    if values.ndim != 4:
        raise ValueError("values must have shape (batch, variates, patches, patch)")
    if rolls <= 0:
        raise ValueError("rolls must be positive")
    batch, variates, patches, patch_length = values.shape
    outputs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for offset in range(1, rolls + 1):
        rolled = torch.roll(values, shifts=-offset, dims=2)
        wrap = torch.zeros(
            1,
            1,
            patches,
            patch_length,
            dtype=torch.bool,
            device=values.device,
        )
        wrap[:, :, patches - min(offset, patches) :, :] = True
        outputs.append(rolled)
        masks.append(wrap.expand(batch, variates, -1, -1))
    return torch.cat(outputs, dim=-1), torch.cat(masks, dim=-1)


def build_dense_training_batch(
    batch: WindowBatch,
    *,
    context_min: int,
    input_patch_length: int,
    output_patch_length: int,
) -> DenseTrainingBatch:
    """Builds the confirmed full-sequence forward contract for T1/T2/T3."""

    context = batch["context_values"]
    context_mask = batch["context_mask"].bool()
    unknown_future = batch["unknown_future_values"]
    unknown_source_mask = batch["unknown_future_mask"].bool()
    known = batch["past_future_values"]
    known_mask = batch["past_future_mask"].bool()
    if context.ndim != 3 or context_mask.shape != context.shape:
        raise ValueError("context values/mask must have shape (B, 1+Vpo, C)")
    batch_size, unknown_variates, context_width = context.shape
    horizon = unknown_future.shape[-1]
    if horizon != output_patch_length:
        raise ValueError("dense v1 requires horizon_length == model output_patch_length")
    if output_patch_length % input_patch_length:
        raise ValueError("output patch must be a multiple of input patch")
    if context_width % input_patch_length:
        raise ValueError("collated context must be patch aligned")
    expected_unknown_shape = (batch_size, unknown_variates, horizon)
    if (
        unknown_future.shape != expected_unknown_shape
        or unknown_source_mask.shape != expected_unknown_shape
    ):
        raise ValueError("unknown future rows must align with [target | past-only] context")
    if known.shape[:2] != known_mask.shape[:2] or known.shape != known_mask.shape:
        raise ValueError("past-future values/mask shape mismatch")
    if known.shape[-1] != context_width + horizon:
        raise ValueError("past-future width must equal context plus horizon")

    unknown_values = torch.cat((context, unknown_future), dim=-1)
    unknown_label_mask = torch.cat((context_mask, unknown_source_mask), dim=-1)

    # Target horizon values are real full-sequence observations. Past-only
    # horizon values are retained only as labels and are hidden as model inputs.
    unknown_input_mask = unknown_label_mask.clone()
    if unknown_variates > 1:
        unknown_input_mask[:, 1:, context_width:] = True
    values = torch.cat((unknown_values, known), dim=1)
    masks = torch.cat((unknown_input_mask, known_mask), dim=1)
    total_width = values.shape[-1]
    if total_width % input_patch_length:
        raise ValueError("full dense sequence must be patch aligned")
    total_patches = total_width // input_patch_length
    context_patches = context_width // input_patch_length
    values_patched = values.reshape(
        batch_size,
        values.shape[1],
        total_patches,
        input_patch_length,
    )
    masks_patched = masks.reshape_as(values_patched)
    unknown_values_patched = unknown_values.reshape(
        batch_size,
        unknown_variates,
        total_patches,
        input_patch_length,
    )
    unknown_label_mask_patched = unknown_label_mask.reshape_as(unknown_values_patched)
    rolls = output_patch_length // input_patch_length
    unknown_labels, wrap_mask = shifted_output_patches(
        unknown_values_patched,
        rolls,
    )
    shifted_masks, _ = shifted_output_patches(
        unknown_label_mask_patched,
        rolls,
    )
    unknown_label_masks = shifted_masks | wrap_mask

    patch_is_target = torch.zeros(
        batch_size,
        values.shape[1],
        total_patches,
        dtype=torch.bool,
        device=values.device,
    )
    patch_is_target[:, :unknown_variates, :] = True

    visible_history = (
        (
            ~context_mask[:, 0].reshape(
                batch_size,
                context_patches,
                input_patch_length,
            )
        )
        .sum(dim=-1)
        .cumsum(dim=1)
    )
    target_patch_mask = unknown_label_mask_patched[:, 0]
    anchor_value_valid = ~target_patch_mask[:, :context_patches, -1]
    target_path_valid = ~unknown_label_masks[:, 0, :context_patches].any(dim=-1)
    eligible_context = (visible_history >= context_min) & anchor_value_valid & target_path_valid
    eligible = torch.zeros(
        batch_size,
        total_patches,
        dtype=torch.bool,
        device=values.device,
    )
    eligible[:, :context_patches] = eligible_context
    final_anchor_indices = torch.full(
        (batch_size,),
        context_patches - 1,
        dtype=torch.long,
        device=values.device,
    )
    patch_offsets = (
        torch.arange(total_patches, device=values.device) - (context_patches - 1)
    ) * input_patch_length
    anchor_timestamps = batch["timestamps"][:, None] + patch_offsets[None, :] * 60_000_000_000
    return DenseTrainingBatch(
        values=values_patched,
        masks=masks_patched,
        patch_is_target=patch_is_target,
        target_labels=unknown_labels[:, 0],
        target_label_mask=unknown_label_masks[:, 0],
        past_only_labels=unknown_labels[:, 1:unknown_variates],
        past_only_label_mask=unknown_label_masks[:, 1:unknown_variates],
        eligible_anchor_mask=eligible,
        final_anchor_indices=final_anchor_indices,
        anchor_timestamps=anchor_timestamps,
        context_patch_count=context_patches,
    )


def gather_final_anchor(
    values: torch.Tensor,
    final_anchor_indices: torch.Tensor,
) -> torch.Tensor:
    if values.shape[0] != final_anchor_indices.shape[0]:
        raise ValueError("batch and final-anchor index counts differ")
    return values[
        torch.arange(values.shape[0], device=values.device),
        final_anchor_indices,
    ]
