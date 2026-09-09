from __future__ import annotations

import pytest
import torch
from timesfm3 import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3Torch,
    TransformerConfig,
)

from timesfm_ft.adapter import TimesFM3Adapter, configure_tuning
from timesfm_ft.config import OptimizerConfig
from timesfm_ft.dense import build_dense_training_batch, gather_final_anchor


def make_tiny_model() -> TimesFM3Torch:
    return TimesFM3Torch(
        input_patch_len=8,
        output_patch_len=16,
        quantiles=[0.1, 0.5, 0.9],
        residual_block_config=ResidualBlockConfig(
            hidden_dims=32,
            output_dims=32,
            use_bias=False,
            activation="relu",
        ),
        transformer_config=StackedTransformersConfig(
            num_layers=2,
            use_remat=False,
            transformer=TransformerConfig(
                model_dims=32,
                hidden_dims=32,
                num_heads=4,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_rope_seq=True,
                use_rope_var=False,
                use_bias=False,
                ff_activation="relu",
                deterministic=True,
                use_sdpa=True,
            ),
        ),
        use_stitching=True,
        use_linear_detrending=True,
        use_iterative_cpm_revin=False,
    )


def test_adapter_matches_official_decode_with_both_covariate_classes():
    torch.manual_seed(7)
    backbone = make_tiny_model().eval()
    context = torch.randn(2, 3, 16)
    context_mask = torch.zeros_like(context, dtype=torch.bool)
    context_mask[0, :, :3] = True
    known = torch.randn(2, 2, 22)
    known_mask = torch.zeros_like(known, dtype=torch.bool)
    known_mask[0, :, :3] = True
    expected_all = backbone.decode(
        target=context[:, :1],
        horizon=6,
        past_only_covariates=context[:, 1:],
        past_future_covariates=known,
        target_mask=context_mask[:, :1],
        past_only_mask=context_mask[:, 1:],
        past_future_mask=known_mask,
        mask=context_mask[:, 0],
    )
    configure_tuning(backbone, mode="head", last_n_layers=1)
    adapter = TimesFM3Adapter(backbone)
    raw = adapter.forward_all(
        context,
        horizon=6,
        context_mask=context_mask,
        context_padding_mask=context_mask[:, 0],
        past_future_values=known,
        past_future_mask=known_mask,
    )
    actual = adapter(
        context,
        horizon=6,
        context_mask=context_mask,
        context_padding_mask=context_mask[:, 0],
        past_future_values=known,
        past_future_mask=known_mask,
    )
    unknown = adapter.forward_unknown(
        context,
        horizon=6,
        context_mask=context_mask,
        context_padding_mask=context_mask[:, 0],
        past_future_values=known,
        past_future_mask=known_mask,
    )
    torch.testing.assert_close(raw, expected_all)
    torch.testing.assert_close(actual, expected_all[:, 0])
    torch.testing.assert_close(unknown.target, expected_all[:, 0])
    torch.testing.assert_close(unknown.past_only, expected_all[:, 1:3])
    assert raw.shape == (2, 5, 6, 3)
    actual.sum().backward()
    assert torch.isfinite(backbone.output_head.weight.grad).all()


def test_constant_target_and_known_event_covariate_have_finite_gradients():
    backbone = make_tiny_model()
    names = configure_tuning(
        backbone,
        mode="lora",
        last_n_layers=1,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0,
    )
    adapter = TimesFM3Adapter(
        backbone,
        tuning_mode="lora",
        trainable_names=names,
    )
    context = torch.ones(2, 2, 16)
    known = torch.zeros(2, 1, 22)
    prediction = adapter(
        context,
        horizon=6,
        past_future_values=known,
    )
    prediction.sum().backward()
    for name, parameter in adapter.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_padding_mask_is_distinct_from_target_missing_mask():
    adapter = TimesFM3Adapter(make_tiny_model())
    context = torch.ones(1, 2, 16)
    context_mask = torch.zeros_like(context, dtype=torch.bool)
    context_mask[:, 0, 5] = True
    padding_mask = torch.zeros(1, 16, dtype=torch.bool)
    padding_mask[:, :2] = True
    prepared = adapter._prepare_decode_inputs(
        context,
        horizon=6,
        context_mask=context_mask,
        context_padding_mask=padding_mask,
    )
    assert prepared["mask"].equal(padding_mask)
    assert prepared["target_mask"][0, 0, 5]
    assert not prepared["past_only_mask"][0, 0, 5]


def test_dense_forward_matches_low_level_backbone_and_backpropagates():
    torch.manual_seed(17)
    backbone = make_tiny_model().eval()
    backbone.use_linear_detrending = False
    values = torch.randn(2, 3, 5, 8)
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.zeros(2, 3, 5, dtype=torch.bool)
    patch_is_target[:, :2] = True
    expected = backbone(
        {
            "values": values,
            "masks": masks,
            "patch_is_target": patch_is_target,
        },
        patch_cpm_mask=None,
    )["logits"]
    configure_tuning(backbone, mode="head", last_n_layers=1)
    adapter = TimesFM3Adapter(backbone)
    actual = adapter.forward_dense(
        values,
        masks=masks,
        patch_is_target=patch_is_target,
        unknown_variates=2,
    )
    torch.testing.assert_close(actual.target, expected[:, 0])
    torch.testing.assert_close(actual.past_only, expected[:, 1:2])
    actual.target.sum().backward()
    assert torch.isfinite(backbone.output_head.weight.grad).all()


def test_dense_forward_rejects_decode_only_linear_detrending():
    adapter = TimesFM3Adapter(make_tiny_model())
    values = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="linear detrending"):
        adapter.forward_dense(
            values,
            masks=torch.zeros_like(values, dtype=torch.bool),
            patch_is_target=torch.ones(1, 1, 4, dtype=torch.bool),
            unknown_variates=1,
        )


def test_public_forward_keeps_interior_masked_patch_and_ignores_role_metadata():
    backbone = make_tiny_model().eval()
    backbone.use_linear_detrending = False
    values = torch.randn(1, 1, 4, 8)
    masks = torch.zeros_like(values, dtype=torch.bool)
    masks[:, :, 0] = True
    masks[:, :, 2] = True
    patch_is_target = torch.ones(1, 1, 4, dtype=torch.bool)
    patch_mask = backbone._preprocess(
        values,
        masks,
        patch_is_target,
    )[2]
    assert patch_mask.tolist() == [[[True, False, True, False]]]
    effective = torch.cumprod(patch_mask.int(), dim=2).bool()
    assert effective.tolist() == [[[True, False, False, False]]]
    base_inputs = {
        "values": values,
        "masks": masks,
        "patch_is_target": patch_is_target,
        "patch_segment_ids": torch.zeros(1, 4, dtype=torch.long),
        "patch_positions": torch.arange(4)[None],
        "patch_is_past_only": torch.zeros(1, 1, 4, dtype=torch.bool),
    }
    base = backbone(base_inputs, return_aux_outputs=True)
    changed_inputs = dict(base_inputs)
    changed_inputs["patch_segment_ids"] = torch.ones(1, 4, dtype=torch.long)
    changed_inputs["patch_positions"] = torch.arange(4, 8)[None]
    changed_inputs["patch_is_past_only"] = torch.ones(1, 1, 4, dtype=torch.bool)
    changed = backbone(changed_inputs, return_aux_outputs=True)
    torch.testing.assert_close(base["logits"], changed["logits"])
    backbone.train()
    trained = backbone(base_inputs, return_aux_outputs=True)
    for eval_mask, train_mask in zip(
        base["__call__:seq_attn_mask"],
        trained["__call__:seq_attn_mask"],
        strict=True,
    ):
        assert torch.equal(eval_mask, train_mask)


def test_dense_final_token_matches_deployment_decode_when_detrending_is_off():
    torch.manual_seed(23)
    backbone = make_tiny_model().eval()
    backbone.use_linear_detrending = False
    batch_size, context, horizon = 2, 16, 16
    context_values = torch.randn(batch_size, 2, context)
    context_mask = torch.zeros_like(context_values, dtype=torch.bool)
    target_future = torch.randn(batch_size, horizon)
    past_only_future = torch.randn(batch_size, 1, horizon)
    known = torch.randn(batch_size, 1, context + horizon)
    batch = {
        "context_values": context_values,
        "context_mask": context_mask,
        "context_padding_mask": torch.zeros(batch_size, context, dtype=torch.bool),
        "past_future_values": known,
        "past_future_mask": torch.zeros_like(known, dtype=torch.bool),
        "unknown_future_values": torch.cat(
            (target_future[:, None], past_only_future),
            dim=1,
        ),
        "unknown_future_mask": torch.zeros(batch_size, 2, horizon, dtype=torch.bool),
        "context_lengths": torch.full((batch_size,), context),
        "timestamps": torch.arange(batch_size),
        "dates": torch.full((batch_size,), 20250102),
        "minute_indices": torch.full((batch_size,), context - 1),
        "last_returns": context_values[:, 0, -1],
        "context_volatility": torch.ones(batch_size),
    }
    expected = backbone.decode(
        target=context_values[:, :1],
        horizon=horizon,
        past_only_covariates=context_values[:, 1:],
        past_future_covariates=known,
        target_mask=context_mask[:, :1],
        past_only_mask=context_mask[:, 1:],
        past_future_mask=torch.zeros_like(known, dtype=torch.bool),
    )[:, 0]
    dense = build_dense_training_batch(
        batch,
        context_min=8,
        input_patch_length=8,
        output_patch_length=16,
    )
    configure_tuning(backbone, mode="head", last_n_layers=1)
    adapter = TimesFM3Adapter(backbone)
    dense_output = adapter.forward_dense(
        dense.values,
        masks=dense.masks,
        patch_is_target=dense.patch_is_target,
        unknown_variates=2,
    )
    actual = gather_final_anchor(
        dense_output.target,
        dense.final_anchor_indices,
    )
    torch.testing.assert_close(actual, expected)
    perturbed_values = dense.values.clone()
    perturbed_values[:, 0, dense.context_patch_count :, :] += 100.0
    perturbed = adapter.forward_dense(
        perturbed_values,
        masks=dense.masks,
        patch_is_target=dense.patch_is_target,
        unknown_variates=2,
    )
    torch.testing.assert_close(
        perturbed.target[:, : dense.context_patch_count],
        dense_output.target[:, : dense.context_patch_count],
    )
    past_only_perturbed = dense.values.clone()
    past_only_perturbed[:, 1, dense.context_patch_count :, :] += 100.0
    past_only_output = adapter.forward_dense(
        past_only_perturbed,
        masks=dense.masks,
        patch_is_target=dense.patch_is_target,
        unknown_variates=2,
    )
    torch.testing.assert_close(
        past_only_output.target[:, : dense.context_patch_count],
        dense_output.target[:, : dense.context_patch_count],
    )
    base_resblock = backbone._preprocess(
        dense.values,
        dense.masks,
        dense.patch_is_target,
    )[0]
    known_perturbed = dense.values.clone()
    known_perturbed[:, 2, dense.context_patch_count :, :] += 100.0
    known_resblock = backbone._preprocess(
        known_perturbed,
        dense.masks,
        dense.patch_is_target,
    )[0]
    assert not torch.allclose(
        base_resblock[:, 2, dense.context_patch_count - 1],
        known_resblock[:, 2, dense.context_patch_count - 1],
    )


def test_adapter_enforces_total_variate_budget():
    adapter = TimesFM3Adapter(make_tiny_model())
    context = torch.zeros(1, 30, 16)
    known = torch.zeros(1, 3, 22)
    try:
        adapter(context, horizon=6, past_future_values=known)
    except ValueError as error:
        assert "at most 32" in str(error)
    else:
        raise AssertionError("expected variate budget failure")


def test_lora_optimizer_groups_preserve_separate_learning_rates():
    backbone = make_tiny_model()
    configure_tuning(
        backbone,
        mode="lora",
        last_n_layers=1,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0,
    )
    adapter = TimesFM3Adapter(backbone)
    groups = adapter.optimizer_parameter_groups(OptimizerConfig())
    rates = {group["group_name"]: group["lr"] for group in groups}
    assert rates == {"head": 3e-4, "adapter": 1e-4}


def test_adapter_load_rejects_feature_order_mismatch(tmp_path):
    backbone = make_tiny_model()
    names = configure_tuning(backbone, mode="head", last_n_layers=1)
    adapter = TimesFM3Adapter(
        backbone,
        checkpoint="tiny",
        tuning_mode="head",
        trainable_names=names,
    )
    adapter.save_adapter(
        tmp_path,
        metadata={"past_only_features": ["spread", "volume"]},
    )
    with pytest.raises(ValueError, match="past_only_features"):
        adapter.load_adapter(
            tmp_path / "adapter.pt",
            expected_metadata={
                "past_only_features": ["volume", "spread"],
            },
        )
