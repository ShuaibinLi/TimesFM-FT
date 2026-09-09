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
