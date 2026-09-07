from __future__ import annotations

import torch
from timesfm3 import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3Torch,
    TransformerConfig,
)

from timesfm_ft.adapter import LoRALinear, TimesFM3Adapter, configure_tuning
from timesfm_ft.losses import ForecastLoss


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
        use_iterative_cpm_revin=True,
    )


def test_differentiable_decode_matches_official_decode_and_backpropagates():
    torch.manual_seed(7)
    backbone = make_tiny_model().eval()
    context = torch.randn(2, 3, 16)
    mask = torch.zeros_like(context, dtype=torch.bool)
    expected = backbone.decode(
        target=context[:, :1],
        horizon=6,
        past_only_covariates=context[:, 1:],
        target_mask=mask[:, :1],
        past_only_mask=mask[:, 1:],
    )[:, 0]

    configure_tuning(backbone, mode="head", last_n_layers=1)
    wrapped = TimesFM3Adapter(backbone)
    actual = wrapped(context, horizon=6, context_mask=mask)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    assert backbone.output_head.weight.grad is not None
    assert torch.isfinite(backbone.output_head.weight.grad).all()


def test_lora_is_initially_output_preserving_and_trainable(tmp_path):
    torch.manual_seed(11)
    backbone = make_tiny_model().eval()
    context = torch.randn(2, 1, 16)
    before = backbone.decode(target=context, horizon=6)[:, 0]

    trainable_names = configure_tuning(
        backbone,
        mode="lora",
        last_n_layers=1,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )
    wrapped = TimesFM3Adapter(backbone).eval()
    after = wrapped(context, horizon=6)
    torch.testing.assert_close(after, before)
    assert any("lora_a" in name for name in trainable_names)
    assert any("lora_b" in name for name in trainable_names)
    assert any(isinstance(module, LoRALinear) for module in backbone.modules())

    after.sum().backward()
    lora_b_grads = [
        parameter.grad
        for name, parameter in backbone.named_parameters()
        if "lora_b" in name
    ]
    assert any(gradient is not None for gradient in lora_b_grads)

    wrapped.save_adapter(tmp_path, metadata={"mode": "lora"})
    expected = {
        name: parameter.detach().clone()
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    }
    with torch.no_grad():
        for parameter in backbone.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    wrapped.load_adapter(tmp_path / "adapter.pt")
    actual = {
        name: parameter.detach()
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    }
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_forecast_loss_is_zero_for_perfect_ordered_forecast():
    target = torch.tensor([[100.0, 100.01]])
    predictions = target[:, :, None].repeat(1, 1, 3)
    loss_fn = ForecastLoss(
        [0.1, 0.5, 0.9],
        tick_size=0.01,
        median_huber_weight=0.5,
        crossing_weight=0.05,
    )
    output = loss_fn(
        predictions,
        target,
        current_price=torch.tensor([99.99]),
    )
    torch.testing.assert_close(output.total, torch.tensor(0.0))
