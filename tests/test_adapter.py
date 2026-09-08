from __future__ import annotations

import json

import pytest
import torch
from timesfm3 import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3Torch,
    TransformerConfig,
)

from timesfm_ft.adapter import LoRALinear, TimesFM3Adapter, configure_tuning
from timesfm_ft.config import AdapterConfig, ModelConfig, OptimizerConfig
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
    inference = wrapped.predict(context, horizon=6, context_mask=mask)
    torch.testing.assert_close(inference, expected)
    assert not inference.requires_grad

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


def test_lora_backward_is_finite_for_constant_input_patches():
    torch.manual_seed(13)
    backbone = make_tiny_model()
    names = configure_tuning(
        backbone,
        mode="lora",
        last_n_layers=1,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )
    adapter = TimesFM3Adapter(
        backbone,
        tuning_mode="lora",
        trainable_names=names,
    )
    context = torch.full((2, 1, 16), 110.0)
    prepared = adapter._prepare_decode_inputs(
        context,
        horizon=6,
        context_mask=None,
    )
    torch.testing.assert_close(prepared["target"][:, :, -1], context[:, :, -1])
    prediction = adapter(context, horizon=6)
    prediction.sum().backward()
    for name, parameter in adapter.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


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


def test_optimizer_groups_use_distinct_learning_rates():
    backbone = make_tiny_model()
    configure_tuning(
        backbone,
        mode="lora",
        last_n_layers=1,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )
    adapter = TimesFM3Adapter(backbone)
    config = OptimizerConfig(
        adapter_learning_rate=1e-4,
        head_learning_rate=3e-4,
        pretrained_learning_rate=1e-5,
    )
    groups = adapter.optimizer_parameter_groups(config)
    group_lrs = {group["group_name"]: group["lr"] for group in groups}
    assert group_lrs == {"head": 3e-4, "adapter": 1e-4}


def test_lora_can_skip_variate_attention_for_univariate_inputs():
    backbone = make_tiny_model()
    names = configure_tuning(
        backbone,
        mode="lora",
        last_n_layers=1,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        lora_sequence_attention=True,
        lora_variate_attention=False,
        lora_feedforward=True,
    )
    assert any("seq_attn" in name and "lora_" in name for name in names)
    assert any(".ff0.lora_" in name for name in names)
    assert not any("var_attn" in name and "lora_" in name for name in names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 smoke")
def test_bfloat16_compute_keeps_fp32_master_weights(monkeypatch):
    monkeypatch.setattr(
        TimesFM3Torch,
        "from_pretrained",
        classmethod(lambda cls, _checkpoint: make_tiny_model()),
    )
    adapter = TimesFM3Adapter.from_pretrained(
        ModelConfig(checkpoint="tiny"),
        AdapterConfig(type="head", last_n_layers=1),
        device=torch.device("cuda"),
        dtype="bfloat16",
    )
    assert adapter.compute_dtype == "bfloat16"
    assert {parameter.dtype for parameter in adapter.parameters()} == {torch.float32}
    context = torch.randn(2, 1, 16, device="cuda")
    prediction = adapter.predict(context, horizon=6)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()


def test_load_adapter_rejects_metadata_mismatch(tmp_path):
    backbone = make_tiny_model()
    names = configure_tuning(backbone, mode="head", last_n_layers=1)
    adapter = TimesFM3Adapter(
        backbone,
        checkpoint="tiny",
        tuning_mode="head",
        trainable_names=names,
    )
    adapter.save_adapter(tmp_path)
    metadata_path = tmp_path / "adapter_config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata mismatch"):
        adapter.load_adapter(tmp_path / "adapter.pt")
