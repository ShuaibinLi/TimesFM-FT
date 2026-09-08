"""Official TimesFM 3 backbone adapter and parameter-efficient tuning."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from timesfm_ft.config import AdapterConfig, ModelConfig, OptimizerConfig


class LoRALinear(nn.Module):
    """Low-rank update for an existing frozen Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.scaling * self.lora_b(
            self.lora_a(self.dropout(inputs))
        )


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _replace_linear(
    parent: nn.Module,
    name: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    layer = getattr(parent, name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"{type(parent).__name__}.{name} is not nn.Linear")
    setattr(
        parent,
        name,
        LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout),
    )


def configure_tuning(
    backbone: nn.Module,
    *,
    mode: Literal["head", "lora", "partial", "full"],
    last_n_layers: int = 4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_sequence_attention: bool = True,
    lora_variate_attention: bool = True,
    lora_feedforward: bool = True,
) -> list[str]:
    """Selects trainable parameters, injecting LoRA modules when requested."""

    if not hasattr(backbone, "transformer_stack") or not hasattr(backbone, "output_head"):
        raise TypeError("expected a TimesFM3Torch-compatible backbone")
    layers = backbone.transformer_stack.layers
    if not 1 <= last_n_layers <= len(layers):
        raise ValueError(f"last_n_layers must be in [1, {len(layers)}]")

    if mode == "full":
        _set_trainable(backbone, True)
    else:
        _set_trainable(backbone, False)
        _set_trainable(backbone.output_head, True)

    selected_layers = list(layers[-last_n_layers:])
    if mode == "partial":
        _set_trainable(backbone.pre_transformer_resblock, True)
        for layer in selected_layers:
            _set_trainable(layer, True)
    elif mode == "lora":
        for layer in selected_layers:
            attentions = []
            if lora_sequence_attention:
                attentions.append(layer.seq_attn)
            if lora_variate_attention:
                attentions.append(layer.var_attn)
            for attention in attentions:
                for name in ("query_proj", "key_proj", "value_proj", "out_proj"):
                    _replace_linear(
                        attention,
                        name,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                    )
            if lora_feedforward:
                for name in ("ff0", "ff1"):
                    _replace_linear(
                        layer,
                        name,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                    )
    elif mode != "head":
        raise ValueError(f"unsupported tuning mode: {mode}")

    names = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError("tuning configuration produced no trainable parameters")
    return names


class TimesFM3Adapter(nn.Module):
    """Thin training adapter around the official ``TimesFM3Torch`` model.

    The released PyTorch ``decode`` method is decorated with ``torch.no_grad``.
    PyTorch preserves the undecorated callable in ``__wrapped__``; this adapter
    calls it explicitly so all official detrending, CPM, RevIN, Transformer,
    and stitching code remains the source of truth while gradients are enabled.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        checkpoint: str | None = None,
        tuning_mode: str | None = None,
        trainable_names: list[str] | None = None,
        compute_dtype: Literal["float32", "bfloat16"] = "float32",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.checkpoint = checkpoint
        self.tuning_mode = tuning_mode
        self.trainable_names = tuple(trainable_names or ())
        self.compute_dtype = compute_dtype
        decorated_decode = getattr(type(backbone), "decode", None)
        decode_impl = getattr(decorated_decode, "__wrapped__", None)
        if decode_impl is None:
            raise TypeError(
                "TimesFM3Torch.decode no longer exposes __wrapped__; "
                "the upstream API changed and the adapter must be reviewed"
            )
        self._decode_impl = decode_impl

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig,
        adapter_config: AdapterConfig,
        *,
        device: torch.device,
        dtype: Literal["float32", "bfloat16"],
        configure_for_training: bool = True,
    ) -> TimesFM3Adapter:
        """Loads and configures the official TimesFM 3 checkpoint."""

        from timesfm3 import TimesFM3Torch

        backbone = TimesFM3Torch.from_pretrained(model_config.checkpoint)
        if model_config.disable_linear_detrending:
            backbone.use_linear_detrending = False
        if model_config.disable_iterative_cpm_revin:
            backbone.use_iterative_cpm_revin = False
        if configure_for_training:
            trainable_names = configure_tuning(
                backbone,
                mode=adapter_config.type,
                last_n_layers=adapter_config.last_n_layers,
                lora_rank=adapter_config.rank,
                lora_alpha=adapter_config.alpha,
                lora_dropout=adapter_config.dropout,
                lora_sequence_attention=adapter_config.lora_sequence_attention,
                lora_variate_attention=adapter_config.lora_variate_attention,
                lora_feedforward=adapter_config.lora_feedforward,
            )
        else:
            _set_trainable(backbone, False)
            trainable_names = []
        effective_compute_dtype = (
            "bfloat16" if dtype == "bfloat16" and device.type == "cuda" else "float32"
        )
        adapter = cls(
            backbone,
            checkpoint=model_config.checkpoint,
            tuning_mode=adapter_config.type if configure_for_training else None,
            trainable_names=trainable_names,
            compute_dtype=effective_compute_dtype,
        )
        # Keep FP32 master parameters and optimizer state. BF16 is a compute
        # policy applied through autocast in forward/predict, not a storage dtype.
        return adapter.to(device=device, dtype=torch.float32)

    def _autocast_context(self) -> torch.autocast:
        device = next(self.parameters()).device
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.compute_dtype == "bfloat16" and device.type == "cuda",
        )

    @property
    def quantiles(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.backbone.quantiles)

    @property
    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable}

    def optimizer_parameter_groups(
        self,
        config: OptimizerConfig,
    ) -> list[dict[str, Any]]:
        """Builds named LR groups for head, LoRA, and unfrozen backbone weights."""

        grouped: dict[str, list[nn.Parameter]] = {
            "head": [],
            "adapter": [],
            "pretrained": [],
        }
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("output_head."):
                grouped["head"].append(parameter)
            elif ".lora_a." in name or ".lora_b." in name:
                grouped["adapter"].append(parameter)
            else:
                grouped["pretrained"].append(parameter)

        learning_rates = {
            "head": config.head_learning_rate,
            "adapter": config.adapter_learning_rate,
            "pretrained": config.pretrained_learning_rate,
        }
        parameter_groups: list[dict[str, Any]] = []
        for name, parameters in grouped.items():
            if not parameters:
                continue
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": learning_rates[name],
                    "weight_decay": 0.0 if name == "adapter" else config.weight_decay,
                    "group_name": name,
                }
            )
        if not parameter_groups:
            raise RuntimeError("adapter produced no optimizer parameter groups")
        return parameter_groups

    def forward(
        self,
        context_values: torch.Tensor,
        *,
        horizon: int,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecasts weighted-mid with optional past-only covariates.

        Variate zero is the weighted-mid target; all remaining variates are
        passed to the official model as past-only covariates.
        """

        decode_kwargs = self._prepare_decode_inputs(
            context_values,
            horizon=horizon,
            context_mask=context_mask,
        )
        with self._autocast_context():
            all_quantiles = self._decode_impl(self.backbone, **decode_kwargs)
        return all_quantiles[:, 0, :horizon, :]

    @torch.inference_mode()
    def predict(
        self,
        context_values: torch.Tensor,
        *,
        horizon: int,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs inference through the official ``TimesFM3Torch.decode`` method."""

        decode_kwargs = self._prepare_decode_inputs(
            context_values,
            horizon=horizon,
            context_mask=context_mask,
        )
        with self._autocast_context():
            all_quantiles = self.backbone.decode(**decode_kwargs)
        return all_quantiles[:, 0, :horizon, :]

    def _prepare_decode_inputs(
        self,
        context_values: torch.Tensor,
        *,
        horizon: int,
        context_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        if context_values.ndim != 3:
            raise ValueError("context_values must have shape (batch, variates, context)")
        if context_values.shape[1] > 32:
            raise ValueError("TimesFM 3 supports at most 32 variates per forward")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if context_mask is not None and context_mask.shape != context_values.shape:
            raise ValueError("context_mask must match context_values")

        context_values = self._stabilize_constant_patches(
            context_values,
            context_mask,
        )
        target = context_values[:, :1, :]
        target_mask = context_mask[:, :1, :] if context_mask is not None else None
        covariates = context_values[:, 1:, :] if context_values.shape[1] > 1 else None
        covariate_mask = (
            context_mask[:, 1:, :]
            if context_mask is not None and context_values.shape[1] > 1
            else None
        )
        return {
            "target": target,
            "horizon": horizon,
            "past_only_covariates": covariates,
            "target_mask": target_mask,
            "past_only_mask": covariate_mask,
        }

    def _stabilize_constant_patches(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Avoid undefined sqrt(0) gradients in upstream running statistics.

        The released inference-only model computes a per-patch standard
        deviation with ``sqrt(var)``. Exactly constant valid patches therefore
        have a finite forward pass but an undefined backward derivative. Add a
        deterministic, sub-tick perturbation only to those patches, ending at
        zero so the forecast cutoff value remains unchanged.
        """

        patch_length = int(self.backbone.input_patch_len)
        if values.shape[-1] % patch_length != 0:
            return values
        patches = values.reshape(*values.shape[:-1], -1, patch_length)
        valid = (
            torch.ones_like(patches, dtype=torch.bool)
            if mask is None
            else ~mask.reshape_as(patches)
        )
        valid_count = valid.sum(dim=-1)
        minimum = torch.where(valid, patches, torch.inf).amin(dim=-1)
        maximum = torch.where(valid, patches, -torch.inf).amax(dim=-1)
        constant = (valid_count > 1) & (minimum == maximum)
        if not constant.any().item():
            return values

        scale = torch.where(
            constant,
            torch.maximum(maximum.abs(), torch.ones_like(maximum)),
            torch.ones_like(maximum),
        )
        amplitude = (
            scale
            * torch.finfo(values.dtype).eps
            * 4.0
        ).detach()
        pattern = torch.linspace(
            -1.0,
            0.0,
            patch_length,
            dtype=values.dtype,
            device=values.device,
        )
        perturbation = (
            constant[..., None]
            * valid
            * amplitude[..., None]
            * pattern
        )
        return (patches + perturbation).reshape_as(values)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {
            name: parameter.detach().cpu()
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        if not trainable:
            raise RuntimeError("no trainable parameters to save")
        return trainable

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "tuning_mode": self.tuning_mode,
            "compute_dtype": self.compute_dtype,
            "trainable_names": self.trainable_names,
            "quantiles": self.quantiles,
            "parameter_summary": self.parameter_summary,
            "use_linear_detrending": bool(self.backbone.use_linear_detrending),
            "use_iterative_cpm_revin": bool(
                self.backbone.use_iterative_cpm_revin
            ),
        }

    def save_adapter(
        self,
        output_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        state_path = destination / "adapter.pt"
        temporary_state = state_path.with_suffix(".pt.tmp")
        torch.save(self.trainable_state_dict(), temporary_state)
        os.replace(temporary_state, state_path)
        combined_metadata = self.checkpoint_metadata()
        combined_metadata.update(metadata or {})
        metadata_path = destination / "adapter_config.json"
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(combined_metadata, handle, indent=2, sort_keys=True)
        os.replace(temporary_metadata, metadata_path)

    def load_adapter(self, path: str | Path) -> None:
        """Loads parameters after applying the matching tuning configuration."""

        checkpoint_path = Path(path)
        metadata_path = checkpoint_path.parent / "adapter_config.json"
        if not metadata_path.exists():
            raise ValueError(f"adapter metadata not found: {metadata_path}")
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected_metadata = {
            "checkpoint": self.checkpoint,
            "tuning_mode": self.tuning_mode,
            "trainable_names": list(self.trainable_names),
            "quantiles": list(self.quantiles),
            "use_linear_detrending": bool(self.backbone.use_linear_detrending),
            "use_iterative_cpm_revin": bool(
                self.backbone.use_iterative_cpm_revin
            ),
        }
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"adapter metadata mismatch for {key}: "
                    f"received {metadata.get(key)!r}, expected {expected_value!r}"
                )

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        ):
            raise ValueError("adapter checkpoint must be a tensor state dictionary")
        self.load_trainable_state_dict(state)

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load an already-deserialized trainable-only state dictionary."""

        expected = {
            name
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        received = set(state)
        if received != expected:
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            raise ValueError(
                f"adapter structure mismatch; missing={missing}, unexpected={unexpected}"
            )
        incompatible = self.backbone.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"unexpected adapter keys: {incompatible.unexpected_keys}")
