"""Official TimesFM 3 backbone adapter and parameter-efficient tuning."""

from __future__ import annotations

import json
import math
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
            for attention in (layer.seq_attn, layer.var_attn):
                for name in ("query_proj", "key_proj", "value_proj", "out_proj"):
                    _replace_linear(
                        attention,
                        name,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                    )
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
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.checkpoint = checkpoint
        self.tuning_mode = tuning_mode
        self.trainable_names = tuple(trainable_names or ())
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
    ) -> TimesFM3Adapter:
        """Loads and configures the official TimesFM 3 checkpoint."""

        from timesfm3 import TimesFM3Torch

        backbone = TimesFM3Torch.from_pretrained(model_config.checkpoint)
        if model_config.disable_linear_detrending:
            backbone.use_linear_detrending = False
        trainable_names = configure_tuning(
            backbone,
            mode=adapter_config.type,
            last_n_layers=adapter_config.last_n_layers,
            lora_rank=adapter_config.rank,
            lora_alpha=adapter_config.alpha,
            lora_dropout=adapter_config.dropout,
        )
        adapter = cls(
            backbone,
            checkpoint=model_config.checkpoint,
            tuning_mode=adapter_config.type,
            trainable_names=trainable_names,
        )
        parameter_dtype = (
            torch.bfloat16 if dtype == "bfloat16" and device.type == "cuda" else torch.float32
        )
        return adapter.to(device=device, dtype=parameter_dtype)

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

        if context_values.ndim != 3:
            raise ValueError("context_values must have shape (batch, variates, context)")
        if context_values.shape[1] > 32:
            raise ValueError("TimesFM 3 supports at most 32 variates per forward")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if context_mask is not None and context_mask.shape != context_values.shape:
            raise ValueError("context_mask must match context_values")

        target = context_values[:, :1, :]
        target_mask = context_mask[:, :1, :] if context_mask is not None else None
        covariates = context_values[:, 1:, :] if context_values.shape[1] > 1 else None
        covariate_mask = (
            context_mask[:, 1:, :]
            if context_mask is not None and context_values.shape[1] > 1
            else None
        )

        all_quantiles = self._decode_impl(
            self.backbone,
            target=target,
            horizon=horizon,
            past_only_covariates=covariates,
            target_mask=target_mask,
            past_only_mask=covariate_mask,
        )
        return all_quantiles[:, 0, :horizon, :]

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
            "trainable_names": self.trainable_names,
            "quantiles": self.quantiles,
            "parameter_summary": self.parameter_summary,
        }

    def save_adapter(
        self,
        output_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state_dict(), destination / "adapter.pt")
        combined_metadata = self.checkpoint_metadata()
        combined_metadata.update(metadata or {})
        with (destination / "adapter_config.json").open("w", encoding="utf-8") as handle:
            json.dump(combined_metadata, handle, indent=2, sort_keys=True)

    def load_adapter(self, path: str | Path) -> None:
        """Loads parameters after applying the matching tuning configuration."""

        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        ):
            raise ValueError("adapter checkpoint must be a tensor state dictionary")
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
