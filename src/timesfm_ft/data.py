"""Window-level data contract for single-target forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowBatch(TypedDict):
    context_values: torch.Tensor
    context_mask: torch.Tensor
    future_values: torch.Tensor
    future_mask: torch.Tensor


class NpzWindowDataset(Dataset[WindowBatch]):
    """Loads pre-windowed 500 ms data from an NPZ file.

    Required arrays:
      context_values: (samples, variates, context) or (samples, context)
      future_values: (samples, horizon), weighted-mid only

    Optional arrays:
      context_mask: same shape as context_values; True means unavailable
      future_mask: same shape as future_values; True means excluded from loss
      timestamps: (samples,), retained by the source file but unused by training
    """

    def __init__(
        self,
        path: str | Path,
        *,
        context_length: int,
        horizon_length: int,
        max_variates: int = 32,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        archive = np.load(self.path, allow_pickle=False)
        if "context_values" not in archive or "future_values" not in archive:
            raise ValueError(
                f"{self.path} must contain context_values and future_values"
            )

        contexts = np.asarray(archive["context_values"], dtype=np.float32)
        futures = np.asarray(archive["future_values"], dtype=np.float32)
        if contexts.ndim == 2:
            contexts = contexts[:, None, :]
        if contexts.ndim != 3:
            raise ValueError(
                f"context_values must be rank 2 or 3, got {contexts.shape}"
            )
        if futures.ndim != 2:
            raise ValueError(f"future_values must be rank 2, got {futures.shape}")
        if contexts.shape[0] != futures.shape[0]:
            raise ValueError("context_values and future_values sample counts differ")
        if contexts.shape[-1] != context_length:
            raise ValueError(
                f"expected context_length={context_length}, got {contexts.shape[-1]}"
            )
        if futures.shape[-1] != horizon_length:
            raise ValueError(
                f"expected horizon_length={horizon_length}, got {futures.shape[-1]}"
            )
        if contexts.shape[1] > max_variates:
            raise ValueError(
                f"received {contexts.shape[1]} variates; limit is {max_variates}"
            )

        context_mask = (
            np.asarray(archive["context_mask"], dtype=np.bool_)
            if "context_mask" in archive
            else np.zeros_like(contexts, dtype=np.bool_)
        )
        future_mask = (
            np.asarray(archive["future_mask"], dtype=np.bool_)
            if "future_mask" in archive
            else np.zeros_like(futures, dtype=np.bool_)
        )
        if context_mask.ndim == 2:
            context_mask = context_mask[:, None, :]
        if context_mask.shape != contexts.shape:
            raise ValueError(
                f"context_mask shape {context_mask.shape} != {contexts.shape}"
            )
        if future_mask.shape != futures.shape:
            raise ValueError(f"future_mask shape {future_mask.shape} != {futures.shape}")

        context_mask = context_mask | ~np.isfinite(contexts)
        future_mask = future_mask | ~np.isfinite(futures)
        if np.any(np.all(context_mask[:, 0, :], axis=-1)):
            raise ValueError("weighted-mid context cannot be fully masked")
        if np.any(context_mask[:, 0, -1]):
            raise ValueError("the weighted-mid value at every forecast cutoff must be valid")
        if np.any(np.all(future_mask, axis=-1)):
            raise ValueError("every sample needs at least one valid future target")

        self.context_values = np.nan_to_num(contexts, copy=True)
        self.context_mask = context_mask
        self.future_values = np.nan_to_num(futures, copy=True)
        self.future_mask = future_mask

    @property
    def num_variates(self) -> int:
        return int(self.context_values.shape[1])

    def __len__(self) -> int:
        return int(self.context_values.shape[0])

    def __getitem__(self, index: int) -> WindowBatch:
        return {
            "context_values": torch.from_numpy(self.context_values[index]),
            "context_mask": torch.from_numpy(self.context_mask[index]),
            "future_values": torch.from_numpy(self.future_values[index]),
            "future_mask": torch.from_numpy(self.future_mask[index]),
        }
