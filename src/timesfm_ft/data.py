"""Window-level data contract for single-target forecasting."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowBatch(TypedDict):
    context_values: torch.Tensor
    context_mask: torch.Tensor
    future_values: torch.Tensor
    future_mask: torch.Tensor


class NpzWindowDataset(Dataset[WindowBatch]):
    """Loads NPZ smoke data or a memory-mapped production window bundle.

    A bundle is a directory containing ``context_values.npy``,
    ``future_values.npy``, ``timestamps.npy``, ``dates.npy``, and
    ``manifest.json``. Optional masks use matching ``*_mask.npy`` names.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        context_length: int,
        horizon_length: int,
        max_variates: int = 32,
        sampling_interval_seconds: float | None = None,
        expected_stride: int | None = None,
        expected_product: str | None = None,
        expected_split: str | None = None,
        expected_dates: set[int] | None = None,
        expected_dates_path: str | Path | None = None,
        require_metadata: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.metadata = self._load_metadata(self.path)
        loaded = self._load_arrays(self.path)
        if self.path.is_dir():
            self._validate_bundle_integrity(loaded, self.metadata)
        contexts = np.asarray(loaded["context_values"], dtype=np.float32)
        futures = np.asarray(loaded["future_values"], dtype=np.float32)
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

        context_mask = self._prepare_mask(
            loaded.get("context_mask"), contexts, "context_mask"
        )
        future_mask = self._prepare_mask(
            loaded.get("future_mask"), futures, "future_mask"
        )
        if context_mask is not None and context_mask.ndim == 2:
            context_mask = context_mask[:, None, :]
        if context_mask is not None and context_mask.shape != contexts.shape:
            raise ValueError(
                f"context_mask shape {context_mask.shape} != {contexts.shape}"
            )
        if future_mask is not None and future_mask.shape != futures.shape:
            raise ValueError(
                f"future_mask shape {future_mask.shape} != {futures.shape}"
            )

        context_nonfinite = ~np.isfinite(contexts)
        future_nonfinite = ~np.isfinite(futures)
        if context_nonfinite.any():
            context_mask = (
                context_nonfinite
                if context_mask is None
                else context_mask | context_nonfinite
            )
            contexts = np.nan_to_num(contexts, copy=True)
        if future_nonfinite.any():
            future_mask = (
                future_nonfinite
                if future_mask is None
                else future_mask | future_nonfinite
            )
            futures = np.nan_to_num(futures, copy=True)
        if context_mask is not None:
            if np.any(np.all(context_mask[:, 0, :], axis=-1)):
                raise ValueError("weighted-mid context cannot be fully masked")
            if np.any(context_mask[:, 0, -1]):
                raise ValueError(
                    "the weighted-mid value at every forecast cutoff must be valid"
                )
        if future_mask is not None and np.any(np.all(future_mask, axis=-1)):
            raise ValueError("every sample needs at least one valid future target")

        self.context_values = contexts
        self.future_values = futures
        self.context_mask = context_mask
        self.future_mask = future_mask
        self.timestamps = self._optional_vector(
            loaded.get("timestamps"), len(contexts), np.int64, "timestamps"
        )
        self.dates = self._optional_vector(
            loaded.get("dates"), len(contexts), np.int32, "dates"
        )
        self._validate_metadata(
            context_length=context_length,
            horizon_length=horizon_length,
            sampling_interval_seconds=sampling_interval_seconds,
            expected_stride=expected_stride,
            expected_product=expected_product,
            expected_split=expected_split,
            expected_dates=expected_dates,
            expected_dates_path=expected_dates_path,
            require_metadata=require_metadata,
        )

    @staticmethod
    def _load_arrays(path: Path) -> dict[str, np.ndarray]:
        names = (
            "context_values",
            "future_values",
            "context_mask",
            "future_mask",
            "timestamps",
            "dates",
        )
        if path.is_dir():
            arrays: dict[str, np.ndarray] = {}
            for name in names:
                array_path = path / f"{name}.npy"
                if array_path.exists():
                    arrays[name] = np.load(
                        array_path,
                        mmap_mode="c",
                        allow_pickle=False,
                    )
            if "context_values" not in arrays or "future_values" not in arrays:
                raise ValueError(
                    f"{path} must contain context_values.npy and future_values.npy"
                )
            return arrays

        with np.load(path, allow_pickle=False) as archive:
            if "context_values" not in archive or "future_values" not in archive:
                raise ValueError(
                    f"{path} must contain context_values and future_values"
                )
            return {name: archive[name] for name in names if name in archive}

    @staticmethod
    def _prepare_mask(
        mask: np.ndarray | None,
        values: np.ndarray,
        name: str,
    ) -> np.ndarray | None:
        if mask is None:
            return None
        result = np.asarray(mask, dtype=np.bool_)
        if result.shape != values.shape and not (
            result.ndim == 2
            and values.ndim == 3
            and values.shape[1] == 1
            and result.shape == (values.shape[0], values.shape[2])
        ):
            raise ValueError(f"{name} shape {result.shape} != {values.shape}")
        return result

    @staticmethod
    def _optional_vector(
        value: np.ndarray | None,
        samples: int,
        dtype: np.dtype[Any],
        name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        result = np.asarray(value, dtype=dtype)
        if result.shape != (samples,):
            raise ValueError(f"{name} must have shape ({samples},), got {result.shape}")
        return result

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any] | None:
        metadata_path = path / "manifest.json" if path.is_dir() else path.with_suffix(".json")
        if not metadata_path.exists():
            return None
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"dataset metadata must be an object: {metadata_path}")
        return metadata

    def _validate_bundle_integrity(
        self,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any] | None,
    ) -> None:
        if metadata is None:
            raise ValueError(f"memory-mapped bundle requires manifest.json: {self.path}")
        if metadata.get("format_version") != 1:
            raise ValueError(
                f"unsupported bundle format_version={metadata.get('format_version')!r}"
            )
        if metadata.get("format") != "timesfm-ft-npy-bundle":
            raise ValueError(f"unsupported bundle format={metadata.get('format')!r}")

        required_dtypes = {
            "context_values": np.dtype(np.float32),
            "future_values": np.dtype(np.float32),
            "timestamps": np.dtype(np.int64),
            "dates": np.dtype(np.int32),
        }
        for name, expected_dtype in required_dtypes.items():
            if name not in arrays:
                raise ValueError(f"bundle is missing required array {name}.npy")
            if arrays[name].dtype != expected_dtype:
                raise ValueError(
                    f"{name} dtype={arrays[name].dtype}, expected {expected_dtype}"
                )
        for name in ("context_mask", "future_mask"):
            if name in arrays and arrays[name].dtype != np.dtype(np.bool_):
                raise ValueError(f"{name} dtype must be bool")

        samples = int(metadata.get("samples", -1))
        if samples < 0:
            raise ValueError("manifest samples must be non-negative")
        for name, array in arrays.items():
            if array.shape[0] != samples:
                raise ValueError(
                    f"{name} sample count={array.shape[0]}, manifest={samples}"
                )

        schema = metadata.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("manifest schema must be an object")
        for name, expected_dtype in required_dtypes.items():
            expected_schema = [str(expected_dtype), *arrays[name].shape]
            if schema.get(name) != expected_schema:
                raise ValueError(
                    f"manifest schema for {name}={schema.get(name)!r}, "
                    f"expected {expected_schema!r}"
                )

        dates = arrays["dates"]
        date_values, counts = np.unique(dates, return_counts=True)
        date_counts = {
            str(int(day)): int(count)
            for day, count in zip(date_values, counts, strict=True)
        }
        manifest_counts = metadata.get("samples_by_day")
        if manifest_counts != date_counts:
            raise ValueError("manifest samples_by_day does not match dates.npy")
        unique_dates = sorted(date_counts)
        if metadata.get("date_count") != len(unique_dates):
            raise ValueError("manifest date_count does not match dates.npy")
        if not unique_dates:
            raise ValueError("bundle contains no dates")
        if metadata.get("first_date") != unique_dates[0]:
            raise ValueError("manifest first_date does not match dates.npy")
        if metadata.get("last_date") != unique_dates[-1]:
            raise ValueError("manifest last_date does not match dates.npy")

        session = metadata.get("session")
        expected_session = {
            "timezone": "America/New_York",
            "start": "09:30:00",
            "end": "16:15:00",
            "early_closes_allowed": True,
        }
        if session != expected_session:
            raise ValueError(
                f"manifest session={session!r}, expected {expected_session!r}"
            )

    def _validate_metadata(
        self,
        *,
        context_length: int,
        horizon_length: int,
        sampling_interval_seconds: float | None,
        expected_stride: int | None,
        expected_product: str | None,
        expected_split: str | None,
        expected_dates: set[int] | None,
        expected_dates_path: str | Path | None,
        require_metadata: bool,
    ) -> None:
        if require_metadata and self.metadata is None:
            raise ValueError(f"dataset metadata is required for {self.path}")
        if require_metadata and (self.timestamps is None or self.dates is None):
            raise ValueError(f"timestamps and dates are required for {self.path}")
        validate_claims = require_metadata or any(
            value is not None
            for value in (
                sampling_interval_seconds,
                expected_stride,
                expected_product,
                expected_split,
                expected_dates,
                expected_dates_path,
            )
        )
        if self.metadata is not None and validate_claims:
            expected = {
                "context_length": context_length,
                "horizon_length": horizon_length,
                "stride": expected_stride,
                "product": expected_product,
                "split": expected_split,
            }
            for key, value in expected.items():
                if value is not None and self.metadata.get(key) != value:
                    raise ValueError(
                        f"{self.path} metadata {key}={self.metadata.get(key)!r}, "
                        f"expected {value!r}"
                    )
            if sampling_interval_seconds is not None and not math.isclose(
                float(self.metadata.get("sampling_interval_seconds", math.nan)),
                sampling_interval_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{self.path} sampling interval does not match config"
                )
            if expected_dates_path is not None:
                date_hash = hashlib.sha256(
                    Path(expected_dates_path).read_bytes()
                ).hexdigest()
                if self.metadata.get("date_file_sha256") != date_hash:
                    raise ValueError(
                        f"{self.path} date-list provenance hash mismatch"
                    )

        if expected_dates is not None:
            if self.dates is None:
                raise ValueError(f"dates are required to validate split {self.path}")
            actual_dates = set(int(value) for value in np.unique(self.dates))
            if actual_dates != expected_dates:
                missing = sorted(expected_dates - actual_dates)
                extra = sorted(actual_dates - expected_dates)
                raise ValueError(
                    f"{self.path} split dates mismatch; missing={missing[:5]} "
                    f"extra={extra[:5]}"
                )
        if expected_stride is not None and sampling_interval_seconds is None:
            raise ValueError(
                "sampling_interval_seconds is required with expected_stride"
            )
        if self.timestamps is not None and self.dates is not None and expected_stride:
            expected_step_ns = int(
                round((sampling_interval_seconds or 0.0) * 1_000_000_000)
            ) * expected_stride
            for day in np.unique(self.dates):
                values = self.timestamps[self.dates == day]
                if len(values) > 1 and not np.all(np.diff(values) == expected_step_ns):
                    raise ValueError(
                        f"{self.path} cutoff cadence mismatch on date {int(day)}"
                    )

    @property
    def num_variates(self) -> int:
        return int(self.context_values.shape[1])

    def __len__(self) -> int:
        return int(self.context_values.shape[0])

    def __getitem__(self, index: int) -> WindowBatch:
        context = torch.from_numpy(np.asarray(self.context_values[index]))
        future = torch.from_numpy(np.asarray(self.future_values[index]))
        context_mask = (
            torch.zeros_like(context, dtype=torch.bool)
            if self.context_mask is None
            else torch.from_numpy(np.asarray(self.context_mask[index]))
        )
        future_mask = (
            torch.zeros_like(future, dtype=torch.bool)
            if self.future_mask is None
            else torch.from_numpy(np.asarray(self.future_mask[index]))
        )
        return {
            "context_values": context,
            "context_mask": context_mask,
            "future_values": future,
            "future_mask": future_mask,
        }
