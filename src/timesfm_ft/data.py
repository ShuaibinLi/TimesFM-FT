"""Audited intraday-minute dataset with dynamic context windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

BUNDLE_FORMAT = "timesfm-ft-intraday-minute-bundle"
BUNDLE_VERSION = 2


class WindowSample(TypedDict):
    context_values: torch.Tensor
    context_mask: torch.Tensor
    past_future_values: torch.Tensor
    past_future_mask: torch.Tensor
    unknown_future_values: torch.Tensor
    unknown_future_mask: torch.Tensor
    context_length: int
    timestamp: int
    date: int
    minute_index: int
    last_return: float
    context_volatility: float


class WindowBatch(TypedDict):
    context_values: torch.Tensor
    context_mask: torch.Tensor
    context_padding_mask: torch.Tensor
    past_future_values: torch.Tensor
    past_future_mask: torch.Tensor
    unknown_future_values: torch.Tensor
    unknown_future_mask: torch.Tensor
    context_lengths: torch.Tensor
    timestamps: torch.Tensor
    dates: torch.Tensor
    minute_indices: torch.Tensor
    last_returns: torch.Tensor
    context_volatility: torch.Tensor


def _date_file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dates(path: str | Path) -> tuple[int, ...]:
    values = tuple(
        int(line.strip())
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if not values:
        raise ValueError(f"no dates in {path}")
    if len(values) != len(set(values)) or values != tuple(sorted(values)):
        raise ValueError(f"dates must be unique and sorted: {path}")
    return values


class IntradayWindowDataset(Dataset[WindowSample]):
    """Builds non-overnight dynamic-context samples from a session bundle.

    The bundle stores one row per trade date and one column per minute. Windows
    are sliced lazily, so overlapping samples do not duplicate the underlying
    feature data on disk.
    """

    _ARRAYS = (
        "target_values",
        "target_mask",
        "past_only_values",
        "past_only_mask",
        "past_future_values",
        "past_future_mask",
        "timestamps",
        "dates",
        "session_lengths",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        context_min: int,
        context_max: int,
        horizon_length: int,
        stride: int,
        past_only_features: Sequence[str] = (),
        past_future_features: Sequence[str] = (),
        max_variates: int = 32,
        require_complete_future: bool = False,
        expected_split: str | None = None,
        expected_dataset_id: str | None = None,
        expected_product: str | None = None,
        expected_target_name: str | None = None,
        expected_target_unit: str | None = None,
        expected_target_price_source: str | None = None,
        expected_target_return_type: str | None = None,
        expected_target_timestamp_semantics: str | None = None,
        expected_target_availability_lag_minutes: int | None = None,
        expected_target_missing_policy: str | None = None,
        expected_frequency_minutes: int | None = None,
        expected_session_minutes: int | None = None,
        expected_dates_path: str | Path | None = None,
        require_metadata: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"intraday bundle directory not found: {self.path}")
        if not 0 < context_min <= context_max:
            raise ValueError("context lengths must satisfy 0 < min <= max")
        if min(horizon_length, stride) <= 0:
            raise ValueError("horizon_length and stride must be positive")

        metadata_path = self.path / "manifest.json"
        if not metadata_path.exists():
            if require_metadata:
                raise ValueError(f"bundle requires manifest.json: {self.path}")
            self.metadata: dict[str, Any] = {}
        else:
            with metadata_path.open(encoding="utf-8") as handle:
                self.metadata = json.load(handle)
            if not isinstance(self.metadata, dict):
                raise ValueError("manifest must be a JSON object")

        arrays: dict[str, np.ndarray] = {}
        for name in self._ARRAYS:
            array_path = self.path / f"{name}.npy"
            if not array_path.exists():
                raise ValueError(f"bundle is missing {name}.npy")
            arrays[name] = np.load(array_path, mmap_mode="c", allow_pickle=False)
        self._validate_integrity(arrays)

        self.target_values = arrays["target_values"]
        self.target_mask = arrays["target_mask"]
        self._all_past_only_values = arrays["past_only_values"]
        self._all_past_only_mask = arrays["past_only_mask"]
        self._all_past_future_values = arrays["past_future_values"]
        self._all_past_future_mask = arrays["past_future_mask"]
        self.timestamps = arrays["timestamps"]
        self.dates = arrays["dates"]
        self.session_lengths = arrays["session_lengths"]

        available_past_only = tuple(self.metadata.get("past_only_features", ()))
        available_past_future = tuple(self.metadata.get("past_future_features", ()))
        self.past_only_features = tuple(past_only_features)
        self.past_future_features = tuple(past_future_features)
        self._past_only_indices = self._feature_indices(
            self.past_only_features, available_past_only, "past-only"
        )
        self._past_future_indices = self._feature_indices(
            self.past_future_features, available_past_future, "past-future"
        )
        self.num_variates = 1 + len(self.past_only_features) + len(self.past_future_features)
        if self.num_variates > max_variates or self.num_variates > 32:
            raise ValueError(
                f"selected {self.num_variates} variates exceeds limit {min(max_variates, 32)}"
            )

        self.context_min = context_min
        self.context_max = context_max
        self.horizon_length = horizon_length
        self.stride = stride
        self.require_complete_future = require_complete_future
        self._validate_metadata(
            expected_split=expected_split,
            expected_dataset_id=expected_dataset_id,
            expected_product=expected_product,
            expected_target_name=expected_target_name,
            expected_target_unit=expected_target_unit,
            expected_target_price_source=expected_target_price_source,
            expected_target_return_type=expected_target_return_type,
            expected_target_timestamp_semantics=expected_target_timestamp_semantics,
            expected_target_availability_lag_minutes=(expected_target_availability_lag_minutes),
            expected_target_missing_policy=expected_target_missing_policy,
            expected_frequency_minutes=expected_frequency_minutes,
            expected_session_minutes=expected_session_minutes,
            expected_dates_path=expected_dates_path,
            require_metadata=require_metadata,
        )

        day_indices: list[int] = []
        anchor_indices: list[int] = []
        context_lengths: list[int] = []
        for day_index, session_length_value in enumerate(self.session_lengths):
            session_length = int(session_length_value)
            for anchor in range(
                self.context_min - 1,
                session_length - self.horizon_length,
                self.stride,
            ):
                if self.target_mask[day_index, anchor]:
                    continue
                future = slice(
                    anchor + 1,
                    anchor + 1 + self.horizon_length,
                )
                if self.require_complete_future and self.target_mask[day_index, future].any():
                    continue
                if self.target_mask[day_index, future].all():
                    continue
                day_indices.append(day_index)
                anchor_indices.append(anchor)
                context_lengths.append(min(self.context_max, anchor + 1))
        if not day_indices:
            raise ValueError("bundle produces no valid intraday windows")
        self._day_indices = np.asarray(day_indices, dtype=np.int32)
        self._anchor_indices = np.asarray(anchor_indices, dtype=np.int16)
        self.context_lengths = np.asarray(context_lengths, dtype=np.int16)

    @staticmethod
    def _feature_indices(
        selected: tuple[str, ...],
        available: tuple[str, ...],
        kind: str,
    ) -> np.ndarray:
        if len(selected) != len(set(selected)):
            raise ValueError(f"selected {kind} features contain duplicates")
        lookup = {name: index for index, name in enumerate(available)}
        missing = [name for name in selected if name not in lookup]
        if missing:
            raise ValueError(f"unknown {kind} features: {missing}")
        return np.asarray([lookup[name] for name in selected], dtype=np.int64)

    def _validate_integrity(self, arrays: dict[str, np.ndarray]) -> None:
        if self.metadata.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format={self.metadata.get('format')!r}")
        if self.metadata.get("format_version") != BUNDLE_VERSION:
            raise ValueError(f"unsupported format_version={self.metadata.get('format_version')!r}")
        expected_dtypes = {
            "target_values": np.dtype(np.float32),
            "target_mask": np.dtype(np.bool_),
            "past_only_values": np.dtype(np.float32),
            "past_only_mask": np.dtype(np.bool_),
            "past_future_values": np.dtype(np.float32),
            "past_future_mask": np.dtype(np.bool_),
            "timestamps": np.dtype(np.int64),
            "dates": np.dtype(np.int32),
            "session_lengths": np.dtype(np.int16),
        }
        for name, dtype in expected_dtypes.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f"{name} dtype={arrays[name].dtype}, expected {dtype}")

        target_shape = arrays["target_values"].shape
        if len(target_shape) != 2:
            raise ValueError("target_values must have shape (days, session_minutes)")
        days, width = target_shape
        expected_shapes = {
            "target_mask": target_shape,
            "past_only_values": (
                days,
                len(self.metadata.get("past_only_features", ())),
                width,
            ),
            "past_only_mask": (
                days,
                len(self.metadata.get("past_only_features", ())),
                width,
            ),
            "past_future_values": (
                days,
                len(self.metadata.get("past_future_features", ())),
                width,
            ),
            "past_future_mask": (
                days,
                len(self.metadata.get("past_future_features", ())),
                width,
            ),
            "timestamps": target_shape,
            "dates": (days,),
            "session_lengths": (days,),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise ValueError(f"{name} shape={arrays[name].shape}, expected {expected_shape}")
        schema = self.metadata.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("manifest schema must be an object")
        for name, array in arrays.items():
            expected = [str(array.dtype), *array.shape]
            if schema.get(name) != expected:
                raise ValueError(
                    f"manifest schema for {name}={schema.get(name)!r}, expected {expected}"
                )
        lengths = arrays["session_lengths"].astype(np.int64)
        if np.any(lengths <= 0) or np.any(lengths > width):
            raise ValueError("session_lengths must be within the stored session width")
        if not np.all(arrays["dates"][1:] > arrays["dates"][:-1]):
            raise ValueError("dates must be strictly increasing")
        for day, length_value in enumerate(lengths):
            length = int(length_value)
            if not np.isfinite(arrays["target_values"][day, :length]).all():
                raise ValueError("target_values must store finite fill values for masked minutes")
            for values_name in ("past_only_values", "past_future_values"):
                if not np.isfinite(arrays[values_name][day, :, :length]).all():
                    raise ValueError(f"{values_name} must store finite fill values")
            timestamps = arrays["timestamps"][day, :length]
            if length > 1 and not np.all(np.diff(timestamps) == 60_000_000_000):
                raise ValueError("timestamps must form an exact 1-minute grid per day")
            if np.any(timestamps <= 0):
                raise ValueError("real-minute timestamps must be positive")

    def _validate_metadata(
        self,
        *,
        expected_split: str | None,
        expected_dataset_id: str | None,
        expected_product: str | None,
        expected_target_name: str | None,
        expected_target_unit: str | None,
        expected_target_price_source: str | None,
        expected_target_return_type: str | None,
        expected_target_timestamp_semantics: str | None,
        expected_target_availability_lag_minutes: int | None,
        expected_target_missing_policy: str | None,
        expected_frequency_minutes: int | None,
        expected_session_minutes: int | None,
        expected_dates_path: str | Path | None,
        require_metadata: bool,
    ) -> None:
        if not require_metadata:
            return
        expected = {
            "split": expected_split,
            "dataset_id": expected_dataset_id,
            "product": expected_product,
            "target_name": expected_target_name,
            "frequency_minutes": expected_frequency_minutes,
            "session_minutes": expected_session_minutes,
        }
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if value is not None and self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"bundle metadata mismatch: {mismatches}")
        target = self.metadata.get("target")
        expected_target = {
            "unit": expected_target_unit,
            "price_source": expected_target_price_source,
            "return_type": expected_target_return_type,
            "timestamp_semantics": expected_target_timestamp_semantics,
            "availability_lag_minutes": expected_target_availability_lag_minutes,
            "missing_policy": expected_target_missing_policy,
        }
        target_mismatches = {
            key: (target.get(key) if isinstance(target, dict) else None, value)
            for key, value in expected_target.items()
            if value is not None and (not isinstance(target, dict) or target.get(key) != value)
        }
        if target_mismatches:
            raise ValueError(f"bundle target-definition mismatch: {target_mismatches}")
        if expected_dates_path is not None:
            expected_dates = read_dates(expected_dates_path)
            actual_dates = tuple(int(value) for value in self.dates)
            if actual_dates != expected_dates:
                raise ValueError("bundle dates do not match configured date list")
            expected_hash = _date_file_sha256(expected_dates_path)
            if self.metadata.get("date_file_sha256") != expected_hash:
                raise ValueError("bundle date-file hash does not match configured list")

    def __len__(self) -> int:
        return len(self._day_indices)

    def padded_context_length(self, index: int, patch_length: int) -> int:
        return math.ceil(int(self.context_lengths[index]) / patch_length) * patch_length

    def __getitem__(self, index: int) -> WindowSample:
        day = int(self._day_indices[index])
        anchor = int(self._anchor_indices[index])
        context_length = int(self.context_lengths[index])
        context_start = anchor - context_length + 1
        future_start = anchor + 1
        future_stop = future_start + self.horizon_length

        target_context = self.target_values[day, context_start:future_start][None, :]
        target_context_mask = self.target_mask[day, context_start:future_start][None, :]
        past_values = self._all_past_only_values[
            day, self._past_only_indices, context_start:future_start
        ]
        past_mask = self._all_past_only_mask[
            day, self._past_only_indices, context_start:future_start
        ]
        context_values = np.concatenate((target_context, past_values), axis=0)
        context_mask = np.concatenate((target_context_mask, past_mask), axis=0)

        known_values = self._all_past_future_values[
            day, self._past_future_indices, context_start:future_stop
        ]
        known_mask = self._all_past_future_mask[
            day, self._past_future_indices, context_start:future_stop
        ]
        covariate_future_values = self._all_past_only_values[
            day, self._past_only_indices, future_start:future_stop
        ]
        covariate_future_mask = self._all_past_only_mask[
            day, self._past_only_indices, future_start:future_stop
        ]
        future_values = self.target_values[day, future_start:future_stop]
        future_mask = self.target_mask[day, future_start:future_stop]
        unknown_future_values = np.concatenate(
            (future_values[None, :], covariate_future_values),
            axis=0,
        )
        unknown_future_mask = np.concatenate(
            (future_mask[None, :], covariate_future_mask),
            axis=0,
        )
        valid_context = target_context[~target_context_mask]
        volatility = float(np.std(valid_context, dtype=np.float64))
        return {
            "context_values": torch.from_numpy(np.array(context_values, copy=True)),
            "context_mask": torch.from_numpy(np.array(context_mask, copy=True)),
            "past_future_values": torch.from_numpy(np.array(known_values, copy=True)),
            "past_future_mask": torch.from_numpy(np.array(known_mask, copy=True)),
            "unknown_future_values": torch.from_numpy(np.array(unknown_future_values, copy=True)),
            "unknown_future_mask": torch.from_numpy(np.array(unknown_future_mask, copy=True)),
            "context_length": context_length,
            "timestamp": int(self.timestamps[day, anchor]),
            "date": int(self.dates[day]),
            "minute_index": anchor,
            "last_return": float(target_context[0, -1]),
            "context_volatility": volatility,
        }


class ContextBucketBatchSampler(Sampler[list[int]]):
    """Groups samples by patch-rounded context length before batching."""

    def __init__(
        self,
        dataset: IntradayWindowDataset,
        *,
        batch_size: int,
        patch_length: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0 or patch_length <= 0:
            raise ValueError("batch_size and patch_length must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_length = patch_length
        self.shuffle = shuffle
        self.generator = generator
        self.drop_last = drop_last
        buckets: dict[int, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            buckets[dataset.padded_context_length(index, patch_length)].append(index)
        self._buckets = dict(sorted(buckets.items()))

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for indices in self._buckets.values():
            ordered = list(indices)
            if self.shuffle:
                permutation = torch.randperm(len(ordered), generator=self.generator).tolist()
                ordered = [ordered[index] for index in permutation]
            stop = (
                len(ordered) - (len(ordered) % self.batch_size) if self.drop_last else len(ordered)
            )
            for start in range(0, stop, self.batch_size):
                batch = ordered[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle and batches:
            permutation = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[index] for index in permutation]
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self._buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


def collate_intraday_windows(
    samples: list[WindowSample],
    *,
    patch_length: int = 32,
) -> WindowBatch:
    """Left-pads a context bucket only to its next patch boundary."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    horizon = int(samples[0]["unknown_future_values"].shape[-1])
    context_variates = int(samples[0]["context_values"].shape[0])
    future_variates = int(samples[0]["past_future_values"].shape[0])
    padded_context = max(
        math.ceil(sample["context_length"] / patch_length) * patch_length for sample in samples
    )
    batch_size = len(samples)
    context_values = torch.zeros(batch_size, context_variates, padded_context, dtype=torch.float32)
    context_mask = torch.ones_like(context_values, dtype=torch.bool)
    context_padding_mask = torch.ones(batch_size, padded_context, dtype=torch.bool)
    known_values = torch.zeros(
        batch_size,
        future_variates,
        padded_context + horizon,
        dtype=torch.float32,
    )
    known_mask = torch.ones_like(known_values, dtype=torch.bool)

    for index, sample in enumerate(samples):
        length = sample["context_length"]
        left = padded_context - length
        context_values[index, :, left:] = sample["context_values"]
        context_mask[index, :, left:] = sample["context_mask"]
        context_padding_mask[index, left:] = False
        known_values[index, :, left:padded_context] = sample["past_future_values"][:, :length]
        known_values[index, :, padded_context:] = sample["past_future_values"][:, length:]
        known_mask[index, :, left:padded_context] = sample["past_future_mask"][:, :length]
        known_mask[index, :, padded_context:] = sample["past_future_mask"][:, length:]

    return {
        "context_values": context_values,
        "context_mask": context_mask,
        "context_padding_mask": context_padding_mask,
        "past_future_values": known_values,
        "past_future_mask": known_mask,
        "unknown_future_values": torch.stack(
            [sample["unknown_future_values"] for sample in samples]
        ),
        "unknown_future_mask": torch.stack([sample["unknown_future_mask"] for sample in samples]),
        "context_lengths": torch.tensor(
            [sample["context_length"] for sample in samples], dtype=torch.int16
        ),
        "timestamps": torch.tensor([sample["timestamp"] for sample in samples], dtype=torch.int64),
        "dates": torch.tensor([sample["date"] for sample in samples], dtype=torch.int32),
        "minute_indices": torch.tensor(
            [sample["minute_index"] for sample in samples], dtype=torch.int16
        ),
        "last_returns": torch.tensor(
            [sample["last_return"] for sample in samples], dtype=torch.float32
        ),
        "context_volatility": torch.tensor(
            [sample["context_volatility"] for sample in samples],
            dtype=torch.float32,
        ),
    }
