"""Ridge and optional LightGBM baselines on the same audited samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from timesfm_ft.data import IntradayWindowDataset


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2:
        return None
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def summarize_samples(
    dataset: IntradayWindowDataset,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Converts variable-length contexts into a fixed, causal baseline matrix."""

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for index in range(len(dataset)):
        sample = dataset[index]
        context = sample["context_values"].numpy()
        context_mask = sample["context_mask"].numpy()
        known = sample["past_future_values"].numpy()
        known_mask = sample["past_future_mask"].numpy()
        features: list[float] = []
        feature_names: list[str] = []
        context_names = ("target_return", *dataset.past_only_features)
        for variate, name in enumerate(context_names):
            valid = context[variate, ~context_mask[variate]]
            features.extend(
                (
                    float(valid[-1]),
                    float(valid.mean()),
                    float(valid.std()),
                    float(valid.sum()),
                )
            )
            feature_names.extend((f"{name}_last", f"{name}_mean", f"{name}_std", f"{name}_sum"))
        context_length = sample["context_length"]
        for variate, name in enumerate(dataset.past_future_features):
            valid_context = known[variate, :context_length][~known_mask[variate, :context_length]]
            valid_future = known[variate, context_length:][~known_mask[variate, context_length:]]
            features.extend(
                (
                    float(valid_context[-1]),
                    float(valid_future.mean()),
                    float(valid_future[-1]),
                )
            )
            feature_names.extend((f"{name}_now", f"{name}_future_mean", f"{name}_future_last"))
        features.extend(
            (
                context_length / dataset.context_max,
                sample["minute_index"] / max(int(dataset.session_lengths.max()) - 1, 1),
            )
        )
        feature_names.extend(("context_fraction", "session_fraction"))
        row = np.asarray(features, dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError(f"baseline row {index} contains non-finite values")
        rows.append(row)
        targets.append(sample["future_values"].numpy())
        current_names = tuple(feature_names)
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("baseline feature schema changed between samples")
    if names is None:
        raise ValueError("dataset contains no samples")
    return np.stack(rows), np.stack(targets), names


def sample_ids(dataset: IntradayWindowDataset) -> np.ndarray:
    return np.asarray(
        [
            f"{sample['date']}:{sample['timestamp']}"
            for sample in (dataset[index] for index in range(len(dataset)))
        ]
    )


class RidgeBaseline:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.intercept: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeBaseline:
        self.mean = features.mean(axis=0, dtype=np.float64)
        self.scale = features.std(axis=0, dtype=np.float64)
        self.scale[self.scale == 0] = 1.0
        normalized = (features - self.mean) / self.scale
        self.intercept = targets.mean(axis=0, dtype=np.float64)
        centered_target = targets - self.intercept
        gram = normalized.T @ normalized
        regularizer = self.alpha * np.eye(gram.shape[0])
        self.coefficients = np.linalg.solve(
            gram + regularizer,
            normalized.T @ centered_target,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if any(
            value is None for value in (self.mean, self.scale, self.coefficients, self.intercept)
        ):
            raise RuntimeError("RidgeBaseline must be fitted before predict")
        assert self.mean is not None
        assert self.scale is not None
        assert self.coefficients is not None
        assert self.intercept is not None
        return ((features - self.mean) / self.scale) @ self.coefficients + self.intercept

    def save(self, path: Path, feature_names: tuple[str, ...]) -> None:
        assert self.mean is not None
        assert self.scale is not None
        assert self.coefficients is not None
        assert self.intercept is not None
        np.savez(
            path,
            alpha=np.asarray(self.alpha),
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
            feature_names=np.asarray(feature_names),
        )


def fit_predict_lightgbm(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, list[Any]]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:
        raise RuntimeError("LightGBM baseline requires `pip install -e '.[baselines]'`") from error
    predictions = np.empty((len(test_features), train_targets.shape[1]), dtype=np.float32)
    models: list[Any] = []
    for horizon in range(train_targets.shape[1]):
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_features, train_targets[:, horizon])
        predictions[:, horizon] = model.predict(test_features)
        models.append(model)
    return predictions, models


def point_forecast_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    horizons: tuple[int, ...],
) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for horizon in horizons:
        prediction = predictions[:, :horizon].sum(axis=1)
        target = targets[:, :horizon].sum(axis=1)
        error = prediction - target
        target_sse = float(np.dot(target, target))
        error_sse = float(np.dot(error, error))
        nonzero = target != 0
        rows.append(
            {
                "horizon_minutes": horizon,
                "samples": len(target),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "ic": _correlation(prediction, target),
                "rank_ic": _correlation(_rank(prediction), _rank(target)),
                "directional_accuracy": (
                    float(np.mean(np.sign(prediction[nonzero]) == np.sign(target[nonzero])))
                    if nonzero.any()
                    else None
                ),
                "oos_r2_vs_zero": (1.0 - error_sse / target_sse if target_sse > 0 else None),
            }
        )
    return rows


def run_baseline(
    train: IntradayWindowDataset,
    test: IntradayWindowDataset,
    *,
    model_name: Literal["ridge", "lightgbm"],
    output_dir: Path,
    horizons: tuple[int, ...],
    ridge_alpha: float = 1.0,
) -> Path:
    train_x, train_y, feature_names = summarize_samples(train)
    test_x, test_y, test_feature_names = summarize_samples(test)
    if feature_names != test_feature_names:
        raise ValueError("train/test baseline feature schemas differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "ridge":
        model = RidgeBaseline(alpha=ridge_alpha).fit(train_x, train_y)
        predictions = model.predict(test_x)
        model.save(output_dir / "ridge_model.npz", feature_names)
    else:
        predictions, models = fit_predict_lightgbm(train_x, train_y, test_x)
        model_dir = output_dir / "lightgbm_models"
        model_dir.mkdir()
        for horizon, model in enumerate(models, start=1):
            model.booster_.save_model(str(model_dir / f"h{horizon:02d}.txt"))
    report = point_forecast_report(predictions, test_y, horizons=horizons)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "features": feature_names,
                "horizons": report,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "predictions.npz",
        predictions=predictions,
        targets=test_y,
        sample_ids=sample_ids(test),
    )
    return output_dir
