from __future__ import annotations

import json

import numpy as np

from timesfm_ft.baselines import RidgeBaseline, run_baseline, summarize_samples
from timesfm_ft.data import IntradayWindowDataset


def _dataset(path, dates_path, split):
    return IntradayWindowDataset(
        path,
        context_min=4,
        context_max=8,
        horizon_length=3,
        stride=1,
        past_only_features=("p1",),
        past_future_features=("tod",),
        expected_split=split,
        expected_dataset_id="test_intraday",
        expected_dates_path=dates_path,
    )


def test_ridge_uses_fixed_causal_summary_and_writes_report(bundle_factory, tmp_path):
    train_path, train_dates = bundle_factory("train", split="train", start_date=20250102)
    test_path, test_dates = bundle_factory("test", split="test", start_date=20250202)
    train = _dataset(train_path, train_dates, "train")
    test = _dataset(test_path, test_dates, "test")
    features, targets, names = summarize_samples(train)
    assert features.shape[0] == len(train)
    assert features.shape[1] == len(names)
    prediction = RidgeBaseline(alpha=1.0).fit(features, targets).predict(features)
    assert prediction.shape == targets.shape
    assert np.isfinite(prediction).all()

    destination = run_baseline(
        train,
        test,
        model_name="ridge",
        output_dir=tmp_path / "ridge",
        horizons=(1, 3),
    )
    report = json.loads((destination / "summary.json").read_text())
    assert report["model"] == "ridge"
    assert [row["horizon_minutes"] for row in report["horizons"]] == [1, 3]
    assert (destination / "ridge_model.npz").exists()
    saved = np.load(destination / "predictions.npz")
    assert len(saved["sample_ids"]) == len(test)
