from __future__ import annotations

import json
from pathlib import Path

import pytest

from timesfm_ft.config import (
    DataConfig,
    ExperimentConfig,
    ObjectiveConfig,
)


def test_active_zn_input_matrix_and_pilot_are_matched():
    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    names = (
        "zn_rank_e0_return_only",
        "zn_rank_e1_selected20",
        "zn_rank_e2_selected20_tod",
        "zn_rank_e2_pilot",
    )
    configs = {name: ExperimentConfig.from_json(root / f"{name}.json") for name in names}
    for config in configs.values():
        assert config.data.frequency_minutes == 1
        assert config.data.horizon_length == 64
        assert config.data.stride == 1
        assert config.data.dataset_id == "zn_rank_selected100_1min_wmid_ticks_v1"
        assert config.data.target_price_source == "WMid"
        assert config.data.target_unit == "ZN_ticks"
        assert config.trainer.checkpoint_metric == "ic"
        assert config.trainer.checkpoint_horizons == (1,)
        assert config.model.disable_iterative_cpm_revin
        assert config.objective.name == "f0_final"
    assert configs["zn_rank_e0_return_only"].data.past_only_features == ()
    assert configs["zn_rank_e0_return_only"].data.past_future_features == ()
    assert len(configs["zn_rank_e1_selected20"].data.past_only_features) == 20
    assert configs["zn_rank_e1_selected20"].data.past_future_features == ()
    assert len(configs["zn_rank_e2_selected20_tod"].data.past_future_features) == 3
    assert configs["zn_rank_e2_pilot"].trainer.epochs == 1


def test_relative_paths_resolve_from_leaf_config(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig.from_json(root / "configs/experiments/zn_rank_e2_selected20_tod.json")
    assert Path(config.data.train_path) == root / "data/zn-rank-selected100-1min/train"
    assert (
        Path(config.data.train_dates_path)
        == root / "configs/splits/zn-rank-selected100/dates-train.txt"
    )


def test_zero_shot_configs_use_the_full_test_bundle_and_one_step_horizon():
    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    for stage in ("e0", "e1", "e2"):
        config = ExperimentConfig.from_json(root / f"zn_rank_{stage}_zero_shot_test.json")
        assert Path(config.data.test_path).name == "test"
        assert Path(config.data.test_dates_path).name == "dates-test.txt"
        assert config.data.horizon_length == 1
        assert config.evaluation.report_horizons == (1,)


def test_owner_frozen_split_counts_and_boundaries():
    root = Path(__file__).resolve().parents[1] / "configs/splits/zn-rank-selected100"
    expected = {
        "train": (476, 20221101, 20240930),
        "val": (209, 20241001, 20250731),
        "test": (128, 20250801, 20260130),
    }
    splits = {}
    for name, (count, first, last) in expected.items():
        values = tuple(
            int(line) for line in (root / f"dates-{name}.txt").read_text().splitlines() if line
        )
        assert (len(values), values[0], values[-1]) == (count, first, last)
        splits[name] = set(values)
    assert not splits["train"] & splits["val"]
    assert not splits["train"] & splits["test"]
    assert not splits["val"] & splits["test"]


def test_variate_budget_and_role_overlap_fail_closed():
    common = {
        "train_path": "train",
        "val_path": "val",
        "require_metadata": False,
        "session_minutes": 390,
    }
    with pytest.raises(ValueError, match="both past-only and past-future|cannot be both"):
        ExperimentConfig(
            data=DataConfig(
                **common,
                past_only_features=("x",),
                past_future_features=("x",),
            )
        ).validate()
    with pytest.raises(ValueError, match="exceeds max_variates"):
        ExperimentConfig(
            data=DataConfig(
                **common,
                max_variates=2,
                past_only_features=("x", "y"),
            )
        ).validate()


def test_unsupported_bar_start_alignment_fails_closed():
    with pytest.raises(ValueError, match="bar_end"):
        ExperimentConfig(
            data=DataConfig(
                train_path="train",
                val_path="val",
                require_metadata=False,
                target_timestamp_semantics="bar_start",
            )
        ).validate()


def test_dense_route_rejects_decode_only_detrending():
    with pytest.raises(ValueError, match="linear_detrending"):
        ExperimentConfig(
            data=DataConfig(
                train_path="train",
                val_path="val",
                require_metadata=False,
            ),
            objective=ObjectiveConfig(name="f0_all"),
        ).validate()


def test_config_inheritance_rejects_cycles(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"extends": "second.json"}))
    second.write_text(json.dumps({"extends": "first.json"}))
    with pytest.raises(ValueError, match="cyclic"):
        ExperimentConfig.from_json(first)
