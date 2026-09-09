from __future__ import annotations

import json
from pathlib import Path

import pytest

from timesfm_ft.config import DataConfig, ExperimentConfig


def test_e0_e5_matrix_matches_training_plan():
    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    configs = {path.stem: ExperimentConfig.from_json(path) for path in root.glob("e*.json")}
    assert set(configs) == {
        "e0_return_only",
        "e1_past_only",
        "e2_past_future",
        "e3_context_128",
        "e4_context_256",
        "e5_context_min_96",
    }
    for config in configs.values():
        assert config.data.frequency_minutes == 1
        assert config.data.horizon_length == 64
        assert config.data.stride == 1
        assert config.objective.name == "pinball"
        assert config.trainer.checkpoint_metric == "mean_pinball"
        assert config.model.disable_iterative_cpm_revin
    assert configs["e0_return_only"].data.past_only_features == ()
    assert len(configs["e1_past_only"].data.past_only_features) == 18
    assert len(configs["e2_past_future"].data.past_future_features) == 3
    assert configs["e3_context_128"].data.context_max == 128
    assert configs["e4_context_256"].data.context_max == 256
    assert configs["e5_context_min_96"].data.context_min == 96


def test_relative_paths_resolve_from_leaf_config(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig.from_json(root / "configs/experiments/e2_past_future.json")
    assert Path(config.data.train_path) == root / "data/intraday-1min/train"
    assert Path(config.data.train_dates_path) == root / "configs/splits/dates-train.txt"


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


def test_config_inheritance_rejects_cycles(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"extends": "second.json"}))
    second.write_text(json.dumps({"extends": "first.json"}))
    with pytest.raises(ValueError, match="cyclic"):
        ExperimentConfig.from_json(first)
