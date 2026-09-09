from __future__ import annotations

import json
from pathlib import Path

import pytest

from timesfm_ft.config import DataConfig, ExperimentConfig


def test_e0_e8_matrix_matches_training_plan():
    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    configs = {path.stem: ExperimentConfig.from_json(path) for path in root.glob("e*.json")}
    assert set(configs) == {
        "e0_return_only",
        "e1_past_only",
        "e2_past_future",
        "e3_context_128",
        "e4_context_256",
        "e5_context_min_96",
        "e6_l0_pinball",
        "e7_l1_business",
        "e8_l2_auxiliary",
    }
    for config in configs.values():
        assert config.data.frequency_minutes == 1
        assert config.data.horizon_length == 64
        assert config.data.stride == 1
        assert config.trainer.checkpoint_metric == "mean_daily_rank_ic"
        assert config.trainer.checkpoint_horizons == (5, 15, 30, 60)
        assert config.model.disable_iterative_cpm_revin
    assert configs["e0_return_only"].data.past_only_features == ()
    assert len(configs["e1_past_only"].data.past_only_features) == 18
    assert len(configs["e2_past_future"].data.past_future_features) == 3
    assert configs["e3_context_128"].data.context_max == 128
    assert configs["e4_context_256"].data.context_max == 256
    assert configs["e5_context_min_96"].data.context_min == 96
    assert configs["e6_l0_pinball"].objective.name == "l0"
    assert configs["e7_l1_business"].objective.name == "l1"
    assert configs["e7_l1_business"].objective.cumulative_huber_weight == 0.3
    assert configs["e7_l1_business"].objective.cumulative_horizons == (
        5,
        15,
        30,
        60,
    )
    assert configs["e8_l2_auxiliary"].objective.name == "l2"
    assert configs["e8_l2_auxiliary"].objective.auxiliary_weight == 0.05
    assert set(configs["e8_l2_auxiliary"].objective.auxiliary_features) == {
        "realized_vol_15m",
        "spread",
        "volume",
    }


def test_relative_paths_resolve_from_leaf_config(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig.from_json(root / "configs/experiments/e2_past_future.json")
    assert Path(config.data.train_path) == root / "data/intraday-1min/train"
    assert Path(config.data.train_dates_path) == root / "configs/splits/dates-train.txt"


def test_smoke_configs_cover_all_loss_routes():
    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [
        ExperimentConfig.from_json(root / name)
        for name in ("smoke.json", "smoke_l1.json", "smoke_l2.json")
    ]
    assert [config.objective.name for config in configs] == ["l0", "l1", "l2"]
    assert configs[1].objective.cumulative_horizons == (5, 10, 16)
    assert configs[2].objective.auxiliary_features == (
        "realized_vol_15m",
        "market_return_1m",
    )


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


def test_config_inheritance_rejects_cycles(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"extends": "second.json"}))
    second.write_text(json.dumps({"extends": "first.json"}))
    with pytest.raises(ValueError, match="cyclic"):
        ExperimentConfig.from_json(first)
