from __future__ import annotations

from pathlib import Path

from timesfm_ft.config import ExperimentConfig


def test_all_repository_configs_use_structured_sections():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs").glob("*.json")):
        config = ExperimentConfig.from_json(path)
        config.validate()
        assert config.optimizer.adapter_learning_rate > 0
        assert config.optimizer.head_learning_rate > 0
        assert config.optimizer.pretrained_learning_rate > 0
        assert config.trainer.log_every_steps > 0


def test_product_configs_use_hardened_training_contract():
    root = Path(__file__).resolve().parents[1]
    for product, tick_size in (("zn", 0.015625), ("es", 0.25)):
        config = ExperimentConfig.from_json(
            root / "configs" / f"{product}_single_input.json"
        )
        assert config.data.product == product.upper()
        assert config.data.require_metadata
        assert config.data.test_path is not None
        assert config.data.context_length == 256
        assert config.data.horizon_length == 64
        assert config.data.stride == 64
        assert config.data.sampling_interval_seconds == 0.5
        assert config.objective.tick_size == tick_size
        assert config.model.disable_iterative_cpm_revin
        assert config.trainer.batch_size == 32
        assert config.trainer.gradient_accumulation_steps == 2
        assert config.trainer.deterministic
        assert config.trainer.dtype == "bfloat16"
        assert config.trainer.early_stopping_patience == 2
        assert config.trainer.checkpoint_metric == "rmse_ticks"


def test_relative_paths_resolve_from_config_directory(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig.from_json(root / "configs" / "zn_single_input.json")
    assert Path(config.data.train_path) == (
        root / "data" / "zn-wmp-500ms" / "splits" / "train_c256_h64"
    )
    assert Path(config.data.train_dates_path) == (
        root / "configs" / "splits" / "dates-train.txt"
    )
