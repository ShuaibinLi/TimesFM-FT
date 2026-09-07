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
