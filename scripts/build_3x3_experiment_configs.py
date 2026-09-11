#!/usr/bin/env python3
"""Build resolved 3x3 pilot configs and select per-input winners."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from timesfm_ft.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "outputs" / "zn-rank-3x3"
RUNTIME_CONFIGS = GRID / "runtime-configs"
STAGE_CONFIGS = {
    "e0": ROOT / "configs" / "experiments" / "zn_rank_e0_return_only.json",
    "e1": ROOT / "configs" / "experiments" / "zn_rank_e1_selected20.json",
    "e2": ROOT / "configs" / "experiments" / "zn_rank_e2_selected20_tod.json",
}
MODES = ("head", "lora", "full")
LOSS_VARIANTS = {
    "p1": {
        "name": "f0_lead1",
        "return_pinball_weight": 0.5,
        "lead1_pinball_weight": 0.5,
        "correlation_weight": 0.0,
        "cumulative_huber_weight": 0.0,
        "cumulative_horizons": [],
    },
    "p2": {
        "name": "f0_lead1",
        "return_pinball_weight": 0.475,
        "lead1_pinball_weight": 0.475,
        "correlation_weight": 0.0,
        "cumulative_huber_weight": 0.05,
        "cumulative_horizons": [1],
    },
    "p3": {
        "name": "f0_lead1",
        "return_pinball_weight": 0.45,
        "lead1_pinball_weight": 0.45,
        "correlation_weight": 0.05,
        "cumulative_huber_weight": 0.05,
        "cumulative_horizons": [1],
    },
}


def _mode_config(base: dict[str, Any], mode: str, *, epochs: int, output: Path) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["adapter"]["type"] = mode
    config["adapter"]["dropout"] = 0.0
    config["trainer"]["epochs"] = epochs
    config["trainer"]["early_stopping_patience"] = 1 if epochs == 1 else 2
    config["trainer"]["output_dir"] = str(output)
    if mode == "full":
        config["trainer"]["batch_size"] = 4
        config["trainer"]["gradient_accumulation_steps"] = 16
    else:
        config["trainer"]["batch_size"] = 32
        config["trainer"]["gradient_accumulation_steps"] = 2
    return config


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_pilots() -> None:
    for stage, source in STAGE_CONFIGS.items():
        base = ExperimentConfig.from_json(source).to_dict()
        for mode in MODES:
            payload = _mode_config(
                base,
                mode,
                epochs=1,
                output=GRID / "pilots" / stage / mode,
            )
            _write(RUNTIME_CONFIGS / f"{stage}-{mode}-pilot.json", payload)
    print(f"wrote 9 pilot configs under {RUNTIME_CONFIGS}")


def _candidate_metrics(directory: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted((directory / "step-checkpoints").glob("*/adapter_config.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "checkpoint": str(path.parent),
                "metric": float(payload["metric"]),
                "kind": "step",
            }
        )
    for name in ("best", "last"):
        path = directory / name / "adapter_config.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("best_metric", payload.get("metric"))
            if value is not None:
                candidates.append(
                    {
                        "checkpoint": str(path.parent),
                        "metric": float(value),
                        "kind": name,
                    }
                )
    if not candidates:
        raise FileNotFoundError(f"no evaluated checkpoints found in {directory}")
    return candidates


def _select_tuning_mode(stage: str) -> tuple[str, dict[str, Any]]:
    mode_rows = {}
    for mode in MODES:
        candidates = _candidate_metrics(GRID / "pilots" / stage / mode)
        best = max(candidates, key=lambda row: row["metric"])
        mode_rows[mode] = {"best": best, "all": candidates}
    winner = max(MODES, key=lambda mode: mode_rows[mode]["best"]["metric"])
    return winner, mode_rows


def build_winners(stages: tuple[str, ...] | None = None) -> None:
    selected_stages = stages or tuple(STAGE_CONFIGS)
    manifest: dict[str, Any] = {
        "selection_metric": "validation rolling-one-step pooled Pearson IC",
        "pilot_epochs": 1,
        "winner_total_epochs": 3,
        "winner_initialization": "resume epoch-1 last training state from selected mode",
        "stages": {},
    }
    for stage in selected_stages:
        source = STAGE_CONFIGS[stage]
        winner, mode_rows = _select_tuning_mode(stage)
        base = ExperimentConfig.from_json(source).to_dict()
        payload = _mode_config(
            base,
            winner,
            epochs=3,
            output=GRID / "winners-3epoch" / stage,
        )
        resume_path = GRID / "pilots" / stage / winner / "last" / "training_state.pt"
        payload["trainer"]["resume_from"] = str(resume_path)
        config_path = RUNTIME_CONFIGS / f"{stage}-winner-{winner}-3epoch.json"
        _write(config_path, payload)
        manifest["stages"][stage] = {
            "winner_mode": winner,
            "winner_metric": mode_rows[winner]["best"]["metric"],
            "winner_pilot_checkpoint": mode_rows[winner]["best"]["checkpoint"],
            "resume_from": str(resume_path),
            "three_epoch_config": str(config_path),
            "modes": mode_rows,
        }
    destination = (
        GRID / "winner-selection" / f"{selected_stages[0]}.json"
        if len(selected_stages) == 1
        else GRID / "winner-selection.json"
    )
    _write(destination, manifest)
    print(json.dumps(manifest["stages"], indent=2))


def build_loss_configs(stage: str) -> None:
    mode, mode_rows = _select_tuning_mode(stage)
    base = ExperimentConfig.from_json(STAGE_CONFIGS[stage]).to_dict()
    manifest = {
        "stage": stage,
        "selection_metric": "validation rolling-one-step pooled Pearson IC",
        "winner_mode": mode,
        "winner_metric": mode_rows[mode]["best"]["metric"],
        "winner_pilot_checkpoint": mode_rows[mode]["best"]["checkpoint"],
        "variants": {"p0": {"source": str(GRID / "pilots" / stage / mode)}},
    }
    for variant, objective in LOSS_VARIANTS.items():
        payload = _mode_config(
            base,
            mode,
            epochs=1,
            output=GRID / "loss-ablation" / stage / variant,
        )
        payload["objective"].update(objective)
        path = RUNTIME_CONFIGS / f"{stage}-{mode}-{variant}-loss-pilot.json"
        _write(path, payload)
        manifest["variants"][variant] = {
            "config": str(path),
            "output": payload["trainer"]["output_dir"],
            "objective": payload["objective"],
        }
    _write(GRID / "loss-ablation" / stage / "tuning-winner.json", manifest)
    print(json.dumps(manifest, indent=2))


def build_loss_winner(stage: str) -> None:
    tuning = json.loads(
        (GRID / "loss-ablation" / stage / "tuning-winner.json").read_text(encoding="utf-8")
    )
    mode = tuning["winner_mode"]
    candidates = {
        "p0": _candidate_metrics(GRID / "pilots" / stage / mode),
        **{
            variant: _candidate_metrics(GRID / "loss-ablation" / stage / variant)
            for variant in LOSS_VARIANTS
        },
    }
    best_by_variant = {
        variant: max(rows, key=lambda row: row["metric"]) for variant, rows in candidates.items()
    }
    winner = max(best_by_variant, key=lambda variant: best_by_variant[variant]["metric"])
    base = ExperimentConfig.from_json(STAGE_CONFIGS[stage]).to_dict()
    payload = _mode_config(
        base,
        mode,
        epochs=3,
        output=GRID / "winners-3epoch" / stage,
    )
    if winner != "p0":
        payload["objective"].update(LOSS_VARIANTS[winner])
        resume_path = GRID / "loss-ablation" / stage / winner / "last" / "training_state.pt"
    else:
        resume_path = GRID / "pilots" / stage / mode / "last" / "training_state.pt"
    payload["trainer"]["resume_from"] = str(resume_path)
    config_path = RUNTIME_CONFIGS / f"{stage}-winner-{mode}-{winner}-3epoch.json"
    _write(config_path, payload)
    result = {
        "stage": stage,
        "winner_mode": mode,
        "winner_loss": winner,
        "winner_metric": best_by_variant[winner]["metric"],
        "best_by_variant": best_by_variant,
        "resume_from": str(resume_path),
        "three_epoch_config": str(config_path),
    }
    _write(GRID / "winner-selection" / f"{stage}.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("pilots", "winners", "losses", "loss-winner"),
    )
    parser.add_argument("--stage", choices=tuple(STAGE_CONFIGS))
    args = parser.parse_args()
    if args.action == "pilots":
        build_pilots()
    elif args.action == "winners":
        build_winners((args.stage,) if args.stage else None)
    elif args.action == "losses":
        if args.stage is None:
            parser.error("losses requires --stage")
        build_loss_configs(args.stage)
    else:
        if args.stage is None:
            parser.error("loss-winner requires --stage")
        build_loss_winner(args.stage)


if __name__ == "__main__":
    main()
