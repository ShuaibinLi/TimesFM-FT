from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_past_only_features",
    ROOT / "scripts/select_past_only_features.py",
)
assert SPEC and SPEC.loader
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def test_feature_selection_rejects_high_missing_candidate(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "split": "train",
                "past_only_features": ["complete", "sparse"],
                "past_only_families": {
                    "complete": "state",
                    "sparse": "state",
                },
            }
        )
    )
    np.save(bundle / "target_values.npy", np.arange(8, dtype=np.float32)[None])
    np.save(bundle / "target_mask.npy", np.zeros((1, 8), dtype=np.bool_))
    np.save(
        bundle / "past_only_values.npy",
        np.asarray([[[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 7]]], dtype=np.float32),
    )
    masks = np.zeros((1, 2, 8), dtype=np.bool_)
    masks[0, 1, :7] = True
    np.save(bundle / "past_only_mask.npy", masks)
    np.save(bundle / "session_lengths.npy", np.asarray([8], dtype=np.int16))

    output = tmp_path / "selection.json"
    selection.select_features(
        bundle,
        output=output,
        context_min=2,
        horizons=(1,),
        stride=1,
        limit=1,
        correlation_limit=0.9,
        family_limit=2,
        max_missing_rate=0.2,
    )
    result = json.loads(output.read_text())
    assert result["selected_features"] == ["complete"]
    assert result["max_missing_rate"] == 0.2
