from __future__ import annotations

from pathlib import Path


def _dates(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_repository_splits_are_disjoint_sorted_and_chronological():
    root = Path(__file__).resolve().parents[1] / "configs" / "splits"
    train = _dates(root / "dates-train.txt")
    val = _dates(root / "dates-val.txt")
    test = _dates(root / "dates-test.txt")

    assert (len(train), len(val), len(test)) == (445, 189, 120)
    assert train == sorted(set(train))
    assert val == sorted(set(val))
    assert test == sorted(set(test))
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert train[-1] < val[0] < val[-1] < test[0]
