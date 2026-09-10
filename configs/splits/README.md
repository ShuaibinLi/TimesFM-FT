# Active ZN split lists

The active split contract is stored under `zn-rank-selected100/`:

```text
zn-rank-selected100/
├── dates-train.txt
├── dates-val.txt
├── dates-test.txt
├── dates-test-202508.txt
└── manifest.json
```

The production train/validation/test lists contain 562/120/121 chronological,
disjoint full sessions. Ten holiday/half-day raw partitions remain archived
but are excluded because they do not provide the frozen 390-minute grid.
`dates-test-202508.txt` is the 21-day August baseline slice and must not be
used for feature selection.

Regenerate the split manifest with
`python scripts/build_rank_selected100_splits.py`. Bundles hash the exact date
file, and training/evaluation reject mismatches.
