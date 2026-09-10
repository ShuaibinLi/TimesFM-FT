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

The production train/validation/test lists contain 476/209/128 chronological,
disjoint sessions with fixed boundaries 2022-11-01–2024-09-30,
2024-10-01–2025-07-31, and 2025-08-01–2026-01-30. No FOMC, minutes,
tariff-week, or other event days are removed. Holiday/half-day sessions remain
in their calendar split with their shorter audited `session_length`; windows
never cross their actual close. Event days should be evaluated as slices,
not silently excluded from training.
`dates-test-202508.txt` is the 21-day August baseline slice and must not be
used for feature selection.

Regenerate the split manifest with
`python scripts/build_rank_selected100_splits.py`. Bundles hash the exact date
file, and training/evaluation reject mismatches.
