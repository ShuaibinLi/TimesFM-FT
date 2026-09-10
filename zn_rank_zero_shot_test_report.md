# ZN Rank 1min Full-Test Zero-Shot Report

## Contract

- Test window: 2025-08-01–2026-01-30.
- Event-day exclusions: none.
- Sessions: 128 total; 124×390min, 3×210min, 1×225min.
- Forecast: rolling one-step P50 (`H=1`), no model training or adapter.
- Points: 41,023 unique `(date, target_timestamp)` minutes.
- Full day: 326 points after the minimum 64-minute context.
- Short day: 146 or 161 points.
- Overall IC: Pearson correlation after chronologically concatenating all daily prediction/target vectors.

## Results

```text
metric                  E0 return-only    E1 +selected20    E2 +selected20+TOD
overall IC                  0.01979            0.00360             0.00288
overall RankIC              0.00449            0.01357             0.01300
mean-daily IC              -0.01984            0.00614             0.00934
mean-daily RankIC          -0.01657           -0.00034            -0.00131
mean pinball                0.13199            0.13067             0.13079
MAE                         0.30377            0.30549             0.30573
RMSE                        0.51157            0.51297             0.51341
OOS R² vs zero             -0.02343           -0.02902            -0.03078
directional accuracy       50.00%              49.97%              49.95%
Q10–Q90 coverage           81.62%              79.13%              79.30%
Q10–Q90 width               1.1811             1.0407              1.0459
```

按交易日 block bootstrap（5,000 次）的 overall IC 95% 区间：

```text
E0 rolling   [-0.0250, 0.0675]
E1 rolling   [-0.0196, 0.0246]
E2 rolling   [-0.0207, 0.0248]
E2 block64   [-0.0124, 0.0131]
```

全部区间都包含 0，当前结果只能视为弱信号或无显著信号。

## Interpretation

- By the frozen primary metric, E0 return-only is best:
  `overall IC=0.01979`.
- E1 features reduce overall Pearson IC to 0.00360, although they improve
  RankIC and mean-daily IC.
- E2 time-of-day covariates reduce overall IC further to 0.00288.
- All variants have negative OOS R² versus predicting zero; distribution
  calibration is acceptable, but P50 point magnitude remains weak.
- Current selected20 is not justified for fine-tuning under the overall-IC
  objective. Feature selection must be reworked against train-only one-step
  pooled IC and chronological stability before a pilot.

## predictions.npz

Each full-test output stores:

```text
predictions  [41023, 1, 9]
targets      [41023, 1]
target_mask  [41023, 1]
quantiles    [9]
```

`predictions[i,0,4]` is the P50 forecast for the next 1min return. No target
minute is duplicated in the daily-series artifact.

## Daily-series artifacts

For each E0/E1/E2 output:

```text
evaluation-test/daily-series-lead1/
├── daily_vectors.npz
├── daily_points.csv
└── summary.json
```

`daily_vectors.npz` contains padded `[128,326]` arrays plus per-day lengths
and masks. `daily_points.csv` is the 41,023-row chronological long form.

## Two stitching methods for E2

```text
method                         points   full-day points   overall IC   RankIC
rolling lead1                  41,023        326           0.00288     0.01300
non-overlapping 64 blocks      40,192        320           0.00061     0.00015
```

Rolling lead1 takes the first P50 from every minute origin. Block64 takes five
complete 64-minute segments per full day, so different target minutes use
lead1 through lead64. Both methods keep each target minute unique; block64
leaves the final six full-session minutes uncovered.

## Next gate

1. Keep train-only rolling one-step pooled IC as the feature score.
2. Add chronological subperiod sign/stability constraints; direct train IC
   ranking alone overfits and underperforms E0 on test.
3. Re-run E0/E1/E2 on validation.
4. Start the one-epoch head pilot only if E1 or E2 improves overall IC over E0.
