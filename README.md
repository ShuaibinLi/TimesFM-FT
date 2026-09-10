# TimesFM-FT: 1-minute intraday returns

This repository is a research baseline for forecasting the next 64 one-minute
returns with TimesFM 3. The active task follows
[`timesfm3_1min_intraday_training_plan_v1.3.md`](timesfm3_1min_intraday_training_plan_v1.3.md)
and [`timesfm3_training_route_deep_dive.md`](timesfm3_training_route_deep_dive.md).

The previous 500 ms single-variable Delta-WMP task is frozen under
[`_archived/500ms_delta_wmp/`](_archived/500ms_delta_wmp/). The active package
does not preserve its data, config, metric, or checkpoint contracts.

> TimesFM 3 weights are distributed under Google's non-commercial,
> non-production license. Confirm the upstream license before using a
> checkpoint or derivative outside research.

## Active contract

- frequency: exactly one minute
- target: trailing 1min WMid displacement in ZN ticks, derived on the frozen grid
- context: 64 to 192 real intraday minutes by default
- horizon: the next 64 individual one-minute returns
- past-only inputs: selected causal market-state features
- past-future inputs: known calendar values over context plus horizon
- session rule: no sample crosses a trade-date/session boundary
- model budget: target + all covariates must not exceed 32 variates
- active training route: F0-final head-only control; LoRA/dense objectives
  remain code-level follow-ups without active configs
- split rule: chronological, whole-day train/validation/test partitions

Context is dynamic. A sample with 83 real minutes is grouped into the 96-point
patch bucket and receives 13 masked values on the left. It is not padded to the
configured 192-minute maximum. The default patch buckets are 64, 96, 128, 160,
and 192 minutes.

`context_padding_mask` marks only that shared left padding. Target and
covariate missingness remain in separate per-variate masks, so a halted/missing
target minute does not erase an otherwise available conditioning feature.

The model predicts `return_1m[t+1:t+64]`. Cumulative 5/10/15/20/30/60-minute point
forecasts are sums of the predicted medians. Quantile paths are evaluated per
lead; quantiles are not summed and mislabeled as cumulative quantiles.
For fractional simple returns this sum is a documented small-return
approximation; exact compounding or direct multi-horizon targets belong to the
planned follow-up experiment.

## Repository layout

```text
configs/
├── datasets/zn_rank_selected100_1min.json
├── experiments/                    # active ZN E0/E1/E2, pilot, August baselines
└── splits/zn-rank-selected100/     # production dates + August test slice
scripts/
├── prepare_intraday_splits.py
├── select_past_only_features.py
├── run_baseline.py
├── prepare_rank_selected100_training.sh
├── run_train_nohup.sh
└── run_zero_shot_matrix_nohup.sh
src/timesfm_ft/
├── adapter.py       # differentiable official decode + both covariate classes
├── data.py          # audited session bundles and dynamic context buckets
├── dense.py         # full-sequence tensors, shifted labels, eligible anchors
├── losses.py        # F0/F1 route objectives and train-only scales
├── metrics.py       # IC/rank IC/calibration/slices/trading proxy
├── baselines.py     # Ridge and optional LightGBM controls
├── trainer.py
└── evaluator.py
```

## Freeze the production data definition first

`configs/datasets/zn_rank_selected100_1min.json` is the explicit boundary between the
model repository and an upstream one-minute Parquet dataset. Before producing
real bundles:

The active production-like research corpus is
`../datas/zn_rank_selected100_1min_20221101_20260130`: a causal
`LastEventAtClockTrigger` minute snapshot with `wmid` and 100 ZN-only candidate
features. The preparer trims it to 390 rows/day and derives the historical
tick-unit `return_1m` target variate. The older 500 ms and TiltGate corpora
remain separate and are not silently accepted by this route.

The active freeze is product ZN, target unit `ZN_ticks`, price source `WMid`,
bar-end timestamps, and zero availability lag. Version `dataset_id` whenever
any of these facts changes.

The preparer normally consumes a frozen target column. A schema may instead
declare the audited `trailing_price_difference_ticks` derivation for causal
clock-sampled raw data: it maps pre-boundary `hwts` to exact bar ends and
computes the realized target from `wmid[t] - wmid[t-1]` using a frozen tick
size. It rejects duplicate, non-minute, or non-390-row sessions.
Missing/halted target minutes are stored as finite fill values with
`target_mask=True`; invalid cutoffs are skipped and all affected loss terms
receive zero weight.
The active v1.3 slicer deliberately accepts only `bar_end` with zero target
availability lag; other semantics fail closed until their decision-time
alignment is specified and tested.
Non-zero feature availability lags are applied before bundle creation, with the
new leading unavailable rows masked rather than backfilled.
Every source Parquet part is content-hashed; the manifest also records the
combined source snapshot and preparer-script hash.
The frozen chronological date lists and their exclusion audit live under
`configs/splits/zn-rank-selected100/`.

Expected source layout defaults to:

```text
<source-root>/date=YYYYMMDD/*.parquet
```

The active ZN selected100 route is reproducible end to end:

```bash
python scripts/build_rank_selected100_schema.py
python scripts/build_rank_selected100_splits.py
scripts/prepare_rank_selected100_training.sh
```

Its raw dump contains 100 candidate past-only columns plus `wmid`. The
preparer derives tick-unit `return_1m` as the target history, builds
chronological 70/15/15 bundles, and the train-only selector reduces the 100
candidates to 20. With three deterministic time covariates, the resulting
TimesFM input has 24 variates and remains under the hard limit of 32.
The split builder retains all 813 raw files but excludes 10 holiday/half-day
sessions that do not provide the frozen 390-minute model grid. Feature
selection also rejects candidates above 5% training missingness before IC and
correlation screening.

Each split is stored once in session-major mmap arrays:

```text
target_values.npy       float32[days, 390]
target_mask.npy         bool[days, 390]
past_only_values.npy    float32[days, features, 390]
past_only_mask.npy      bool[days, features, 390]
past_future_values.npy  float32[days, known_features, 390]
past_future_mask.npy    bool[days, known_features, 390]
timestamps.npy          int64[days, 390]
dates.npy               int32[days]
session_lengths.npy     int16[days]
manifest.json
```

Overlapping model windows are sliced lazily rather than duplicated on disk.
At runtime, future-unknown labels use one slot-aligned tensor:
`unknown_future_values[B, 1+V_past_only, H]`, ordered exactly like
`context_values` as `[target | past-only]`; row zero is the business target.

Feature selection is deliberately restricted to a bundle declaring
`split=train`:

```bash
python scripts/select_past_only_features.py \
  --bundle data/zn-rank-selected100-1min/train \
  --output outputs/feature-selection/zn-rank-selected100.json \
  --limit 20 \
  --max-missing-rate 0.05
```

The selector ranks incremental candidates by training-period IC/rank IC,
enforces family and correlation caps, and never opens validation/test bundles.

## Active experiment configs

`configs/experiments/_base.json` is an internal parent, not a runnable
experiment. The runnable production matrix is:

- `zn_rank_e0_return_only.json`: historical `return_1m` only;
- `zn_rank_e1_selected20.json`: target plus 20 train-only selected features;
- `zn_rank_e2_selected20_tod.json`: E1 plus three deterministic time covariates;
- `zn_rank_e2_pilot.json`: one-epoch E2 head-only training gate.

The frozen August 2025 zero-shot reports use:

- `zn_rank_e0_zero_shot_202508.json`;
- `zn_rank_e1_zero_shot_202508.json`;
- `zn_rank_e2_zero_shot_202508.json`.

Run/reproduce the August zero-shot matrix:

```bash
scripts/run_zero_shot_matrix_nohup.sh
```

Run the one-epoch training gate:

```bash
scripts/run_train_nohup.sh configs/experiments/zn_rank_e2_pilot.json
```

Only after the pilot improves validation mean-daily RankIC should the complete
five-epoch config be launched:

```bash
scripts/run_train_nohup.sh configs/experiments/zn_rank_e2_selected20_tod.json
```

The active objective is F0-final Pinball in ZN tick units. Checkpoints are
selected by mean validation daily RankIC across 5/15/30/60 minutes, not total
loss. LoRA, F0-all, F1, and F1-MV remain implemented research routes but have
no active configs until the head-only input value is established.

## Required baselines

Ridge and LightGBM consume the same selected variables and chronological split.
They receive causal fixed-length summaries of each dynamic context.

```bash
python scripts/run_baseline.py \
  --config configs/experiments/zn_rank_e2_selected20_tod.json \
  --model ridge

pip install -e '.[baselines]'
python scripts/run_baseline.py \
  --config configs/experiments/zn_rank_e2_selected20_tod.json \
  --model lightgbm
```

## Evaluation outputs

Evaluation writes:

- `summary.json`: overall errors, zero/last-return controls, calibration, and
  the overlapping-signal trading proxy;
- `per_lead.csv`: 1 through 64-minute lead metrics, daily IC/rank IC,
  Q10–Q90 coverage/width, and crossings;
- `cumulative_horizons.csv`: point metrics, daily IC/rank IC, direction,
  conditional means, and prediction-decile monotonicity;
- `slices.csv`: context-length, session-phase, and volatility slices;
- `predictions.npz`: targets, all quantiles, timestamps, dates, and context
  lengths when enabled.

The frozen August 2025 zero-shot input-ablation report is
[`zn_rank_zero_shot_baseline_202508_report.md`](zn_rank_zero_shot_baseline_202508_report.md).

“Daily IC” here means time-series correlation across intraday decision windows,
computed within each trade date and then averaged across dates. It is not a
cross-sectional multi-instrument IC.

The trading number is explicitly labeled a research proxy. It is not a
capacity-valid backtest; production evaluation still needs a frozen spread,
fees, slippage, latency, and execution model.

## Smoke and verification

```bash
python scripts/make_synthetic_data.py
pytest
ruff check src tests scripts

python scripts/smoke_real_checkpoint.py \
  --config configs/experiments/zn_rank_e2_pilot.json \
  --minimum-context 96
```

The synthetic bundle checks dynamic context, past-only/past-future routing,
masking, training, and evaluation plumbing. It is not a forecasting benchmark.
