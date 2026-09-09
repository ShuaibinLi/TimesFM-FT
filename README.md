# TimesFM-FT: 1-minute intraday returns

This repository is a research baseline for forecasting the next 64 one-minute
returns with TimesFM 3. The active task follows
[`timesfm3_1min_intraday_training_plan.md`](timesfm3_1min_intraday_training_plan.md).

The previous 500 ms single-variable Delta-WMP task is frozen under
[`_archived/500ms_delta_wmp/`](_archived/500ms_delta_wmp/). The active package
does not preserve its data, config, metric, or checkpoint contracts.

> TimesFM 3 weights are distributed under Google's non-commercial,
> non-production license. Confirm the upstream license before using a
> checkpoint or derivative outside research.

## Active contract

- frequency: exactly one minute
- target: a frozen, source-provided `return_1m` series
- context: 64 to 192 real intraday minutes by default
- horizon: the next 64 individual one-minute returns
- past-only inputs: selected causal market-state features
- past-future inputs: known calendar values over context plus horizon
- session rule: no sample crosses a trade-date/session boundary
- model budget: target + all covariates must not exceed 32 variates
- objective: Pinball loss over all nine TimesFM quantiles
- split rule: chronological, whole-day train/validation/test partitions

Context is dynamic. A sample with 83 real minutes is grouped into the 96-point
patch bucket and receives 13 masked values on the left. It is not padded to the
configured 192-minute maximum. The default patch buckets are 64, 96, 128, 160,
and 192 minutes.

The model predicts `return_1m[t+1:t+64]`. Cumulative 5/10/20/30/60-minute point
forecasts are sums of the predicted medians. Quantile paths are evaluated per
lead; quantiles are not summed and mislabeled as cumulative quantiles.

## Repository layout

```text
configs/
├── datasets/intraday_1min_schema.json # source columns and frozen target semantics
├── experiments/                    # E0-E5 matrix
├── smoke.json
└── splits/                         # chronological date lists
scripts/
├── prepare_intraday_splits.py
├── select_past_only_features.py
├── make_synthetic_data.py
├── run_baseline.py
├── run_train_nohup.sh
└── run_zero_shot_matrix_nohup.sh
src/timesfm_ft/
├── adapter.py       # differentiable official decode + both covariate classes
├── data.py          # audited session bundles and dynamic context buckets
├── losses.py        # pure Pinball objective
├── metrics.py       # IC/rank IC/calibration/slices/trading proxy
├── baselines.py     # Ridge and optional LightGBM controls
├── trainer.py
└── evaluator.py
```

## Freeze the production data definition first

`configs/datasets/intraday_1min_schema.json` is the explicit boundary between the
model repository and an upstream one-minute Parquet dataset. Before producing
real bundles:

No matching production corpus is currently present in this workspace. The
available WMP data is a 500 ms, roughly 405-minute Treasury session, while the
992-column TiltGate corpus is irregular event/cell data. Neither is silently
resampled or accepted as the plan's neutral 390×1min source.

1. replace `PRIMARY`, `frozen_source_unit`, and `frozen_source_price`;
2. verify every source feature column and family;
3. freeze whether timestamps denote bar start or bar end;
4. verify `return_1m[t]` is available at decision time `t`;
5. declare `availability_lag_minutes` for every past-only source column;
6. version `dataset_id` whenever any of these facts changes.

The preparer deliberately does not recompute the target. It rejects duplicate,
missing, non-minute, non-390-row, or non-finite target sessions.
Non-zero feature availability lags are applied before bundle creation, with the
new leading unavailable rows masked rather than backfilled.
Create new source-derived `configs/splits/dates-{train,val,test}.txt` files as
described in `configs/splits/README.md`; the archived 500 ms lists are not
active defaults.

Expected source layout defaults to:

```text
<source-root>/date=YYYYMMDD/*.parquet
```

Build chronological bundles:

```bash
python scripts/prepare_intraday_splits.py \
  --source-root gs://bucket/frozen-intraday-1min
```

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

Feature selection is deliberately restricted to a bundle declaring
`split=train`:

```bash
python scripts/select_past_only_features.py \
  --bundle data/intraday-1min/train \
  --output outputs/feature-selection/train-only.json
```

The selector ranks incremental candidates by training-period IC/rank IC,
enforces family and correlation caps, and never opens validation/test bundles.

## Experiments

The checked-in matrix isolates one change at a time:

- E0: return only, context 64-192
- E1: E0 plus 18 past-only features
- E2: E1 plus TOD sine/cosine and time-to-close
- E3: E2 with `C_max=128`
- E4: E2 with `C_max=256`
- E5: E2 with `C_min=96`

Run a zero-shot experiment:

```bash
timesfm-eval \
  --config configs/experiments/e2_past_future.json \
  --split test
```

Run the complete zero-shot matrix under `nohup`:

```bash
scripts/run_zero_shot_matrix_nohup.sh
```

Only after the input ablations establish value should adaptation be run:

```bash
scripts/run_train_nohup.sh configs/experiments/e2_past_future.json
```

The default adaptation is head-only. Change `adapter.type` to `lora` only for a
separate, matched experiment. The official TimesFM submodule remains unmodified.

## Required baselines

Ridge and LightGBM consume the same selected variables and chronological split.
They receive causal fixed-length summaries of each dynamic context.

```bash
python scripts/run_baseline.py \
  --config configs/experiments/e2_past_future.json \
  --model ridge

pip install -e '.[baselines]'
python scripts/run_baseline.py \
  --config configs/experiments/e2_past_future.json \
  --model lightgbm
```

## Evaluation outputs

Evaluation writes:

- `summary.json`: overall errors, zero/last-return controls, calibration, and
  the overlapping-signal trading proxy;
- `per_lead.csv`: 1 through 64-minute lead metrics, IC, rank IC, coverage, and
  crossings;
- `cumulative_horizons.csv`: point metrics at configured cumulative horizons;
- `slices.csv`: context-length, session-phase, and volatility slices;
- `predictions.npz`: targets, all quantiles, timestamps, dates, and context
  lengths when enabled.

The trading number is explicitly labeled a research proxy. It is not a
capacity-valid backtest; production evaluation still needs a frozen spread,
fees, slippage, latency, and execution model.

## Smoke and verification

```bash
python scripts/make_synthetic_data.py
pytest
ruff check src tests scripts
```

The synthetic bundle checks dynamic context, past-only/past-future routing,
masking, training, and evaluation plumbing. It is not a forecasting benchmark.
