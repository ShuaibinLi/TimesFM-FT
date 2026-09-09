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
- target: a frozen, source-provided `return_1m` series
- context: 64 to 192 real intraday minutes by default
- horizon: the next 64 individual one-minute returns
- past-only inputs: selected causal market-state features
- past-future inputs: known calendar values over context plus horizon
- session rule: no sample crosses a trade-date/session boundary
- model budget: target + all covariates must not exceed 32 variates
- training routes: F0-final deployment control, F0-all dense shifted
  forecasting, recommended F1 final-business extension, and optional F1-MV
  auxiliary ablation
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
├── datasets/intraday_1min_schema.json # source columns and frozen target semantics
├── experiments/                    # E0-E5 inputs + T0-T3 training routes
├── smoke_t{0,1,2,3}_*.json
└── splits/                         # chronological date lists
scripts/
├── prepare_intraday_splits.py
├── select_past_only_features.py
├── make_synthetic_data.py
├── run_baseline.py
├── run_train_nohup.sh
├── run_training_route_matrix_nohup.sh
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
non-minute, or non-390-row sessions. Missing/halted target minutes are stored as
finite fill values with `target_mask=True`; invalid cutoffs are skipped and all
affected loss terms receive zero weight.
The active v1.3 slicer deliberately accepts only `bar_end` with zero target
availability lag; other semantics fail closed until their decision-time
alignment is specified and tested.
Non-zero feature availability lags are applied before bundle creation, with the
new leading unavailable rows masked rather than backfilled.
Every source Parquet part is content-hashed; the manifest also records the
combined source snapshot and preparer-script hash.
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
At runtime, future-unknown labels use one slot-aligned tensor:
`unknown_future_values[B, 1+V_past_only, H]`, ordered exactly like
`context_values` as `[target | past-only]`; row zero is the business target.

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

The matched training-route matrix is separate:

- T0 / F0-final: differentiable deployment suffix decode, final Pinball
- T1 / F0-all: public low-level full-sequence forward, dense eligible Pinball
- T2 / F1: T1 + `0.3 ×` final-anchor cumulative P50 Huber
- T3 / F1-MV: T2 + `0.05 ×` selected dense past-only auxiliary Pinball

T1–T3 are explicitly labeled pretraining-like downstream engineering routes,
not reconstructions of Google's unpublished recipe. They call the public
inference-specialized Torch `forward()` with full causal sequences,
`patch_cpm_mask=None`, explicit next-64 roll labels, and explicit eligible
anchor masks. Linear detrending is disabled for all matched T0–T3 configs so
the dense final token is numerically checked against deployment `decode()`.
Validation business metrics and checkpoint selection always use deployment
`decode()` predictions; dense final-token parity is rechecked on every
validation batch.
Random CPM, role reassignment, and sequence packing remain deferred research:
the public Torch port does not expose the unpublished training branch needed
to claim official parity.

With stride-1 overlapping windows, one absolute dense origin may be supervised
under several truncated-history views. This is intentional context
augmentation, not extra independent data; every epoch logs eligible anchors,
unique `(date,timestamp)` origins, and their repeat factor.

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
scripts/run_train_nohup.sh configs/experiments/t0_f0_final.json
```

The default adaptation is head-only. Change `adapter.type` to `lora` only for a
separate, matched experiment. The official TimesFM submodule remains unmodified.

Run the gated T0 → T1 → T2 → T3 comparison under `nohup`:

```bash
scripts/run_training_route_matrix_nohup.sh
```

F1 cumulative scales at 5/15/30/60 minutes and F1-MV feature scales are fitted
only from the declared training bundle. Their values, method, date-list hash,
feature-schema/manifest hashes, valid counts, raw estimates, explicit fallback,
and state fingerprint are written to `loss_scales.json` and embedded in every
checkpoint. Past-future rows are structurally excluded from both
`forward_unknown()` and `forward_dense()` objective outputs and never enter
forecast supervision.

F0 Pinball remains in the frozen target unit exactly as specified by v1.3,
whereas F1/F1-MV add normalized components. Consequently `0.3` and `0.05` are
unit-specific starting weights, not portable constants: changing ticks/bps/log
units requires a new dataset ID and validation ablation.

Production configs select checkpoints by the mean validation daily rank IC
across 5/15/30/60 minutes, not by total loss. History and
checkpoint metadata retain IC, rank IC, direction, prediction deciles,
calibration, net utility, turnover, and drawdown for the final joint decision.

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
- `per_lead.csv`: 1 through 64-minute lead metrics, daily IC/rank IC,
  Q10–Q90 coverage/width, and crossings;
- `cumulative_horizons.csv`: point metrics, daily IC/rank IC, direction,
  conditional means, and prediction-decile monotonicity;
- `slices.csv`: context-length, session-phase, and volatility slices;
- `predictions.npz`: targets, all quantiles, timestamps, dates, and context
  lengths when enabled.

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
  --config configs/smoke_t0_final.json
python scripts/smoke_real_checkpoint.py \
  --config configs/smoke_t3_f1_mv.json --minimum-context 96
```

The synthetic bundle checks dynamic context, past-only/past-future routing,
masking, training, and evaluation plumbing. It is not a forecasting benchmark.
