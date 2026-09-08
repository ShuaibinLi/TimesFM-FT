# TimesFM-FT

Research fine-tuning toolkit for TimesFM 3, focused on weighted-mid forecasting
with one data point every 500 ms.

The first supported task uses a historical context window to predict the next
64 points (32 seconds) of one weighted-mid target from 256 context points
(128 seconds). Two aligned input contracts
are supported:

1. single input: weighted-mid history only;
2. multi input: weighted-mid plus past-only covariates, with weighted-mid still
   the only supervised output.

> TimesFM 3 weights are currently distributed under Google's non-commercial,
> non-production license. This repository is for research use. Review the
> upstream model license before using any checkpoint or derivative.

## Why a custom trainer?

The released TimesFM 3 PyTorch implementation is marked inference-only.
`TimesFM3Torch.forward()` returns quantile logits, while `decode()` is decorated
with `torch.no_grad()` and no v3 loss or Trainer is shipped.

`TimesFM3Adapter` calls the undecorated implementation retained by
PyTorch in `decode.__wrapped__`. This keeps the official detrending, CPM,
iterative RevIN, and stitching behavior while enabling autograd. A compatibility
guard fails loudly if the upstream implementation changes.

The toolkit adds:

- memory-mapped production bundles plus NPZ smoke-data validation;
- single-target extraction from multivariate TimesFM output;
- tail-quantile pinball, P50 Huber, and quantile-crossing losses in tick space;
- head-only, native LoRA, partial-unfreeze, and full fine-tuning modes;
- train/validation loops, deterministic loading, early stopping, warmup/cosine
  scheduling, and atomic resumable checkpoints;
- tiny-model tests that verify decode parity and gradient flow.

## Repository layout

```text
TimesFM-FT/
├── 3rdparty/timesfm/        # pinned official upstream submodule
├── configs/                 # matched single/multi experiment configs
├── scripts/
│   ├── make_synthetic_data.py
│   ├── prepare_single_product_splits.py
│   ├── run_zn_zero_shot.py
│   └── smoke_real_checkpoint.py
├── src/timesfm_ft/
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── adapter.py            # all TimesFM-specific loading and adaptation
│   ├── evaluator.py          # official decode + streaming forecast metrics
│   ├── trainer.py
│   └── cli.py
└── tests/
```

The upstream submodule is intentionally not modified and is pinned at
`0df95ae62085a6ac0d0afd1ad40dee2e6c1356ab`.

## Setup

Conda:

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate timesfm-ft
```

Equivalent virtualenv setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For the full 330M checkpoint, use CUDA for real training. Apple Silicon MPS can
run the small head-only smoke test; CPU execution is intended primarily for
unit tests with a reduced model.

## Data contract

Production datasets are memory-mapped directories containing pre-windowed,
chronologically split `.npy` arrays plus `manifest.json`. NPZ remains supported
for small smoke fixtures.

Required arrays:

```text
context_values: float32[S, V, C] or float32[S, C]
future_values:  float32[S, H]
```

Production bundles also require:

```text
timestamps: int64[S]  # forecast cutoff on the audited grid
dates:      int32[S]  # trade date used for split enforcement
manifest.json         # product/cadence/C/H/stride/source/date-list hashes
```

Optional mask arrays use `True` for unavailable/excluded values.

Conventions:

- every adjacent data point is exactly 500 ms apart;
- `C=256` (128 seconds) and `H=64` (32 seconds) in the production configs;
- variate zero must always be weighted-mid;
- `V=1` for the single-input route and `1<V<=32` for the multi-input route;
- variates `1..V-1` are past-only covariates;
- the weighted-mid value at the forecast cutoff must be present;
- `future_values` always contains weighted-mid only;
- train, validation, and test bundles must come from disjoint chronological
  split lists, never random row splits.

Both supplied datasets must share exactly the same timestamps and
`future_values` before A/B model comparisons are accepted.

### ZN/ES chronological splits

The checked-in single-product configs use the canonical non-overlapping date
lists under `configs/splits/`:

- train: 445 days, 2022-11-01 through 2024-09-30;
- validation: 189 days, 2024-10-01 through 2025-07-31;
- test: 120 days, 2025-08-01 through 2026-01-30.

All observations are exactly 500 ms apart. Production windows use 256 context
points (128 seconds), 64 future points (32 seconds), and a 64-point stride.
Data preparation requires GCS application-default credentials. Build both
products' day-safe, memory-mapped bundles without crossing session boundaries:

```bash
gcloud auth application-default login
python scripts/prepare_single_product_splits.py
```

Train the two single-variable models independently:

```bash
timesfm-ft --config configs/zn_single_input.json
timesfm-ft --config configs/es_single_input.json
```

## Contract smoke test

Generate small aligned single/multi datasets:

```bash
python scripts/make_synthetic_data.py
```

This writes ignored files under:

```text
data/single/{train,val,test}.npz
data/multi/{train,val,test}.npz
```

The synthetic values only test plumbing. They are not a forecasting benchmark.

To reproduce the minimal data used by the full-checkpoint smoke test:

```bash
python scripts/make_synthetic_data.py \
  --output-root data/dummy \
  --num-samples 4 \
  --context-length 32 \
  --horizon-length 16 \
  --num-variates 8
```

Then run one head-only epoch for each input route:

```bash
timesfm-ft --config configs/smoke_test.json
timesfm-ft --config configs/smoke_test_multi.json
```

The smoke configs intentionally keep `C=32` and `H=16` to test plumbing
quickly; they are not experiment configs. All production experiment configs
use the 500 ms, `C=256` (128 seconds), `H=64` (32 seconds) contract.

## Train

ZN and ES configs already pin their actual tick sizes. Generic templates must
set `objective.tick_size` before use. Parameters, optimizer state, and
tick-space loss remain FP32; production configs use BF16 autocast for model
compute. Differentiable adaptation disables iterative CPM-RevIN refinement,
whose upstream `sqrt(0)` backward is undefined, and applies a few-ULP
perturbation only to exactly constant input patches while preserving the cutoff
value. Real-checkpoint shuffled-batch gates pass in FP32 and BF16. The trainer
still fails closed on any non-finite loss or gradient.

Single-input route:

```bash
timesfm-ft --config configs/single_input.json
```

Multi-input, single-output route:

```bash
timesfm-ft --config configs/multi_input.json
```

These two files are templates: their `data/{single,multi}` paths are generated
by `scripts/make_synthetic_data.py` or replaced with real audited bundles.
The ZN/ES configs are the ready-to-run production definitions. Both generic
routes use matched defaults so that only the input variates differ.
Training writes atomic `best/` and `last/` states. Set
`trainer.resume_from` to either checkpoint directory or its
`training_state.pt` to resume optimizer, scheduler, epoch, history, DataLoader,
and RNG state exactly.

## Inference and evaluation

Fine-tuned evaluation reconstructs the official backbone and matching adapter
structure, loads `adapter.pt`, and then calls the official
`TimesFM3Torch.decode()` method. The differentiable decode bypass is used only
during training. Parity is with raw `TimesFM3Torch.decode`; unlike the
high-level `TimesFM3Forecaster`, quantiles are not sorted after decode so
crossing remains observable and measurable.

Evaluate a fine-tuned adapter:

```bash
timesfm-eval \
  --config outputs/zn-single-input-c256-h64/experiment_config.json \
  --adapter outputs/zn-single-input-c256-h64/best/adapter.pt \
  --split test
```

When `test_path` exists, `test` is the default. Use `--split val` only for
model development. Explicit `--data` is mutually exclusive with `--split`,
still requires a manifest declaring `val` or `test`, and rejects both the
configured train path and copied bundles declaring `split=train`.

Run the holdout with the untouched official checkpoint:

```bash
timesfm-eval --config configs/zn_single_input.json --split test
```

Optional arguments:

```text
--split {val,test} choose a configured chronological split
--data PATH        explicit non-training dataset (mutually exclusive with --split)
--unsafe-data      bypass explicit-data provenance checks (requires --data)
--output-dir PATH  override the metric directory
--batch-size N     override trainer.batch_size
--device DEVICE    override trainer.device
```

Evaluation writes:

```text
evaluation/
├── summary.json       # overall point, baseline, and probabilistic metrics
└── per_horizon.csv    # metrics from 0.5s through 32s
```

Reported metrics include P50 MAE/RMSE in ticks, persistence RMSE, out-of-sample
R² versus persistence, mean pinball loss, quantile coverage and calibration
error, quantile crossing rate, and per-horizon breakdowns.

## Fine-tuning modes

Set `adapter.type` to:

- `head`: trains only the 1280-to-576 quantile output projection;
- `lora`: trains the output head and low-rank updates in sequence attention,
  variate attention, and FFN modules of the last N layers;
- `partial`: trains the input residual block, last N transformer layers, and
  output head;
- `full`: trains all parameters.

Start with `head` as an integration check, then use `lora` for the first real
experiment. Full fine-tuning should not be the default.

## Optimizer and scheduler

Optimizer and runtime settings are intentionally separate:

```text
optimizer.adapter_learning_rate    # LoRA matrices, default 1e-4
optimizer.head_learning_rate       # quantile output head, default 3e-4
optimizer.pretrained_learning_rate # partially/full-unfrozen weights, default 1e-5
scheduler.warmup_ratio              # linear warmup fraction
scheduler.min_lr_ratio              # final LR / initial group LR
trainer.log_every_steps             # console logging interval
```

All active groups use AdamW. LoRA matrices have zero weight decay; the output
head and unfrozen pretrained weights use `optimizer.weight_decay`. After
warmup, every group follows the same cosine multiplier while preserving its
own base LR.

## Objective

The backbone produces absolute-price quantiles. Before loss computation,
predictions and labels are converted to displacement from the cutoff price:

```text
ticks[h] = (weighted_mid[t+h] - weighted_mid[t]) / tick_size
```

The default objective is:

```text
pinball(P10, P20, P30, P40, P60, P70, P80, P90)
+ 0.5 * Huber(P50)
+ 0.05 * quantile-crossing penalty
```

P50 is deliberately excluded from Pinball so it is not supervised twice.
Best-checkpoint selection uses mask-aware validation P50 RMSE in ticks, while
composite loss, persistence RMSE, OOS R², direction, coverage, and crossing are
all logged. TimesFM emits and returns the requested 64 points (32 seconds).

## Outputs

Each experiment directory contains:

```text
experiment_config.json
history.jsonl
best/
├── adapter.pt
├── adapter_config.json
└── training_state.pt
last/
├── adapter.pt
├── adapter_config.json
└── training_state.pt
```

`adapter.pt` contains only parameters marked trainable by the selected tuning
mode. `training_state.pt` additionally contains optimizer, scheduler, epoch,
history, RNG, config, and data provenance required for exact resume. Adapter
metadata is validated against the current base checkpoint and tuning structure
before loading.

Console logs include:

- run configuration, dataset sizes, device, dtype, and trainable parameter count;
- parameter count, LR, and weight decay for every active optimizer group;
- step-level total/pinball/Huber/crossing loss and current group LRs;
- epoch-level train/validation components, gradient norm, elapsed time, and
  samples per second;
- persistence-relative validation metrics, early stopping, checkpoint, resume,
  and run-completion events.

`history.jsonl` stores the epoch metrics and current LR for machine-readable
analysis.

## Verification

```bash
conda activate timesfm-ft
pytest
ruff check src tests scripts
```

The critical tests verify:

1. differentiable decode equals official inference decode before training;
2. gradients reach the output head;
3. mixed-precision plumbing preserves FP32 master weights and dtype-safe inference;
4. balanced loss, masks, crossings, and all-masked rejection;
5. NPZ and memory-mapped bundle contracts plus split/session provenance;
6. partial accumulation, early stopping, atomic checkpoints, and exact resume;
7. explicit holdout selection and adapter metadata compatibility.

Before a long run, execute:

```bash
python scripts/smoke_real_checkpoint.py \
  --dtype bfloat16 --batch-size 64 --shuffle
```

It performs a real-checkpoint train step over randomized production windows
plus save/load/resume parity.
Then compare untouched zero-shot, head-only, and LoRA against persistence on
the same test bundle.