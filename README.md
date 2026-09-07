# TimesFM-FT

Research fine-tuning toolkit for TimesFM 3, initially focused on 500 ms
weighted-mid forecasting.

The first supported task uses a historical context window to predict the next
60 points (30 seconds) of one weighted-mid target. Two aligned input contracts
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

- NPZ data validation for both input routes;
- single-target extraction from multivariate TimesFM output;
- pinball, median Huber, and quantile-crossing losses in tick space;
- head-only, native LoRA, partial-unfreeze, and full fine-tuning modes;
- train/validation loops, gradient accumulation, warmup/cosine scheduling, and
  adapter-only checkpoints;
- tiny-model tests that verify decode parity and gradient flow.

## Repository layout

```text
TimesFM-FT/
├── 3rdparty/timesfm/        # pinned official upstream submodule
├── configs/                 # matched single/multi experiment configs
├── scripts/                 # synthetic contract data generator
├── src/timesfm_ft/
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── adapter.py            # all TimesFM-specific loading and adaptation
│   ├── trainer.py
│   └── cli.py
└── tests/
```

The upstream submodule is intentionally not modified.

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

Each train or validation file is an NPZ archive containing pre-windowed,
chronologically split samples.

Required arrays:

```text
context_values: float32[S, V, C] or float32[S, C]
future_values:  float32[S, H]
```

Optional arrays:

```text
context_mask: bool[S, V, C] or bool[S, C]  # True means unavailable
future_mask:  bool[S, H]                   # True means exclude from loss
timestamps:   int64[S]                     # retained for auditing, not training
```

Conventions:

- `C=512` and `H=60` in the initial configs;
- variate zero must always be weighted-mid;
- `V=1` for the single-input route and `1<V<=32` for the multi-input route;
- variates `1..V-1` are past-only covariates;
- the weighted-mid value at the forecast cutoff must be present;
- `future_values` always contains weighted-mid only;
- train and validation files must already come from purged chronological
  splits, never random row splits.

Both supplied datasets must share exactly the same timestamps and
`future_values` before A/B model comparisons are accepted.

## Contract smoke test

Generate small aligned single/multi datasets:

```bash
python scripts/make_synthetic_data.py
```

This writes ignored files under:

```text
data/single/{train,val}.npz
data/multi/{train,val}.npz
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

## Train

First update `loss.tick_size` in both configs to the instrument's actual tick
size.

Single-input route:

```bash
timesfm-ft --config configs/single_input.json
```

Multi-input, single-output route:

```bash
timesfm-ft --config configs/multi_input.json
```

Both routes use matched defaults so that only the input variates differ.

## Fine-tuning modes

Set `model.tuning_mode` to:

- `head`: trains only the 1280-to-576 quantile output projection;
- `lora`: trains the output head and low-rank updates in sequence attention,
  variate attention, and FFN modules of the last N layers;
- `partial`: trains the input residual block, last N transformer layers, and
  output head;
- `full`: trains all parameters.

Start with `head` as an integration check, then use `lora` for the first real
experiment. Full fine-tuning should not be the default.

## Objective

The backbone produces absolute-price quantiles. Before loss computation,
predictions and labels are converted to displacement from the cutoff price:

```text
ticks[h] = (weighted_mid[t+h] - weighted_mid[t]) / tick_size
```

The default objective is:

```text
pinball(all quantiles)
+ 0.5 * Huber(P50)
+ 0.05 * quantile-crossing penalty
```

TimesFM 3 emits 64 output points per anchor. The requested 60-point horizon is
returned directly by official decode logic; the internal padding points are
not exposed to the loss.

## Outputs

Each experiment directory contains:

```text
experiment_config.json
history.jsonl
best/
├── adapter.pt
└── adapter_config.json
```

`adapter.pt` contains only parameters marked trainable by the selected tuning
mode. For `full`, this is necessarily the complete model state.

## Verification

```bash
pytest
ruff check src tests scripts
```

The critical tests verify:

1. differentiable decode equals official inference decode before training;
2. gradients reach the output head;
3. injected LoRA is initially output-preserving and receives gradients;
4. single and multi NPZ contracts are validated consistently.

## Next integration step

When the real datasets arrive:

1. map their storage format into the documented window contract;
2. verify single/multi timestamps and labels are exactly aligned;
3. add day/session metadata validation and a split audit;
4. run head-only overfit tests on a tiny subset;
5. run matched zero-shot, head, and LoRA experiments.