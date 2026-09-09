# Archived 500 ms Delta-WMP Experiments

This directory preserves the superseded single-variable research task.

## Frozen task

- frequency: 500 ms
- target: one-step `delta_wmp / tick_size`
- context: 256 points (128 seconds)
- horizon: 64 points (32 seconds)
- products: ZN, with ES as a separate control
- experiments: zero-shot, head-only, LoRA, and full fine-tuning

The suite was stopped on 2026-09-09 before the full fine-tuning run completed.
Head-only and LoRA runs completed; the full run stopped during epoch 3.

## Contents

- `configs/`: all legacy experiment and smoke-test configurations
- `scripts/`: all legacy data preparation, launch, and visualization scripts
- `legacy_README.md`: repository documentation for the frozen task
- `data/`: ignored local mmap/NPZ artifacts, moved without conversion
- `outputs/`: ignored checkpoints, logs, metrics, and plots, moved without conversion

The active repository intentionally does not maintain compatibility with this
contract. Git history plus this archive are the only supported references.
