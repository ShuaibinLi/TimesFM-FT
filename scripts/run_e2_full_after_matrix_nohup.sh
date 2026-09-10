#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_DIR="$ROOT/outputs/zn-rank-e2-pilot-matrix"
MATRIX_PID_FILE="$MATRIX_DIR/matrix.pid"
MATRIX_LOG="$MATRIX_DIR/matrix.log"
RUN_DIR="$MATRIX_DIR/full-scheduler"
LOG_FILE="$RUN_DIR/run.log"
PID_FILE="$RUN_DIR/run.pid"
CONFIG="$ROOT/configs/experiments/zn_rank_e2_full_pilot.json"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--worker" ]]; then
  if [[ ! -f "$MATRIX_PID_FILE" ]]; then
    echo "pilot matrix PID file not found: $MATRIX_PID_FILE" >&2
    exit 1
  fi
  matrix_pid="$(cat "$MATRIX_PID_FILE")"
  while kill -0 "$matrix_pid" 2>/dev/null; do
    printf '[%s] waiting for head/LoRA matrix PID %s\n' \
      "$(date --iso-8601=seconds)" "$matrix_pid"
    sleep 60
  done
  if ! rg -q "train complete zn_rank_e2_lora_pilot" "$MATRIX_LOG"; then
    echo "head/LoRA matrix did not complete cleanly; full pilot blocked" >&2
    exit 1
  fi
  cd "$ROOT"
  printf '[%s] full smoke start\n' "$(date --iso-8601=seconds)"
  conda run --no-capture-output -n timesfm-ft \
    python "$ROOT/scripts/smoke_real_checkpoint.py" \
    --config "$CONFIG" \
    --batch-size 1 \
    --dtype bfloat16 \
    --minimum-context 96
  printf '[%s] full train start\n' "$(date --iso-8601=seconds)"
  conda run --no-capture-output -n timesfm-ft \
    timesfm-ft --config "$CONFIG"
  printf '[%s] full train complete\n' "$(date --iso-8601=seconds)"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Full pilot waiter already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup "$0" --worker >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
echo "Started PID $!"
echo "Log: $LOG_FILE"
