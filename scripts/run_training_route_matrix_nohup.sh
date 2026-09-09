#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_ROOT/outputs/training-route-matrix"
LOG_FILE="$RUN_DIR/matrix.log"
PID_FILE="$RUN_DIR/matrix.pid"
CONFIGS=(
  t0_f0_final
  t1_f0_all
  t2_f1
  t3_f1_mv
)
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--worker" ]]; then
  for name in "${CONFIGS[@]}"; do
    config="$REPO_ROOT/configs/experiments/$name.json"
    printf '[%s] start %s\n' "$(date --iso-8601=seconds)" "$name"
    conda run --no-capture-output -n timesfm-ft \
      python "$REPO_ROOT/scripts/smoke_real_checkpoint.py" \
      --config "$config" --dtype bfloat16 --batch-size 2
    conda run --no-capture-output -n timesfm-ft \
      timesfm-ft --config "$config"
    printf '[%s] complete %s\n' "$(date --iso-8601=seconds)" "$name"
  done
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Training route matrix already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup "$0" --worker >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
echo "Started PID $!"
echo "Log: $LOG_FILE"
