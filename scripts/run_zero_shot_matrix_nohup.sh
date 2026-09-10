#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS=(
  zn_rank_e0_zero_shot_202508
  zn_rank_e1_zero_shot_202508
  zn_rank_e2_zero_shot_202508
)

RUN_DIR="$REPO_ROOT/outputs/zn-rank-zero-shot-202508-matrix"
LOG_FILE="$RUN_DIR/matrix.log"
PID_FILE="$RUN_DIR/matrix.pid"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--worker" ]]; then
  for name in "${CONFIGS[@]}"; do
    config="$REPO_ROOT/configs/experiments/$name.json"
    output="$RUN_DIR/$name"
    printf '[%s] start %s\n' "$(date --iso-8601=seconds)" "$name"
    conda run --no-capture-output -n timesfm-ft \
      timesfm-eval --config "$config" --split test --output-dir "$output"
    printf '[%s] complete %s\n' "$(date --iso-8601=seconds)" "$name"
  done
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Matrix already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup "$0" --worker >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
echo "Started PID $!"
echo "Log: $LOG_FILE"
