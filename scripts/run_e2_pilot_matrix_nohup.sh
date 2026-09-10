#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/outputs/zn-rank-e2-pilot-matrix"
LOG_FILE="$RUN_DIR/matrix.log"
PID_FILE="$RUN_DIR/matrix.pid"
CONFIGS=(
  zn_rank_e2_pilot
  zn_rank_e2_lora_pilot
  zn_rank_e2_full_pilot
)
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--worker" ]]; then
  cd "$ROOT"
  for name in "${CONFIGS[@]}"; do
    config="$ROOT/configs/experiments/$name.json"
    printf '[%s] smoke start %s\n' "$(date --iso-8601=seconds)" "$name"
    conda run --no-capture-output -n timesfm-ft \
      python "$ROOT/scripts/smoke_real_checkpoint.py" \
      --config "$config" \
      --batch-size 1 \
      --dtype bfloat16 \
      --minimum-context 96
    printf '[%s] train start %s\n' "$(date --iso-8601=seconds)" "$name"
    conda run --no-capture-output -n timesfm-ft \
      timesfm-ft --config "$config"
    printf '[%s] train complete %s\n' "$(date --iso-8601=seconds)" "$name"
  done
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "E2 pilot matrix already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup "$0" --worker >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
echo "Started PID $!"
echo "Log: $LOG_FILE"
