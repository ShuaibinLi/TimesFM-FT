#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE_DIR="$REPO_ROOT/outputs/zn-pinball-suite"
SUITE_LOG="$SUITE_DIR/suite.log"
SUITE_PID_FILE="$SUITE_DIR/suite.pid"
WAIT_PID_FILE="$REPO_ROOT/outputs/zn-small-lora-c256-h64/train.pid"
CONFIGS=(
  "$REPO_ROOT/configs/zn_pinball_head.json"
  "$REPO_ROOT/configs/zn_pinball_lora.json"
  "$REPO_ROOT/configs/zn_pinball_full.json"
)

output_dir_for_config() {
  conda run --no-capture-output -n timesfm-ft python - "$1" <<'PY'
import sys
from timesfm_ft.config import ExperimentConfig

print(ExperimentConfig.from_json(sys.argv[1]).trainer.output_dir)
PY
}

run_worker() {
  if [[ -f "$WAIT_PID_FILE" ]]; then
    WAIT_PID="$(cat "$WAIT_PID_FILE")"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      echo "[$(date --iso-8601=seconds)] waiting for PID $WAIT_PID"
      sleep 30
    done
  fi

  for config in "${CONFIGS[@]}"; do
    output_dir="$(output_dir_for_config "$config")"
    mkdir -p "$output_dir"
    model_log="$output_dir/train.log"
    model_pid_file="$output_dir/train.pid"
    echo "[$(date --iso-8601=seconds)] gating $config"
    env PYTHONUNBUFFERED=1 \
      conda run --no-capture-output -n timesfm-ft \
      python "$REPO_ROOT/scripts/smoke_real_checkpoint.py" \
      --config "$config" --dtype bfloat16 --batch-size 32 --shuffle \
      >>"$model_log" 2>&1
    echo "[$(date --iso-8601=seconds)] starting $config"
    env PYTHONUNBUFFERED=1 \
      conda run --no-capture-output -n timesfm-ft \
      timesfm-ft --config "$config" >>"$model_log" 2>&1 &
    model_pid=$!
    echo "$model_pid" >"$model_pid_file"
    set +e
    wait "$model_pid"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      echo "[$(date --iso-8601=seconds)] failed $config status=$status"
      exit "$status"
    fi
    echo "[$(date --iso-8601=seconds)] completed $config"
  done
  echo "[$(date --iso-8601=seconds)] suite completed"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

mkdir -p "$SUITE_DIR"
if [[ -f "$SUITE_PID_FILE" ]]; then
  existing_pid="$(cat "$SUITE_PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Error: suite already running with PID $existing_pid" >&2
    exit 1
  fi
fi

nohup "$0" --worker >>"$SUITE_LOG" 2>&1 </dev/null &
suite_pid=$!
echo "$suite_pid" >"$SUITE_PID_FILE"
echo "Queued suite PID $suite_pid"
echo "Log: $SUITE_LOG"
