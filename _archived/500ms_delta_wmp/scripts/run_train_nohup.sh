#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") CONFIG [RESUME_CHECKPOINT]

Start TimesFM-FT training under nohup. Logs append to:
  <trainer.output_dir>/train.log

The launcher writes the background PID to:
  <trainer.output_dir>/train.pid
EOF
}

if [[ $# -lt 1 || $# -gt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]] && exit 0
  exit 2
fi

CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO_ROOT/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found: $CONFIG" >&2
  exit 1
fi

OUTPUT_DIR="$(
  conda run --no-capture-output -n timesfm-ft python - "$CONFIG" <<'PY'
import sys
from timesfm_ft.config import ExperimentConfig

print(ExperimentConfig.from_json(sys.argv[1]).trainer.output_dir)
PY
)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"
PID_FILE="$OUTPUT_DIR/train.pid"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE")"
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Error: training is already running with PID $EXISTING_PID" >&2
    exit 1
  fi
fi

COMMAND=(
  conda run --no-capture-output -n timesfm-ft
  timesfm-ft --config "$CONFIG"
)
if [[ $# -eq 2 ]]; then
  RESUME="$2"
  if [[ "$RESUME" != /* ]]; then
    RESUME="$REPO_ROOT/$RESUME"
  fi
  COMMAND+=(--resume "$RESUME")
fi

{
  printf '\n[%s] starting:' "$(date --iso-8601=seconds)"
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} >>"$LOG_FILE"

nohup env PYTHONUNBUFFERED=1 "${COMMAND[@]}" \
  >>"$LOG_FILE" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$PID_FILE"

echo "Started PID $PID"
echo "Log: $LOG_FILE"
echo "Follow: tail -f '$LOG_FILE'"
