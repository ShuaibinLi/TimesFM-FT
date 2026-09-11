#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/shuaibin.li/miniconda3/envs/timesfm-ft/bin/python"
TRAIN="/home/shuaibin.li/miniconda3/envs/timesfm-ft/bin/timesfm-ft"
GRID="$ROOT/outputs/zn-rank-3x3"
CURRENT="$ROOT/outputs/zn-rank-e2-pilot-matrix"
RUNTIME_CONFIGS="$GRID/runtime-configs"
LOG_FILE="$GRID/orchestrator.log"
PID_FILE="$GRID/orchestrator.pid"
SPARK4="spark4.nyc01.vaticlabs.net"
SPARK5="spark5.nyc01.vaticlabs.net"

remote_active() {
  local host="$1"
  local pid_file="$2"
  ssh "$host" "pid=\$(cat '$pid_file' 2>/dev/null || true); [[ -n \"\$pid\" ]] && kill -0 \"\$pid\" 2>/dev/null"
}

wait_remote() {
  local stage="$1"
  local host="$2"
  local pid_file="$3"
  while remote_active "$host" "$pid_file"; do
    printf '[%s] %s waiting for pid on %s\n' \
      "$(date --iso-8601=seconds)" "$stage" "$host"
    sleep 60
  done
}

run_remote() {
  local host="$1"
  local config="$2"
  local log="$3"
  ssh "$host" "mkdir -p '$(dirname "$log")' && cd '$ROOT' && '$TRAIN' --config '$config' >'$log' 2>&1"
}

run_local() {
  local config="$1"
  local log="$2"
  mkdir -p "$(dirname "$log")"
  "$TRAIN" --config "$config" >"$log" 2>&1
}

sync_runtime_configs() {
  local host="$1"
  ssh "$host" "mkdir -p '$RUNTIME_CONFIGS'"
  rsync -a "$RUNTIME_CONFIGS/" "$host:$RUNTIME_CONFIGS/"
}

winner_config() {
  local stage="$1"
  "$PYTHON" -c \
    "import json; p=json.load(open('$GRID/winner-selection/$stage.json')); print(p.get('stages', {}).get('$stage', p)['three_epoch_config'])"
}

if [[ "${1:-}" == "--worker" ]]; then
  cd "$ROOT"
  "$PYTHON" scripts/build_3x3_experiment_configs.py pilots
  sync_runtime_configs "$SPARK4"
  sync_runtime_configs "$SPARK5"

  # spark6 owns the complete E1 series.
  (
    for mode in head lora full; do
      run_local \
        "$RUNTIME_CONFIGS/e1-$mode-pilot.json" \
        "$GRID/pilots/e1/$mode/train.log"
    done
    "$PYTHON" scripts/build_3x3_experiment_configs.py losses --stage e1
    for variant in p1 p2 p3; do
      config="$(
        "$PYTHON" -c "import json; print(json.load(open('$GRID/loss-ablation/e1/tuning-winner.json'))['variants']['$variant']['config'])"
      )"
      run_local "$config" "$GRID/loss-ablation/e1/$variant/train.log"
    done
    "$PYTHON" scripts/build_3x3_experiment_configs.py loss-winner --stage e1
    run_local "$(winner_config e1)" "$GRID/winners-3epoch/e1/train.log"
  ) &
  e1_lane=$!

  # spark4 finishes the existing E2 full pilot, then owns the complete E0 series.
  (
    wait_remote full "$SPARK4" "$CURRENT/full/remote.pid"
    for mode in head lora full; do
      run_remote \
        "$SPARK4" \
        "$RUNTIME_CONFIGS/e0-$mode-pilot.json" \
        "$GRID/pilots/e0/$mode/train.log"
    done
    for mode in head lora full; do
      mkdir -p "$GRID/pilots/e0/$mode"
      rsync -a --delete \
        "$SPARK4:$GRID/pilots/e0/$mode/" \
        "$GRID/pilots/e0/$mode/"
    done
    "$PYTHON" scripts/build_3x3_experiment_configs.py winners --stage e0
    sync_runtime_configs "$SPARK4"
    run_remote "$SPARK4" "$(winner_config e0)" "$GRID/winners-3epoch/e0/train.log"
    mkdir -p "$GRID/winners-3epoch/e0"
    rsync -a --delete \
      "$SPARK4:$GRID/winners-3epoch/e0/" \
      "$GRID/winners-3epoch/e0/"
  ) &
  e0_lane=$!

  # spark5 finishes E2 head, waits for E2 full, then resumes the best E2 mode.
  (
    wait_remote head "$SPARK5" "$CURRENT/head/remote.pid"
    wait_remote full "$SPARK4" "$CURRENT/full/remote.pid"
    collector_pid="$(cat "$CURRENT/remote-collector.pid" 2>/dev/null || true)"
    [[ -n "$collector_pid" ]] && kill "$collector_pid" 2>/dev/null || true
    mkdir -p "$GRID/pilots/e2/head" "$GRID/pilots/e2/lora" "$GRID/pilots/e2/full"
    rsync -a --delete "$SPARK5:$CURRENT/head/" "$GRID/pilots/e2/head/"
    rsync -a --delete "$CURRENT/lora/" "$GRID/pilots/e2/lora/"
    rsync -a --delete "$SPARK4:$CURRENT/full/" "$GRID/pilots/e2/full/"
    "$PYTHON" scripts/build_3x3_experiment_configs.py winners --stage e2
    rsync -a --delete "$GRID/pilots/e2/" "$SPARK5:$GRID/pilots/e2/"
    sync_runtime_configs "$SPARK5"
    run_remote "$SPARK5" "$(winner_config e2)" "$GRID/winners-3epoch/e2/train.log"
    mkdir -p "$GRID/winners-3epoch/e2"
    rsync -a --delete \
      "$SPARK5:$GRID/winners-3epoch/e2/" \
      "$GRID/winners-3epoch/e2/"
  ) &
  e2_lane=$!

  wait "$e0_lane" "$e1_lane" "$e2_lane"
  rm -rf "$CURRENT"
  printf '[%s] E0/E1/E2 series complete\n' "$(date --iso-8601=seconds)"
  exit 0
fi

mkdir -p "$GRID"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "E-series orchestrator already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup "$0" --worker >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
echo "Started PID $!"
echo "Log: $LOG_FILE"
