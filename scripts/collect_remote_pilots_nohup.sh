#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="$ROOT/outputs/zn-rank-e2-pilot-matrix"
LOG_FILE="$GROUP/remote-collector.log"
PID_FILE="$GROUP/remote-collector.pid"
ENTRIES=(
  head:spark5.nyc01.vaticlabs.net
  lora:spark3.nyc01.vaticlabs.net
  full:spark4.nyc01.vaticlabs.net
)

if [[ "${1:-}" == "--worker" ]]; then
  while true; do
    active=0
    for entry in "${ENTRIES[@]}"; do
      stage="${entry%%:*}"
      host="${entry#*:}"
      remote="$ROOT/outputs/zn-rank-e2-pilot-matrix/$stage"
      mkdir -p "$GROUP/$stage"
      rsync -a "$host:$remote/" "$GROUP/$stage/" 2>/dev/null || true
      remote_pid="$(ssh "$host" "cat '$remote/remote.pid' 2>/dev/null || true")"
      if [[ -n "$remote_pid" ]] && ssh "$host" "kill -0 '$remote_pid' 2>/dev/null"; then
        active=$((active + 1))
        printf '[%s] %s active pid=%s host=%s\n' \
          "$(date --iso-8601=seconds)" "$stage" "$remote_pid" "$host"
      else
        printf '[%s] %s finished host=%s\n' \
          "$(date --iso-8601=seconds)" "$stage" "$host"
      fi
    done
    [[ "$active" -eq 0 ]] && break
    sleep 60
  done
  for entry in "${ENTRIES[@]}"; do
    stage="${entry%%:*}"
    host="${entry#*:}"
    remote="$ROOT/outputs/zn-rank-e2-pilot-matrix/$stage"
    rsync -a --delete "$host:$remote/" "$GROUP/$stage/"
  done
  printf '[%s] all remote pilot outputs collected\n' "$(date --iso-8601=seconds)"
  exit 0
fi

mkdir -p "$GROUP"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Remote collector already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup "$0" --worker >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
echo "Started PID $!"
echo "Log: $LOG_FILE"
