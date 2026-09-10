#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/shuaibin.li/miniconda3/envs/timesfm-ft/bin/python"
RAW_ROOT="/home/shuaibin.li/projects/timesfm_ft/datas/zn_rank_selected100_1min_20221101_20260130"
DOWNLOAD_PID_FILE="/home/shuaibin.li/projects/timesfm_ft/datas/wenyao_features/logs/download_zn_rank_selected100_1min.pid"
SCHEMA="$ROOT/configs/datasets/zn_rank_selected100_1min.json"
SPLITS="$ROOT/configs/splits/zn-rank-selected100"
BUNDLES="$ROOT/data/zn-rank-selected100-1min"
SELECTION="$ROOT/outputs/feature-selection/zn-rank-selected100.json"
EXPERIMENT="$ROOT/configs/experiments/zn_rank_selected_e2.json"

if [[ -f "$DOWNLOAD_PID_FILE" ]]; then
  download_pid="$(cat "$DOWNLOAD_PID_FILE")"
  while kill -0 "$download_pid" 2>/dev/null; do
    echo "[$(date --iso-8601=seconds)] waiting for download PID $download_pid"
    sleep 60
  done
fi

partition_count="$(
  "$PYTHON" - "$RAW_ROOT" <<'PY'
from pathlib import Path
import sys

print(sum(1 for _ in Path(sys.argv[1]).glob("date=*/part*.parquet")))
PY
)"
if [[ "$partition_count" -ne 813 ]]; then
  echo "expected 813 raw partitions, found $partition_count" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" scripts/build_rank_selected100_schema.py
"$PYTHON" scripts/build_rank_selected100_splits.py
"$PYTHON" scripts/prepare_intraday_splits.py \
  --schema "$SCHEMA" \
  --source-root "$RAW_ROOT" \
  --dates-dir "$SPLITS" \
  --output-root "$BUNDLES" \
  --overwrite
"$PYTHON" scripts/select_past_only_features.py \
  --bundle "$BUNDLES/train" \
  --output "$SELECTION" \
  --limit 20 \
  --family-limit 4 \
  --correlation-limit 0.9 \
  --max-missing-rate 0.05
"$PYTHON" scripts/build_rank_selected_experiment.py \
  --selection "$SELECTION" \
  --output "$EXPERIMENT"

"$PYTHON" - "$BUNDLES" "$SELECTION" "$SPLITS" <<'PY'
import json
from pathlib import Path
import sys

from timesfm_ft.data import IntradayWindowDataset

bundles = Path(sys.argv[1])
selection = json.loads(Path(sys.argv[2]).read_text())
splits = Path(sys.argv[3])
dataset = IntradayWindowDataset(
    bundles / "train",
    context_min=64,
    context_max=192,
    horizon_length=64,
    stride=1,
    past_only_features=selection["selected_features"],
    past_future_features=("sin_time_of_day", "cos_time_of_day", "time_to_close"),
    max_variates=32,
    expected_split="train",
    expected_dataset_id="zn_rank_selected100_1min_wmid_ticks_v1",
    expected_product="ZN",
    expected_target_name="return_1m",
    expected_target_unit="ZN_ticks",
    expected_target_price_source="WMid",
    expected_target_return_type="tick_displacement",
    expected_target_timestamp_semantics="bar_end",
    expected_target_availability_lag_minutes=0,
    expected_target_missing_policy="mask",
    expected_frequency_minutes=1,
    expected_session_minutes=390,
    expected_dates_path=splits / "dates-train.txt",
)
if dataset.num_variates != 24:
    raise SystemExit(f"expected 24 final variates, found {dataset.num_variates}")
print(
    f"training adaptation complete: windows={len(dataset)}, "
    f"variates={dataset.num_variates}"
)
PY
