#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/LLaVA-NeXT-Video-7B-hf}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/vetbench/llava_next_video_7b_replication_v1}"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
[[ "$NUM_SHARDS" -gt 0 ]] || NUM_SHARDS="${#GPUS[@]}"

cd "$ROOT"
"$PYTHON_BIN" scripts/llava_next_video_replication.py smoke
mkdir -p "$OUT_DIR/shards/discovery" "$OUT_DIR/shards/validation"

run_side() {
  local side="$1"
  for ((i=0; i<NUM_SHARDS; i++)); do
    local csv="$OUT_DIR/shards/$side/behavior_shard_${i}.csv"
    local marker="$OUT_DIR/shards/$side/behavior_shard_${i}.complete.json"
    if [[ -e "$csv" && -e "$marker" ]]; then
      if "$PYTHON_BIN" - "$marker" "$i" "$NUM_SHARDS" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if int(m.get("shard_index", -1)) != int(sys.argv[2]) or int(m.get("num_shards", -1)) != int(sys.argv[3]):
    raise SystemExit(1)
PY
      then
        continue
      fi
    fi
    CUDA_VISIBLE_DEVICES="${GPUS[$((i % ${#GPUS[@]}))]}" \
      "$PYTHON_BIN" scripts/llava_next_video_replication.py run \
      --model-dir "$MODEL_DIR" --side "$side" --shard-index "$i" --num-shards "$NUM_SHARDS" \
      --device cuda:0 &
  done
  wait
}

run_side discovery
run_side validation
"$PYTHON_BIN" scripts/llava_next_video_replication.py merge --num-shards "$NUM_SHARDS"
"$PYTHON_BIN" scripts/llava_next_video_probe.py
"$PYTHON_BIN" scripts/llava_next_video_recursive.py
