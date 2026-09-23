#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-outputs/vetbench/multi_position_delta_rescue_v1}"
# Four GPUs are the portable default here.  NUM_SHARDS may be larger: those
# logical shards are executed in waves, so a 4-GPU host can safely resume an
# 8-shard run without ever addressing CUDA devices 4..7.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; N="${#GPUS[@]}"
NUM_SHARDS="${NUM_SHARDS:-$N}"
[[ "$N" -gt 0 && "$NUM_SHARDS" -gt 0 ]] || { echo "GPU_IDS / NUM_SHARDS must be nonempty" >&2; exit 2; }
python3 scripts/multi_position_delta_rescue.py prepare --out "$OUT_DIR"
python3 scripts/multi_position_delta_rescue.py unit --out "$OUT_DIR"
mkdir -p "$OUT_DIR/shards"
for ((first=0; first<NUM_SHARDS; first+=N)); do
  pids=()
  for ((slot=0; slot<N && first+slot<NUM_SHARDS; slot++)); do
    i=$((first+slot))
    if [[ -f "$OUT_DIR/shards/multi_position_shard_${i}.csv" && -f "$OUT_DIR/shards/multi_position_shard_${i}.complete.json" ]]; then
      echo "reuse completed shard $i"
      continue
    fi
    echo "start logical shard $i on GPU ${GPUS[$slot]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" python3 scripts/multi_position_delta_rescue.py run --out "$OUT_DIR" --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" --shard-index "$i" --num-shards "$NUM_SHARDS" >"$OUT_DIR/shards/multi_position_shard_${i}.log" 2>&1 & pids+=("$!")
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done
  [[ "$bad" == 0 ]] || { echo "a shard failed; no merge attempted" >&2; exit 1; }
done
for ((i=0; i<NUM_SHARDS; i++)); do test -f "$OUT_DIR/shards/multi_position_shard_${i}.complete.json" || { echo "missing completion marker $i" >&2; exit 1; }; done
python3 scripts/multi_position_delta_rescue.py analyze --out "$OUT_DIR" --num-shards "$NUM_SHARDS"
