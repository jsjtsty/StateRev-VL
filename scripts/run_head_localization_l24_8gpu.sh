#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
STAGE="${STAGE:-discovery}"
CONDITION="${CONDITION:-source_current_transplant}"
HEADS="${HEADS:-all}"
JOINT_LABEL="${JOINT_LABEL:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/head_localization_l24_v1}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_SHARDS="${#GPUS[@]}"
[[ "$NUM_SHARDS" -gt 0 ]] || { echo "GPU_IDS must contain at least one GPU" >&2; exit 2; }
[[ "$STAGE" == discovery || "$STAGE" == validation ]] || { echo "bad STAGE" >&2; exit 2; }
[[ -f "$OUT_DIR/discovery_pair_manifest.json" ]] || { echo "missing manifest; copy from circuit_localization_v2 or prepare it" >&2; exit 1; }
mkdir -p "$OUT_DIR/shards/$STAGE"
pids=()
for shard in "${!GPUS[@]}"; do
  d="$OUT_DIR/shards/$STAGE/shard_$shard"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" python3 scripts/head_localization_l24.py run \
    --stage "$STAGE" --condition "$CONDITION" --model-dir "$MODEL_DIR" \
    --dataset "$DATASET_DIR" --out "$OUT_DIR" --shard-index "$shard" --num-shards "$NUM_SHARDS" \
    --heads "$HEADS" \
    --joint-label "$JOINT_LABEL" \
    > "$d/head_${CONDITION}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "head shard failure; inspect logs" >&2; exit $status; }
echo "completed $STAGE $CONDITION"
