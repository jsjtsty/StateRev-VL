#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
PY="${PY:-python3}"
OUT="${OUT_DIR:-outputs/vetbench/conditional_l32_min_intervention_v1}"
SRC="${SOURCE_MANIFEST:-outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_manifest.json}"
GATE="${GATE_CONFIG:-outputs/vetbench/recursive_state_recovery_v1/gate_config.json}"
GATE_RESULTS="${GATE_RESULTS:-outputs/vetbench/recursive_state_recovery_v1/gate_results.csv}"
MODEL="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATA="${DATASET_DIR:-dataset/vetbench/cup}"
BEH="${BEHAVIOR:-outputs/vetbench/composition_analysis_v1/transformers_behavior.csv}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
echo "[1/4] freeze manifest"
$PY scripts/conditional_l32_min_intervention.py prepare --out "$OUT" --source-manifest "$SRC" --gate-config "$GATE"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; N=${#GPUS[@]}; mkdir -p "$OUT/shards"
echo "[2/4] launch $N L32 shards on $GPU_IDS"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" $PY scripts/conditional_l32_min_intervention.py run \
    --out "$OUT" --model-dir "$MODEL" --dataset "$DATA" --behavior "$BEH" \
    --gate-results "$GATE_RESULTS" --shard-index "$i" --num-shards "$N" \
    > "$OUT/shards/conditional_l32_shard_${i}.log" 2>&1 &
  pids+=("$!")
done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
[[ "$failed" == 0 ]] || { echo "conditional L32 shard failed" >&2; exit 1; }
echo "[3/4] verify completion markers"
for i in "${!GPUS[@]}"; do test -f "$OUT/shards/conditional_l32_shard_${i}.complete.json"; done
echo "[4/4] clustered offline analysis"
$PY scripts/conditional_l32_min_intervention.py analyze --out "$OUT" --num-shards "$N"
echo "conditional L32 experiment complete: $OUT"
