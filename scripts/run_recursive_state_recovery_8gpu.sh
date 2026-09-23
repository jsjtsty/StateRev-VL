#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
PY="${PY:-python3}"
OUT="${OUT_DIR:-outputs/vetbench/recursive_state_recovery_v1}"
BEH="${BEHAVIOR:-outputs/vetbench/composition_analysis_v1/transformers_behavior.csv}"
NATIVE="${NATIVE_BEHAVIOR:-outputs/vetbench/mechanism_gate_final/behavior.csv}"
SPLIT="${SPLIT:-outputs/vetbench/circuit_localization_v1/discovery_validation_split.json}"
MODEL="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATA="${DATASET_DIR:-dataset/vetbench/cup}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
mkdir -p "$OUT/shards"
echo "[1/6] prepare"
$PY scripts/recursive_state_recovery.py prepare --out "$OUT" --behavior "$BEH" --native-behavior "$NATIVE" --split "$SPLIT"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; N=${#GPUS[@]}
echo "[2/6] launch $N event-probability shards on $GPU_IDS"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" $PY scripts/recursive_state_recovery.py event-probs \
    --out "$OUT" --model-dir "$MODEL" --dataset "$DATA" --shard-index "$i" --num-shards "$N" \
    > "$OUT/shards/event_probs_shard_${i}.log" 2>&1 &
  pids+=("$!")
done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
[[ "$failed" == 0 ]] || { echo "event-probability shard failed" >&2; exit 1; }
echo "[3/6] verify completion markers"
for i in "${!GPUS[@]}"; do test -f "$OUT/shards/event_probs_shard_${i}.complete.json"; done
echo "[4/6] recursive hard/probabilistic analysis"
$PY scripts/recursive_state_recovery.py analyze --out "$OUT" --event-probs --num-shards "$N"
echo "[5/6] discovery-only gate fitting"
$PY scripts/recursive_state_recovery.py fit-gate --out "$OUT" --split "$SPLIT"
echo "[6/6] frozen validation gate evaluation"
$PY scripts/recursive_state_recovery.py evaluate-gate --out "$OUT" --split "$SPLIT"
echo "recursive recovery complete: $OUT"
