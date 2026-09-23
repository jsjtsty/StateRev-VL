#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/metbench_chess/single_piece_tracking_v1}"
MANIFEST="${MANIFEST:-$OUT_DIR/manifest.csv}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_GAMES="${MAX_GAMES:-0}"
MAX_T="${MAX_T:-0}"
QWEN_DIR="${MODEL_DIR_QWEN:-$ROOT/models/Qwen3-VL-8B-Instruct}"
LLAVA_DIR="${MODEL_DIR_LLAVA:-$ROOT/models/LLaVA-NeXT-Video-7B-hf}"
PY="${PYTHON:-python}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} > 4 )); then echo "最多支持 4 张 GPU: GPU_IDS=$GPU_IDS" >&2; exit 2; fi
mkdir -p "$OUT_DIR"

prepare() {
  "$PY" "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" build-manifest
  "$PY" "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" audit
}

extract_model() {
  local model="$1" dir="$2"
  local pids=() i gpu
  for ((i=0; i<NUM_SHARDS; i++)); do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    if [[ -f "$OUT_DIR/extract_${model}_${i}.complete.json" && "${OVERWRITE:-0}" != 1 ]]; then continue; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/metbench_chess_experiment.py" extract \
      --out "$OUT_DIR" --manifest "$MANIFEST" --model "$model" --model-dir "$dir" \
      --shard-index "$i" --num-shards "$NUM_SHARDS" --max-games "$MAX_GAMES" --max-t "$MAX_T" \
      $(if [[ "${OVERWRITE:-0}" == 1 ]]; then echo --overwrite; fi) >"$OUT_DIR/extract_${model}_${i}.log" 2>&1 & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  "$PY" "$ROOT/scripts/metbench_chess_experiment.py" merge --out "$OUT_DIR" --model "$model" --num-shards "$NUM_SHARDS"
}

fit_eval() {
  local model="$1"
  "$PY" "$ROOT/scripts/metbench_chess_experiment.py" fit --out "$OUT_DIR" --manifest "$MANIFEST" --model "$model" \
    --candidate-layers "${CANDIDATE_LAYERS:-0}" --candidate-c "${CANDIDATE_C:-0.1,1,10}" --epochs "${UPDATER_EPOCHS:-3}" \
    --max-fit-games "${MAX_FIT_GAMES:-2000}"
  "$PY" "$ROOT/scripts/metbench_chess_experiment.py" evaluate --out "$OUT_DIR" --manifest "$MANIFEST" --model "$model"
}

case "${1:-help}" in
  prepare) prepare ;;
  extract-qwen) extract_model qwen "$QWEN_DIR" ;;
  extract-llava) extract_model llava "$LLAVA_DIR" ;;
  fit-qwen) fit_eval qwen ;;
  fit-llava) fit_eval llava ;;
  all)
    prepare
    extract_model qwen "$QWEN_DIR"
    extract_model llava "$LLAVA_DIR"
    fit_eval qwen
    fit_eval llava
    ;;
  help|*)
    echo "usage: $0 {prepare|extract-qwen|extract-llava|fit-qwen|fit-llava|all}"
    echo "GPU_IDS=0,1,2,3 NUM_SHARDS=4 MAX_GAMES=0 MAX_T=10 $0 all"
    ;;
esac
