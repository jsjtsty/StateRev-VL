#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/metbench_chess/llava_compact_event_replication_v1}"
MANIFEST="${MANIFEST:-$ROOT/outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1/pilot_manifest.csv}"
MODEL_DIR="${MODEL_DIR_LLAVA:-$ROOT/models/LLaVA-NeXT-Video-7B-hf}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"; NUM_SHARDS="${NUM_SHARDS:-4}"; CANDIDATE_LAYERS="${CANDIDATE_LAYERS:-0,8,16,24,32}"
PY="${PYTHON:-python}"; IFS=',' read -r -a GPUS <<< "$GPU_IDS"
(( ${#GPUS[@]} <= 4 )) || { echo '最多支持4卡' >&2; exit 2; }
mkdir -p "$OUT_DIR"
run_phase() {
 local phase="$1"; local pids=() i gpu
 for ((i=0;i<NUM_SHARDS;i++)); do
  gpu="${GPUS[$((i%${#GPUS[@]}))]}"
  if [[ -f "$OUT_DIR/extract_llava_${phase}_${i}.complete.json" && "${OVERWRITE:-0}" != 1 ]]; then continue; fi
  local native=(); [[ "$phase" == validation ]] && native=(--run-native-state)
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/metbench_chess_llava_replication.py" --out "$OUT_DIR" --manifest "$MANIFEST" extract \
   --model-dir "$MODEL_DIR" --phase "$phase" --shard-index "$i" --num-shards "$NUM_SHARDS" --candidate-layers "$CANDIDATE_LAYERS" "${native[@]}" \
   $(if [[ "${OVERWRITE:-0}" == 1 ]]; then echo --overwrite; fi) >"$OUT_DIR/extract_llava_${phase}_${i}.log" 2>&1 & pids+=("$!")
 done
 for p in "${pids[@]}"; do wait "$p"; done
 "$PY" "$ROOT/scripts/metbench_chess_llava_replication.py" --out "$OUT_DIR" --manifest "$MANIFEST" merge --phase "$phase" --num-shards "$NUM_SHARDS"
}
case "${1:-all}" in
 extract-discovery) run_phase discovery ;;
 extract-validation) run_phase validation ;;
 evaluate) "$PY" "$ROOT/scripts/metbench_chess_llava_replication.py" --out "$OUT_DIR" --manifest "$MANIFEST" evaluate ;;
 all) run_phase discovery; run_phase validation; "$PY" "$ROOT/scripts/metbench_chess_llava_replication.py" --out "$OUT_DIR" --manifest "$MANIFEST" evaluate ;;
 *) echo "usage: $0 {extract-discovery|extract-validation|evaluate|all}"; exit 2 ;;
esac
