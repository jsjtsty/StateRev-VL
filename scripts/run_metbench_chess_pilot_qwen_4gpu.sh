#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1}"
BASE_OUT="${BASE_OUT:-$ROOT/outputs/metbench_chess/single_piece_tracking_v1}"
MANIFEST="${MANIFEST:-$BASE_OUT/manifest.csv}"
PILOT_MANIFEST="${PILOT_MANIFEST:-$OUT_DIR/pilot_manifest.csv}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"; NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_DISCOVERY_GAMES="${MAX_DISCOVERY_GAMES:-200}"; MAX_VALIDATION_GAMES="${MAX_VALIDATION_GAMES:-100}"; MAX_T="${MAX_T:-10}"
SEED="${SEED:-20260918}"; QWEN_DIR="${MODEL_DIR_QWEN:-$ROOT/models/Qwen3-VL-8B-Instruct}"; PY="${PYTHON:-python}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; (( ${#GPUS[@]} <= 4 )) || { echo '最多4卡' >&2; exit 2; }; mkdir -p "$OUT_DIR"
prepare() { "$PY" "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT_DIR" pilot-manifest --source-manifest "$MANIFEST" --pilot-out "$PILOT_MANIFEST" --max-discovery-games "$MAX_DISCOVERY_GAMES" --max-validation-games "$MAX_VALIDATION_GAMES" --max-t "$MAX_T" --seed "$SEED"; }
run_phase() {
 local phase="$1"; local pids=() i gpu
 for ((i=0;i<NUM_SHARDS;i++)); do gpu="${GPUS[$((i%${#GPUS[@]}))]}"; [[ -f "$OUT_DIR/extract_qwen_${phase}_${i}.complete.json" && "${OVERWRITE:-0}" != 1 ]] && continue
  local native=(); [[ "$phase" == validation ]] && native=(--run-native-state)
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT_DIR" --manifest "$PILOT_MANIFEST" extract --model qwen --model-dir "$QWEN_DIR" --phase "$phase" --shard-index "$i" --num-shards "$NUM_SHARDS" --max-games 0 --max-t "$MAX_T" "${native[@]}" $(if [[ "${OVERWRITE:-0}" == 1 ]]; then echo --overwrite; fi) >"$OUT_DIR/extract_qwen_${phase}_${i}.log" 2>&1 & pids+=("$!")
 done
 for p in "${pids[@]}"; do wait "$p"; done
 "$PY" "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT_DIR" merge --model qwen --phase "$phase" --num-shards "$NUM_SHARDS"
}
case "${1:-all}" in
 prepare) prepare ;;
 extract) prepare; run_phase discovery; run_phase validation ;;
 all) prepare; run_phase discovery; run_phase validation; "$PY" "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT_DIR" --manifest "$PILOT_MANIFEST" fit --model qwen --candidate-layers "${CANDIDATE_LAYERS:-0}" --candidate-c "${CANDIDATE_C:-0.1,1,10}" --epochs "${UPDATER_EPOCHS:-3}"; "$PY" "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT_DIR" --manifest "$PILOT_MANIFEST" evaluate --model qwen --eval-split validation ;;
 *) echo "usage: $0 {prepare|extract|all}"; exit 2 ;;
esac
