#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/path_dependence_v1}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
MIN_T="${MIN_T:-3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT="${SPLIT:-outputs/vetbench/circuit_localization_v2/discovery_validation_split.json}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]]; then
  echo "GPU_IDS has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

echo "[1/4] prepare fixed manifest (CPU-only)"
"${PYTHON_BIN}" scripts/path_dependence_experiment.py prepare \
  --out "${OUT_DIR}" --behavior outputs/vetbench/composition_analysis_v1/transformers_behavior.csv \
  --split "${SPLIT}" \
  --dataset "${DATASET_DIR}" --model-dir "${MODEL_DIR}" --min-t "${MIN_T}"

echo "[2/4] audit manifest and run unit checks (CPU-only)"
"${PYTHON_BIN}" scripts/path_dependence_experiment.py audit --out "${OUT_DIR}"
"${PYTHON_BIN}" scripts/path_dependence_experiment.py unit

mkdir -p "${OUT_DIR}/shards"
echo "[3/4] launch ${NUM_SHARDS} shards on GPUs: ${GPU_IDS}"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$shard]}"
  log="${OUT_DIR}/shards/path_shard_${shard}.log"
  echo "starting shard ${shard}/${NUM_SHARDS} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/path_dependence_experiment.py run \
    --out "${OUT_DIR}" --behavior outputs/vetbench/composition_analysis_v1/transformers_behavior.csv \
    --split "${SPLIT}" \
    --dataset "${DATASET_DIR}" --model-dir "${MODEL_DIR}" \
    --baseline-hidden outputs/vetbench/mechanism_gate_final/hidden_baseline.npz \
    --min-t "${MIN_T}" --shard-index "${shard}" --num-shards "${NUM_SHARDS}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one shard failed. Inspect ${OUT_DIR}/shards/path_shard_*.log" >&2
  exit "${status}"
fi

echo "[4/4] verify coverage and run offline clustered analysis"
"${PYTHON_BIN}" scripts/analyze_path_dependence.py --out "${OUT_DIR}" --num-shards "${NUM_SHARDS}"
echo "complete: ${OUT_DIR}/path_dependence_report.md"
