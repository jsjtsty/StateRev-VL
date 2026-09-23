#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/history_readout_mediation_v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
PATH_MANIFEST="${PATH_MANIFEST:-outputs/vetbench/path_dependence_v1/path_dependence_manifest.json}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]]; then
  echo "GPU_IDS count ${#GPUS[@]} != NUM_SHARDS ${NUM_SHARDS}" >&2
  exit 2
fi

echo "[1/4] prepare fixed mediation manifest"
"${PYTHON_BIN}" scripts/history_readout_mediation.py prepare --out "${OUT_DIR}" --path-manifest "${PATH_MANIFEST}"
echo "[2/4] audit and unit"
"${PYTHON_BIN}" scripts/history_readout_mediation.py audit --out "${OUT_DIR}" --dataset "${DATASET_DIR}"
"${PYTHON_BIN}" scripts/history_readout_mediation.py unit

mkdir -p "${OUT_DIR}/shards"
echo "[3/4] launch ${NUM_SHARDS} shards on GPUs ${GPU_IDS}"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$shard]}"
  log="${OUT_DIR}/shards/mediation_shard_${shard}.log"
  echo "starting shard ${shard} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/history_readout_mediation.py run \
    --out "${OUT_DIR}" --shard-index "${shard}" --num-shards "${NUM_SHARDS}" \
    --model-dir "${MODEL_DIR}" --dataset "${DATASET_DIR}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "A shard failed; inspect ${OUT_DIR}/shards/mediation_shard_*.log" >&2
  exit "${status}"
fi

echo "[4/4] offline clustered analysis"
"${PYTHON_BIN}" scripts/analyze_history_readout_mediation.py --out "${OUT_DIR}" --num-shards "${NUM_SHARDS}"
echo "complete: ${OUT_DIR}/history_readout_mediation_report.md"
