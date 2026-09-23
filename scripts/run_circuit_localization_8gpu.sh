#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

STAGE="${STAGE:-discovery}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/circuit_localization_v2}"
PATCH_MODULE="${PATCH_MODULE:-block_output}"
LAYERS="${LAYERS:-4,8,12,16,20,24,28,32,36}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_LIST[@]}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi
if [[ "${STAGE}" != "discovery" && "${STAGE}" != "validation" ]]; then
  echo "STAGE must be discovery or validation" >&2
  exit 2
fi
[[ -f "${OUT_DIR}/discovery_validation_split.json" ]] || {
  echo "Missing fixed split; run circuit_localization.py prepare first" >&2
  exit 1
}
[[ -f "${OUT_DIR}/discovery_pair_manifest.json" ]] || {
  echo "Missing fixed pair manifest; run circuit_localization.py prepare first" >&2
  exit 1
}
mkdir -p "${OUT_DIR}/shards/${STAGE}"

pids=()
for shard in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard]}"
  shard_dir="${OUT_DIR}/shards/${STAGE}/shard_${shard}"
  mkdir -p "${shard_dir}"
  echo "starting ${STAGE} shard ${shard}/${NUM_GPUS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_circuit_localization.py \
    --stage "${STAGE}" \
    --model-dir "${MODEL_DIR}" \
    --dataset "${DATASET_DIR}" \
    --out "${OUT_DIR}" \
    --shard-index "${shard}" \
    --num-shards "${NUM_GPUS}" \
    --patch-module "${PATCH_MODULE}" \
    --layers "${LAYERS}" \
    > "${shard_dir}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one shard failed; inspect ${OUT_DIR}/shards/${STAGE}/shard_*/run.log" >&2
  exit "${status}"
fi
echo "${STAGE} completed; shard outputs are under ${OUT_DIR}/shards/${STAGE}/"
