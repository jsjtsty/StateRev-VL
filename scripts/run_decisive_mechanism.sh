#!/usr/bin/env bash
# Launch the final StateRev-VL mechanism experiment in a fixed order.
# Override GPU_ID/MODEL_DIR/OUT_DIR when running on another machine.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS="${NUM_GPUS:-${#GPU_LIST[@]}}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
BEHAVIOR_CSV="${BEHAVIOR_CSV:-outputs/vetbench/composition_analysis_v1/transformers_behavior.csv}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/mechanism_gate_final}"

echo "[1/5] Static preflight"
python -m py_compile \
  scripts/audit_probe_discrepancy.py \
  scripts/decisive_mechanism_experiment.py \
  scripts/analyze_decisive_mechanism.py \
  scripts/mg_common.py

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "[2/5] Reconcile existing probe artifacts"
  python scripts/audit_probe_discrepancy.py \
    --out "${OUT_DIR}/probe_discrepancy_audit.json"

  echo "[3/5] Build fixed eligible-pair manifest"
  python scripts/decisive_mechanism_experiment.py manifest \
    --behavior "${BEHAVIOR_CSV}" \
    --out "${OUT_DIR}"

  echo "[4/5] Dry-run plan"
  python scripts/decisive_mechanism_experiment.py dry-run \
    --behavior "${BEHAVIOR_CSV}" \
    --out "${OUT_DIR}"
else
  echo "[2-4/5] Preflight skipped (SKIP_PREFLIGHT=1)"
  [[ -f "${OUT_DIR}/pair_manifest.json" ]] || {
    echo "Missing ${OUT_DIR}/pair_manifest.json" >&2
    exit 1
  }
fi

echo "[5/5] ${NUM_GPUS}-GPU experiment and clustered offline analysis"
if (( NUM_GPUS < 1 || NUM_GPUS > ${#GPU_LIST[@]} )); then
  echo "NUM_GPUS=${NUM_GPUS} requires that many comma-separated GPU_IDS" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/shards"
PIDS=()
for (( shard=0; shard<NUM_GPUS; shard++ )); do
  shard_out="${OUT_DIR}/shards/shard_${shard}"
  mkdir -p "${shard_out}"
  gpu="${GPU_LIST[$shard]}"
  echo "Starting shard ${shard}/${NUM_GPUS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/decisive_mechanism_experiment.py run \
    --model-dir "${MODEL_DIR}" \
    --dataset "${DATASET_DIR}" \
    --behavior "${BEHAVIOR_CSV}" \
    --manifest "${OUT_DIR}/pair_manifest.json" \
    --out "${shard_out}" \
    --shard-index "${shard}" \
    --num-shards "${NUM_GPUS}" \
    > "${shard_out}/run.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if (( status != 0 )); then
  echo "At least one GPU shard failed; inspect ${OUT_DIR}/shards/shard_*/run.log" >&2
  exit "${status}"
fi

python scripts/decisive_mechanism_experiment.py merge \
  --out "${OUT_DIR}" \
  --num-shards "${NUM_GPUS}"

python scripts/analyze_decisive_mechanism.py \
  --out "${OUT_DIR}"

echo "Done. Results: ${OUT_DIR}"
