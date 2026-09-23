#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/hidden_event_recursive_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MODE="${MODE:-multi}"

COMMON=(
  --out "${OUT_DIR}"
  --behavior "${BEHAVIOR:-outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv}"
  --split "${SPLIT:-outputs/vetbench/circuit_localization_v2/discovery_validation_split.json}"
  --hidden-cache "${HIDDEN_CACHE:-outputs/vetbench/hidden_state_probe/hidden_states.npz}"
  --native-event-prob-dir "${NATIVE_EVENT_PROB_DIR:-outputs/vetbench/recursive_state_recovery_v1/shards}"
  --symbolic "${SYMBOLIC:-outputs/vetbench/composition_analysis_v1/symbolic_composition.csv}"
)

# Content-disjoint manifests embed their fingerprint source and therefore
# assert it automatically. This optional override supports relocated inputs.
if [[ -n "${FINGERPRINT_FILE:-}" ]]; then
  COMMON+=(--fingerprint-file "${FINGERPRINT_FILE}")
fi

if [[ "${MODE}" != "single" && "${MODE}" != "multi" ]]; then
  echo "MODE must be single or multi" >&2
  exit 2
fi

echo "[1/4] schema check"
"${PYTHON_BIN}" scripts/hidden_event_recursive_tracking.py schema-check "${COMMON[@]}"

echo "[2/4] discovery-only decoder fit"
"${PYTHON_BIN}" scripts/hidden_event_recursive_tracking.py fit "${COMMON[@]}" \
  --candidate-layers "${CANDIDATE_LAYERS:-24,28}" \
  --candidate-pca-dims "${CANDIDATE_PCA_DIMS:-0,40}" \
  --candidate-c "${CANDIDATE_C:-0.1,1.0,10.0}"

if [[ "${MODE}" == "single" ]]; then
  echo "[3/4] frozen validation prediction (single process)"
  "${PYTHON_BIN}" scripts/hidden_event_recursive_tracking.py predict "${COMMON[@]}"
else
  IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
  if [[ "${#GPU_LIST[@]}" -lt "${NUM_SHARDS}" ]]; then
    echo "NUM_SHARDS=${NUM_SHARDS} exceeds GPU_IDS count ${#GPU_LIST[@]}" >&2
    exit 2
  fi
  mkdir -p "${OUT_DIR}/shards"
  echo "[3/4] frozen validation prediction (${NUM_SHARDS} trajectory shards)"
  pids=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_LIST[$shard]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/hidden_event_recursive_tracking.py predict \
      "${COMMON[@]}" --shard-index "${shard}" --num-shards "${NUM_SHARDS}" \
      > "${OUT_DIR}/shards/prediction_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" != 0 ]]; then
    echo "A prediction shard failed; merge was not attempted" >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/hidden_event_recursive_tracking.py merge "${COMMON[@]}" --num-shards "${NUM_SHARDS}"
fi

echo "[4/4] frozen validation analysis"
"${PYTHON_BIN}" scripts/hidden_event_recursive_tracking.py analyze "${COMMON[@]}"
echo "hidden-event recursive tracking complete: ${OUT_DIR}"
