#!/usr/bin/env bash
set -euo pipefail

# Offline learned-state-updater driver. All inputs are existing caches; this
# script does not launch VLM forward jobs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${ROOT}/scripts/learned_state_updater.py"
OUT="${OUT:-${ROOT}/outputs/vetbench/learned_state_updater_v1}"
LOG_DIR="${OUT}/logs"
MODE="${1:-help}"

mkdir -p "${OUT}" "${LOG_DIR}"
cd "${ROOT}"

run_logged() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  echo "[$(date '+%F %T')] CMD $*" | tee "${LOG_DIR}/${name}.cmd"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] DONE ${name}"
}

fit_all() {
  run_logged fit_qwen "${PYTHON_BIN}" "${SCRIPT}" fit \
    --model qwen --out "${OUT}"
  run_logged fit_llava "${PYTHON_BIN}" "${SCRIPT}" fit \
    --model llava --out "${OUT}"
}

fit_parallel() {
  # The updater is CPU/offline. This is process-level parallelism, not VLM
  # multi-GPU inference; each model writes disjoint artifact names.
  run_logged fit_qwen "${PYTHON_BIN}" "${SCRIPT}" fit \
    --model qwen --out "${OUT}" &
  local qwen_pid=$!
  run_logged fit_llava "${PYTHON_BIN}" "${SCRIPT}" fit \
    --model llava --out "${OUT}" &
  local llava_pid=$!
  wait "${qwen_pid}"
  wait "${llava_pid}"
}

evaluate_all() {
  run_logged eval_qwen_to_qwen "${PYTHON_BIN}" "${SCRIPT}" evaluate \
    --source-model qwen --target-model qwen \
    --artifact "${OUT}/qwen_updater.joblib" --out "${OUT}"
  run_logged eval_llava_to_llava "${PYTHON_BIN}" "${SCRIPT}" evaluate \
    --source-model llava --target-model llava \
    --artifact "${OUT}/llava_updater.joblib" --out "${OUT}"
  run_logged eval_qwen_to_llava "${PYTHON_BIN}" "${SCRIPT}" evaluate \
    --source-model qwen --target-model llava \
    --artifact "${OUT}/qwen_updater.joblib" --out "${OUT}"
  run_logged eval_llava_to_qwen "${PYTHON_BIN}" "${SCRIPT}" evaluate \
    --source-model llava --target-model qwen \
    --artifact "${OUT}/llava_updater.joblib" --out "${OUT}"
}

curves_all() {
  run_logged curve_qwen "${PYTHON_BIN}" "${SCRIPT}" curve \
    --model qwen --out "${OUT}"
  run_logged curve_llava "${PYTHON_BIN}" "${SCRIPT}" curve \
    --model llava --out "${OUT}"
}

case "${MODE}" in
  fit)
    fit_all
    ;;
  fit-parallel)
    fit_parallel
    ;;
  evaluate)
    evaluate_all
    ;;
  curves)
    curves_all
    ;;
  all)
    fit_all
    evaluate_all
    curves_all
    ;;
  all-parallel-fit)
    fit_parallel
    evaluate_all
    curves_all
    ;;
  help|-h|--help)
    cat <<EOF
Usage: $0 {fit|fit-parallel|evaluate|curves|all|all-parallel-fit}

Environment:
  PYTHON_BIN=python                         Python executable
  OUT=outputs/vetbench/learned_state_updater_v1

Modes:
  fit               Fit Qwen and LLaVA updater artifacts sequentially.
  fit-parallel      Fit both artifacts concurrently using CPU processes.
  evaluate          Run Qwen->Qwen, LLaVA->LLaVA, Qwen->LLaVA,
                    and LLaVA->Qwen validation evaluations.
  curves            Run 1/2/5/10/20/30-trajectory curves, 5 repeats.
  all               Fit, evaluate, then run curves sequentially.
  all-parallel-fit  Parallelize only the two fitting jobs, then continue.

Logs are written under: 
  ${OUT}/logs/
EOF
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    "$0" --help >&2
    exit 2
    ;;
esac
