#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
BUNDLE="${BUNDLE:-state_rev_history_readout_mediation_v1.tar.gz}"

for path in \
  scripts/history_readout_mediation.py \
  scripts/analyze_history_readout_mediation.py \
  scripts/run_history_readout_mediation_8gpu.sh \
  scripts/state_rev_input_pipeline.py \
  scripts/mg_common.py \
  scripts/run_state_rev_audit.py \
  scripts/run_vetbench_screening.py \
  outputs/vetbench/path_dependence_v1/path_dependence_manifest.json \
  outputs/vetbench/circuit_localization_v2/discovery_validation_split.json \
  outputs/vetbench/composition_analysis_v1/transformers_behavior.csv \
  outputs/vetbench/mechanism_gate_final/hidden_baseline.npz; do
  [[ -f "${path}" ]] || { echo "missing required bundle file: ${path}" >&2; exit 1; }
done

tar -czf "${BUNDLE}" \
  scripts/history_readout_mediation.py \
  scripts/analyze_history_readout_mediation.py \
  scripts/run_history_readout_mediation_8gpu.sh \
  scripts/state_rev_input_pipeline.py \
  scripts/mg_common.py \
  scripts/run_state_rev_audit.py \
  scripts/run_vetbench_screening.py \
  outputs/vetbench/path_dependence_v1/path_dependence_manifest.json \
  outputs/vetbench/circuit_localization_v2/discovery_validation_split.json \
  outputs/vetbench/composition_analysis_v1/transformers_behavior.csv \
  outputs/vetbench/mechanism_gate_final/hidden_baseline.npz
echo "created ${BUNDLE}"
du -h "${BUNDLE}"
