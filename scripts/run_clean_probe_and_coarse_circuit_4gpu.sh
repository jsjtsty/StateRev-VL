#!/usr/bin/env bash
# Content-disjoint ordinary probe (CPU/cache only) + coarse circuit sweep (4 GPUs).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SPLIT="${SPLIT:-outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json}"
FINGERPRINTS="${FINGERPRINTS:-outputs/vetbench/validity_gate_v1/input_fingerprints.csv}"
PROBE_OUT="${PROBE_OUT:-outputs/vetbench/ordinary_probe_content_disjoint_v1}"
CIRCUIT_OUT="${CIRCUIT_OUT:-outputs/vetbench/circuit_localization_content_disjoint_v2}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ "${#GPUS[@]}" -eq 4 ]] || { echo "This runner requires exactly four GPU IDs; got: $GPU_IDS" >&2; exit 2; }
[[ -f "$SPLIT" ]] || { echo "missing split: $SPLIT" >&2; exit 1; }
[[ ! -e "$PROBE_OUT" ]] || { echo "refusing to overwrite probe output: $PROBE_OUT" >&2; exit 1; }
[[ ! -e "$CIRCUIT_OUT" ]] || { echo "refusing to overwrite circuit output: $CIRCUIT_OUT" >&2; exit 1; }

echo "[0/5] content-disjoint assertion"
"$PYTHON_BIN" scripts/content_disjoint_split.py audit \
  --split "$SPLIT" --fingerprints "$FINGERPRINTS"

echo "[1/5] ordinary probe: cache-only, discovery fit / frozen validation"
"$PYTHON_BIN" scripts/content_disjoint_ordinary_probe.py \
  --out "$PROBE_OUT" --split "$SPLIT" --fingerprints "$FINGERPRINTS"

echo "[2/5] prepare coarse circuit manifest"
"$PYTHON_BIN" scripts/circuit_localization.py prepare \
  --out "$CIRCUIT_OUT" --source outputs/vetbench/mechanism_gate_final --split "$SPLIT"

echo "[3/5] coarse circuit discovery on four GPUs"
OUT_DIR="$CIRCUIT_OUT" STAGE=discovery GPU_IDS="$GPU_IDS" \
  MODEL_DIR="$MODEL_DIR" DATASET_DIR="$DATASET_DIR" \
  bash scripts/run_circuit_localization_8gpu.sh

echo "[4/5] coarse circuit frozen validation on four GPUs"
OUT_DIR="$CIRCUIT_OUT" STAGE=validation GPU_IDS="$GPU_IDS" \
  MODEL_DIR="$MODEL_DIR" DATASET_DIR="$DATASET_DIR" \
  bash scripts/run_circuit_localization_8gpu.sh

echo "[5/5] merge and clustered analysis"
"$PYTHON_BIN" scripts/analyze_circuit_localization.py \
  --out "$CIRCUIT_OUT" --module block_output
"$PYTHON_BIN" scripts/analyze_circuit_localization_v2.py \
  --root "$CIRCUIT_OUT" --out "$CIRCUIT_OUT" --module block_output

echo "complete"
echo "ordinary probe: $PROBE_OUT/ordinary_probe_report.md"
echo "coarse circuit: $CIRCUIT_OUT/circuit_localization_report.md"
