#!/usr/bin/env bash
set -euo pipefail

# Complete L24 head-localization campaign.
# The script intentionally runs discovery before validation and candidate-group
# experiments.  Candidate heads are read only from discovery output.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/head_localization_l24_v1}"
SOURCE_DIR="${SOURCE_DIR:-outputs/vetbench/circuit_localization_v2}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_env() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ env '
    printf '%q ' "$@"
    printf '\n'
  else
    env "$@"
  fi
}

stage_complete() {
  local stage="$1"
  local condition="$2"
  python3 - "$OUT_DIR" "$stage" "$condition" <<'PY'
import csv, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); stage=sys.argv[2]; condition=sys.argv[3]
manifest=json.loads((root/'discovery_pair_manifest.json').read_text())
expected={p['pair_id'] for p in manifest['discovery_pairs' if stage=='discovery' else 'validation_pairs']}
paths=sorted((root/'shards'/stage).glob(f'shard_*/head_{condition}_{stage}.csv'))
seen=[]; good=True
for p in paths:
    with p.open(newline='') as f: seen.extend(csv.DictReader(f))
ids=[r['pair_id'] for r in seen]
if set(ids)!=expected or len(ids)!=len(expected)*64 or len(set((r['pair_id'],r['head'],r['direction']) for r in seen)) != len(ids):
    good=False
print('1' if good else '0')
PY
}

run_stage_condition() {
  local stage="$1"; local condition="$2"
  if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ "$(stage_complete "$stage" "$condition")" == "1" ]]; then
    echo "skip complete $stage $condition"
    return 0
  fi
  run_env STAGE="$stage" CONDITION="$condition" HEADS=all \
    MODEL_DIR="$MODEL_DIR" DATASET_DIR="$DATASET_DIR" OUT_DIR="$OUT_DIR" \
    GPU_IDS="$GPU_IDS" bash scripts/run_head_localization_l24_8gpu.sh
}

echo "[1/6] prepare and structure audit"
run_cmd python3 scripts/head_localization_l24.py prepare \
  --source "$SOURCE_DIR" --out "$OUT_DIR"
run_cmd python3 scripts/head_localization_l24.py audit \
  --model-dir "$MODEL_DIR" --out "$OUT_DIR"
run_cmd python3 scripts/head_localization_l24.py unit

conditions=(target_self same_event_source matched_history_transplant source_current_transplant)

echo "[2/6] discovery: all 32 heads and all controls"
for condition in "${conditions[@]}"; do
  run_stage_condition discovery "$condition"
done

echo "[3/6] discovery head selection"
run_cmd python3 scripts/analyze_head_localization_l24.py --root "$OUT_DIR" --stage discovery

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: candidate groups are not available until discovery analysis runs."
  exit 0
fi

event_heads="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(map(str,d.get("event_routing_heads",[]))))' "$OUT_DIR/candidate_heads.json")"
state_heads="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(map(str,d.get("event_to_state_heads",[]))))' "$OUT_DIR/candidate_heads.json")"

echo "discovery event-routing heads: ${event_heads:-<none>}"
echo "discovery event-to-state heads: ${state_heads:-<none>}"

echo "[4/6] validation: all 32 heads and all controls"
for condition in "${conditions[@]}"; do
  run_stage_condition validation "$condition"
done

echo "[5/6] validation per-head analysis"
run_cmd python3 scripts/analyze_head_localization_l24.py --root "$OUT_DIR" --stage all

run_joint_group() {
  local label="$1"
  local heads="$2"
  if [[ -z "$heads" ]]; then
    echo "skip empty candidate group: $label"
    return 0
  fi
  echo "[joint] $label heads=$heads discovery"
  for condition in "${conditions[@]}"; do
    run_env STAGE=discovery CONDITION="$condition" HEADS="$heads" \
      JOINT_LABEL="$label" MODEL_DIR="$MODEL_DIR" DATASET_DIR="$DATASET_DIR" \
      OUT_DIR="$OUT_DIR" GPU_IDS="$GPU_IDS" bash scripts/run_head_localization_l24_8gpu.sh
  done
  echo "[joint] $label heads=$heads validation"
  for condition in "${conditions[@]}"; do
    run_env STAGE=validation CONDITION="$condition" HEADS="$heads" \
      JOINT_LABEL="$label" MODEL_DIR="$MODEL_DIR" DATASET_DIR="$DATASET_DIR" \
      OUT_DIR="$OUT_DIR" GPU_IDS="$GPU_IDS" bash scripts/run_head_localization_l24_8gpu.sh
  done
}

echo "[6/6] frozen candidate-group joint interventions"
run_joint_group event_routing "$event_heads"
run_joint_group event_to_state "$state_heads"

run_cmd python3 scripts/analyze_head_joint_l24.py --root "$OUT_DIR"
echo "L24 head-localization campaign completed. Outputs: $OUT_DIR"
