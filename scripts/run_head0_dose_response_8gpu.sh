#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/head0_dose_response_v1}"
RESCUE_MANIFEST="${RESCUE_MANIFEST:-outputs/vetbench/head0_causal_rescue_v1/rescue_manifest.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FORCE_RERUN="${FORCE_RERUN:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_SHARDS="${#GPUS[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

echo "[1/4] freeze and audit existing target-donor pairs"
python3 scripts/head0_dose_response.py prepare \
  --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" \
  --rescue-manifest "$RESCUE_MANIFEST" --out "$OUT_DIR"
python3 scripts/head0_dose_response.py audit \
  --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" \
  --rescue-manifest "$RESCUE_MANIFEST" --out "$OUT_DIR"
python3 scripts/head0_dose_response.py unit --out "$OUT_DIR"

shard_complete() {
  local index="$1"
  python3 - "$OUT_DIR" "$index" "$NUM_SHARDS" <<'PY'
import json, pathlib, sys
import pandas as pd
root=pathlib.Path(sys.argv[1]); index=int(sys.argv[2]); nshards=int(sys.argv[3])
manifest=json.loads((root/'dose_response_manifest.json').read_text())
expected={row['target_prefix'] for row in manifest['targets'][index::nshards]}
path=root/'shards'/f'dose_response_shard_{index}.csv'
if not path.exists():
    print(0); raise SystemExit
try:
    data=pd.read_csv(path)
    valid=(set(data['target_prefix'])==expected and len(data)==len(expected)*12
           and not data.duplicated(['target_prefix','donor_type','alpha']).any()
           and set(data['alpha'])==set(manifest['alphas'])
           and set(data['donor_type'])==set(manifest['donor_types']))
except Exception:
    valid=False
print(1 if valid else 0)
PY
}

echo "[2/4] launch $NUM_SHARDS shards on GPUs: $GPU_IDS"
mkdir -p "$OUT_DIR/shards"
pids=()
for index in "${!GPUS[@]}"; do
  log="$OUT_DIR/shards/dose_response_shard_${index}.log"
  if [[ "$FORCE_RERUN" != "1" && "$(shard_complete "$index")" == "1" ]]; then
    echo "skip verified-complete shard $index"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" \
    python3 scripts/head0_dose_response.py run \
      --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" \
      --rescue-manifest "$RESCUE_MANIFEST" --out "$OUT_DIR" \
      --shard-index "$index" --num-shards "$NUM_SHARDS" \
      >"$log" 2>&1 &
  pids+=("$!")
  echo "started shard $index on GPU ${GPUS[$index]} (pid ${pids[-1]})"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" == "1" ]]; then
  echo "At least one dose-response shard failed. Inspect $OUT_DIR/shards/*.log" >&2
  exit 1
fi

echo "[3/4] verify all shards"
for index in "${!GPUS[@]}"; do
  [[ "$(shard_complete "$index")" == "1" ]] || {
    echo "missing or incomplete shard $index" >&2
    exit 1
  }
done

echo "[4/4] trajectory-clustered offline analysis"
python3 scripts/analyze_head0_dose_response.py --out "$OUT_DIR" --num-shards "$NUM_SHARDS"
echo "Head 0 dose response complete: $OUT_DIR"
