#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"
OUT_DIR="${OUT_DIR:-outputs/vetbench/head0_causal_rescue_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SPLIT="${SPLIT:-outputs/vetbench/head_localization_l24_v1/discovery_validation_split.json}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_SHARDS="${#GPUS[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

shard_complete() {
  local index="$1"
  python3 - "$OUT_DIR" "$index" "$NUM_SHARDS" <<'PY'
import json, pathlib, sys
import pandas as pd
root=pathlib.Path(sys.argv[1]); index=int(sys.argv[2]); nshards=int(sys.argv[3])
manifest=json.loads((root/'rescue_manifest.json').read_text())
expected={row['target_prefix'] for row in manifest['targets'][index::nshards]}
path=root/'shards'/f'rescue_shard_{index}.csv'
if not path.exists():
    print(0); raise SystemExit
try:
    data=pd.read_csv(path)
    valid=(set(data['target_prefix'])==expected and len(data)==len(expected)*5
           and not data.duplicated(['target_prefix','condition']).any()
           and data.groupby('target_prefix')['condition'].nunique().eq(5).all())
except Exception:
    valid=False
print(1 if valid else 0)
PY
}

echo "[1/4] freeze rescue split, groups, and donor manifest"
python3 scripts/head0_causal_rescue.py prepare \
  --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" --out "$OUT_DIR" --split "$SPLIT"
python3 scripts/head0_causal_rescue.py audit \
  --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" --out "$OUT_DIR" --split "$SPLIT"
python3 scripts/head0_causal_rescue.py unit --out "$OUT_DIR"

echo "[2/4] launch $NUM_SHARDS shards on GPUs: $GPU_IDS"
mkdir -p "$OUT_DIR/shards"
pids=()
for index in "${!GPUS[@]}"; do
  log="$OUT_DIR/shards/rescue_shard_${index}.log"
  if [[ "$FORCE_RERUN" != "1" && "$(shard_complete "$index")" == "1" ]]; then
    echo "skip verified-complete shard $index"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" \
    python3 scripts/head0_causal_rescue.py run \
      --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" --out "$OUT_DIR" \
      --split "$SPLIT" \
      --shard-index "$index" --num-shards "$NUM_SHARDS" \
      >"$log" 2>&1 &
  pids+=("$!")
  echo "started shard $index on GPU ${GPUS[$index]} (pid ${pids[-1]})"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" == "1" ]]; then
  echo "At least one rescue shard failed. Inspect $OUT_DIR/shards/rescue_shard_*.log" >&2
  exit 1
fi

echo "[3/4] verify shard completion"
for index in "${!GPUS[@]}"; do
  [[ "$(shard_complete "$index")" == "1" ]] || {
    echo "missing or incomplete shard $index" >&2
    exit 1
  }
done

echo "[4/4] offline clustered analysis"
python3 scripts/analyze_head0_causal_rescue.py --out "$OUT_DIR" --num-shards "$NUM_SHARDS"
echo "Head 0 causal rescue complete: $OUT_DIR"
