#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"; DATASET_DIR="${DATASET_DIR:-dataset/vetbench/cup}"; OUT_DIR="${OUT_DIR:-outputs/vetbench/head0_downstream_delta_rescue_v1}"; JOINT_MANIFEST="${JOINT_MANIFEST:-outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_manifest.json}"; GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"; FORCE_RERUN="${FORCE_RERUN:-0}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; NUM_SHARDS="${#GPUS[@]}"; [[ "$NUM_SHARDS" -gt 0 ]] || { echo "GPU_IDS is empty" >&2; exit 2; }
echo "[1/4] freeze and audit manifest"; python3 scripts/head0_downstream_delta_rescue.py prepare --out "$OUT_DIR" --joint-manifest "$JOINT_MANIFEST" --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR"; python3 scripts/head0_downstream_delta_rescue.py audit --out "$OUT_DIR" --joint-manifest "$JOINT_MANIFEST" --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR"; python3 scripts/head0_downstream_delta_rescue.py unit --out "$OUT_DIR"
complete(){ local i="$1"; python3 - "$OUT_DIR" "$i" "$NUM_SHARDS" <<'PY'
import json,pathlib,sys,pandas as pd
r=pathlib.Path(sys.argv[1]); i=int(sys.argv[2]); n=int(sys.argv[3]); m=json.loads((r/'delta_rescue_manifest.json').read_text()); exp={x['target_prefix'] for x in m['targets'][i::n]}; p=r/'shards'/f'delta_rescue_shard_{i}.csv'
if not p.exists(): print(0); raise SystemExit
try:
 d=pd.read_csv(p); ok=set(d.target_prefix)==exp and len(d)==len(exp)*len(m['donor_types'])*len(m['betas'])*len(m['conditions']) and not d.duplicated(['target_prefix','donor_type','beta','condition']).any() and set(d.condition)==set(m['conditions']) and set(d.donor_type)==set(m['donor_types']) and set(d.beta)==set(m['betas'])
except Exception: ok=False
print(1 if ok else 0)
PY
}
echo "[2/4] launch $NUM_SHARDS shards on GPUs: $GPU_IDS"; mkdir -p "$OUT_DIR/shards"; pids=()
for i in "${!GPUS[@]}"; do log="$OUT_DIR/shards/delta_rescue_shard_${i}.log"; if [[ "$FORCE_RERUN" != 1 && "$(complete "$i")" == 1 ]]; then echo "skip verified-complete shard $i"; continue; fi; CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python3 scripts/head0_downstream_delta_rescue.py run --out "$OUT_DIR" --model-dir "$MODEL_DIR" --dataset "$DATASET_DIR" --shard-index "$i" --num-shards "$NUM_SHARDS" >"$log" 2>&1 & pids+=("$!"); echo "started shard $i on GPU ${GPUS[$i]} (pid ${pids[-1]})"; done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done; [[ "$failed" == 0 ]] || { echo "delta-rescue shard failed; inspect logs" >&2; exit 1; }
echo "[3/4] verify shards"; for i in "${!GPUS[@]}"; do [[ "$(complete "$i")" == 1 ]] || { echo "incomplete shard $i" >&2; exit 1; }; done
echo "[4/4] offline clustered analysis"; python3 scripts/analyze_head0_downstream_delta_rescue.py --out "$OUT_DIR" --num-shards "$NUM_SHARDS"; echo "Composable delta rescue complete: $OUT_DIR"
