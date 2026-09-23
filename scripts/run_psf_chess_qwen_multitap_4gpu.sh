#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT_DIR:-$ROOT/outputs/psf_v1/formal_precheck/chess_qwen_multitap}"
MANIFEST="${MANIFEST:-$ROOT/outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1/pilot_manifest.csv}"
MODEL="${MODEL_DIR:-$ROOT/models/Qwen3-VL-8B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"; IFS=',' read -ra GS <<< "$GPU_IDS"; N=${#GS[@]}; mkdir -p "$OUT"
for phase in discovery validation; do
 pids=()
 for ((i=0;i<N;i++)); do
  CUDA_VISIBLE_DEVICES="${GS[$i]}" python "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT" --manifest "$MANIFEST" extract --model qwen --model-dir "$MODEL" --phase "$phase" --shard-index "$i" --num-shards "$N" --max-t 10 --layer-ids 9,18,27,36 >"$OUT/extract_qwen_${phase}_${i}.log" 2>&1 & pids+=("$!")
 done
 for p in "${pids[@]}"; do wait "$p"; done
 python "$ROOT/scripts/metbench_chess_experiment.py" --out "$OUT" --manifest "$MANIFEST" merge --model qwen --phase "$phase" --num-shards "$N"
done
python - <<'PY' "$OUT"
import sys,json,numpy as np
from pathlib import Path
o=Path(sys.argv[1]); out={}
for phase in ('discovery','validation'):
 z=np.load(o/f'hidden_qwen_{phase}.npz'); shapes={tuple(v.shape) for v in z.values()}
 assert shapes=={(4,4096)}, shapes
 out[phase]={'keys':len(z.files),'shapes':[4,4096]}
(o/'multitap_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'MULTITAP_AUDIT_PASS':True,**out}))
PY
