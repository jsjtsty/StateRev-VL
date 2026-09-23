#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/metbench_chess/llava_native_state_audit_v1/corrected_native_max64_user}"
MANIFEST="${MANIFEST:-$ROOT/outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1/pilot_manifest.csv}"
MODEL_DIR="${MODEL_DIR_LLAVA:-$ROOT/models/LLaVA-NeXT-Video-7B-hf}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PYTHON_BIN="${PYTHON:-python}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "GPU_IDS must contain exactly NUM_SHARDS entries: GPU_IDS=$GPU_IDS NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "model directory not found: $MODEL_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"

PIDS=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  marker="$OUT_DIR/corrected_native_${shard}.complete.json"
  if [[ -f "$marker" && "${OVERWRITE:-0}" != 1 ]]; then
    echo "skip completed shard $shard: $marker"
    continue
  fi
  gpu="${GPUS[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/rerun_llava_native_state_corrected.py" \
    --out "$OUT_DIR" \
    --manifest "$MANIFEST" \
    --model-dir "$MODEL_DIR" \
    --shard-index "$shard" \
    --num-shards "$NUM_SHARDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    $([[ "${OVERWRITE:-0}" == 1 ]] && echo --overwrite) \
    > "$OUT_DIR/shard_${shard}.log" 2>&1 &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

MANIFEST_PATH="$MANIFEST" OUT_PATH="$OUT_DIR" "$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from pathlib import Path

out = Path(os.environ["OUT_PATH"])
manifest = Path(os.environ["MANIFEST_PATH"])
markers = sorted(out.glob("corrected_native_*.complete.json"))
assert len(markers) == 4, f"expected 4 completion markers, got {len(markers)}"
sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
games = rows = 0
for p in markers:
    m = json.loads(p.read_text())
    assert m["status"] == "complete"
    assert m["num_shards"] == 4
    assert m["max_new_tokens"] == 64
    assert m["manifest_sha256"] == sha
    games += int(m["games"]); rows += int(m["rows"])
    print(f"{p.name}: games={m['games']} rows={m['rows']}")
assert games == 100, games
assert rows == 948, rows
print("CORRECTED_NATIVE_EXTRACTION_PASS games=100 rows=948")
PY
