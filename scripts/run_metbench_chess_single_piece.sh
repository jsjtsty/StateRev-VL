#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/dataset/MET-Bench-Chess}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/metbench_chess/single_piece_tracking_v1}"
MODEL_DIR_QWEN="${MODEL_DIR_QWEN:-$ROOT/models/Qwen3-VL-8B-Instruct}"
MODEL_DIR_LLAVA="${MODEL_DIR_LLAVA:-$ROOT/models/LLaVA-NeXT-Video-7B-hf}"

case "${1:-help}" in
  prepare)
    python "$ROOT/scripts/metbench_chess_single_piece.py" --data-dir "$DATA_DIR" --out "$OUT_DIR" build-manifest
    python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" audit
    ;;
  smoke-qwen)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" model-smoke --model qwen --model-dir "$MODEL_DIR_QWEN" --limit "${LIMIT:-2}"
    ;;
  smoke-llava)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" model-smoke --model llava --model-dir "$MODEL_DIR_LLAVA" --limit "${LIMIT:-2}"
    ;;
  native-qwen)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" native-smoke --model qwen --model-dir "$MODEL_DIR_QWEN" --limit "${LIMIT:-2}"
    ;;
  native-llava)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" native-smoke --model llava --model-dir "$MODEL_DIR_LLAVA" --limit "${LIMIT:-2}"
    ;;
  shard)
    : "${SHARD_INDEX:?set SHARD_INDEX}"; : "${NUM_SHARDS:?set NUM_SHARDS}"
    python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" manifest-shard --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS"
    ;;
  merge)
    : "${NUM_SHARDS:?set NUM_SHARDS}"
    python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" manifest-merge --num-shards "$NUM_SHARDS"
    ;;
  unit)
    python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" unit
    python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" special-unit
    python "$ROOT/scripts/metbench_chess_single_piece.py" --out "$OUT_DIR" shard-unit
    ;;
  help|*)
    echo "usage: $0 {prepare|smoke-qwen|smoke-llava|native-qwen|native-llava|shard|merge|unit}"
    echo "Full native/probe/updater experiments are intentionally not started by this smoke-stage wrapper."
    ;;
esac
