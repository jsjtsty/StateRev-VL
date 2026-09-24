#!/usr/bin/env bash
# Addendum S: fixes with the pixel-fraction change detector (GPU0: CPM, GPU1: PCD 8B then 9B)
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( CUDA_VISIBLE_DEVICES=0 python scripts/tracking_algo_fix.py --model qwen3vl8b --cpm > $L/fix_cpm_qwen3vl8b.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 python scripts/tracking_algo_fix.py --model qwen3vl8b > $L/fix_pcd_qwen3vl8b.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python scripts/tracking_algo_fix.py --model qwen35_9b > $L/fix_pcd_qwen35_9b.log 2>&1 ) &
wait
