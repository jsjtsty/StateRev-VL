#!/usr/bin/env bash
# non-Qwen-vision models: native + PCD (fix.py), image-only controls; GPU1 after the Qwen3.5 PCD run
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "N_ball" $L/fix_pcd_qwen35_9b.log 2>/dev/null; do sleep 30; done
for m in internvl35_8b llava_ov7b; do
  if [ "$m" = llava_ov7b ]; then until grep -q "Download completed successfully" $L/hfd_llava-onevision-qwen2-7b-ov-hf.log; do sleep 60; done; fi
  CUDA_VISIBLE_DEVICES=1 python scripts/tracking_algo_fix.py --model $m > $L/fix_pcd_$m.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python scripts/tracking_algo_anchor.py --model $m --sets O,M --image-only > $L/anchor_imgonly_$m.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python scripts/tracking_algo_chess_now.py --model $m --captures --image-only > $L/chesscap_${m}_img.log 2>&1
done
