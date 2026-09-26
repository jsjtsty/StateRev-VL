#!/usr/bin/env bash
# Addendum U: native + PCD and image-only, 5 models
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { CUDA_VISIBLE_DEVICES=$1 python scripts/tracking_algo_fix.py --model $2 --only U --device-map $3 > $L/fixU_$2.log 2>&1
        CUDA_VISIBLE_DEVICES=$1 python scripts/tracking_algo_fix.py --model $2 --only U --image-only --device-map $3 > $L/fixU_img_$2.log 2>&1; }
( run 0 qwen3vl8b cuda:0; run 0 qwen35_9b cuda:0 ) &
( run 1 internvl35_8b cuda:0; run 1 llava_ov7b cuda:0 ) &
( run 2,3 qwen3vl32b auto ) &
wait
