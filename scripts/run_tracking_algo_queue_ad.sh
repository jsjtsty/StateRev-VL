#!/usr/bin/env bash
# Addendum AD: timeline factors on synthetic word tiles (GPU3, sequential).
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in qwen3vl8b qwen35_9b internvl35_8b llava_ov7b gemma3_12b; do
  CUDA_VISIBLE_DEVICES=3 $PY scripts/tracking_algo_synthword2.py --model $m > $L/synthword2_$m.log 2>&1; echo "synthword2 $m DONE $(date)" >> $L/ad_done.txt
done
echo "ALL DONE $(date)" >> $L/ad_done.txt
