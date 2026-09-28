#!/bin/bash
# Addendum AH: layer masking on the AG real/syn held clips; one model per GPU.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs; PY=/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_real7.py --model $2 --device-map cuda:0 > $L/real7_$2.log 2>&1 && echo "real7 $2 DONE $(date)" >> $L/ah_done.txt || echo "real7 $2 FAILED $(date)" >> $L/ah_done.txt; }
run 0 qwen3vl8b &
run 1 llava_ov7b &
run 2 gemma3_12b &
wait; echo "ALL DONE $(date)" >> $L/ah_done.txt
