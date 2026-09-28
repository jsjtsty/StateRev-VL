#!/bin/bash
# Addendum AG (revision 1): real vs rendered, held vs frozen. GPU0/GPU1 two 8B models each, GPU2 Gemma then 32B on GPUs 2,3.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs; PY=/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_real6.py --model $2 --device-map $3 > $L/real6_$2.log 2>&1 && echo "real6 $2 DONE $(date)" >> $L/ag_done.txt || echo "real6 $2 FAILED $(date)" >> $L/ag_done.txt; }
(run 0 qwen3vl8b cuda:0; run 0 qwen35_9b cuda:0) &
(run 1 internvl35_8b cuda:0; run 1 llava_ov7b cuda:0) &
(run 2 gemma3_12b cuda:0; run 2,3 qwen3vl32b auto) &
wait; echo "ALL DONE $(date)" >> $L/ag_done.txt
