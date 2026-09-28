#!/bin/bash
# Qwen3-VL-32B for AB, AC, AE, AF on GPUs 1,2 (the earlier abc/ae 32B launches OOMed next to the AF run on GPU0).
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs; PY=/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python
export CUDA_VISIBLE_DEVICES=1,2
for s in real3 synthword real4 real5; do
  $PY scripts/tracking_algo_$s.py --model qwen3vl32b --device-map auto > $L/${s}_qwen3vl32b.log 2>&1 && echo "$s qwen3vl32b DONE $(date)" >> $L/q32b_done.txt || echo "$s qwen3vl32b FAILED $(date)" >> $L/q32b_done.txt
done
echo "ALL DONE $(date)" >> $L/q32b_done.txt
