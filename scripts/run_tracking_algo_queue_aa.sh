#!/usr/bin/env bash
# Addendum AA: real footage with a controlled final dwell.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
r() { for m in "${@:3}"; do CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_real2.py --model $m --device-map $2 > $L/real2_$m.log 2>&1; echo "real2 $m DONE $(date)" >> $L/aa_done.txt; done; }
r 0 cuda:0 qwen3vl8b llava_ov7b &
r 1 cuda:0 qwen35_9b internvl35_8b &
( r 2,3 auto qwen3vl32b; r 2 cuda:0 gemma3_12b ) &
wait
echo "ALL DONE $(date)" >> $L/aa_done.txt
