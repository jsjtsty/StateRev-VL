#!/usr/bin/env bash
# Addendum Z (real footage pilot). Waits for the X/Y queue to finish, then runs one model per GPU.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "ALL DONE" $L/xy_done.txt 2>/dev/null; do sleep 60; done
z() { for m in "${@:2}"; do CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_real.py --model $m > $L/real_$m.log 2>&1; echo "real $m DONE $(date)" >> $L/z_done.txt; done; }
z 0 qwen3vl8b llava_ov7b &
z 1 qwen35_9b &
z 2 internvl35_8b &
z 3 gemma3_12b &
wait
CUDA_VISIBLE_DEVICES=0,1 $PY scripts/tracking_algo_real.py --model qwen3vl32b --device-map auto > $L/real_qwen3vl32b.log 2>&1
echo "ALL DONE $(date)" >> $L/z_done.txt
