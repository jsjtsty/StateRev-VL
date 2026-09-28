#!/usr/bin/env bash
# Addendum X (per-item logit dumps for PCD gates) and Addendum Y (layer-resolved masking, Gemma-3 / Idefics3).
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
dump() {  # $1 = GPU(s), $2 = model, $3 = device map
  D="--device-map ${3:-cuda:0}"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 --dump $D > $L/dump_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 --dump --only U $D > $L/dumpU_$2.log 2>&1
  echo "dump $2 DONE $(date)" >> $L/xy_done.txt
}
mask() {
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_layermask.py --model $2 > $L/layermask_$2.log 2>&1
  echo "layermask $2 DONE $(date)" >> $L/xy_done.txt
}
( mask 0 gemma3_12b ) &
( mask 1 idefics3_8b; dump 1 idefics3_8b ) &
( dump 2 qwen3vl8b; dump 2 qwen35_9b; dump 2 internvl35_8b ) &
( dump 3 gemma3_12b; dump 3 llava_ov7b ) &
wait
dump 0,1 internvl_gptoss auto
echo "ALL DONE $(date)" >> $L/xy_done.txt
