#!/usr/bin/env bash
# Tail of the X queue, rescheduled so GPUs 0/1 do not idle: InternVL-GPT-OSS now on GPUs 0/1; Gemma's U dump after its
# running main dump (GPU3); LLaVA-OV on whichever GPU frees first (GPUs 0 after GPT-OSS, or GPU2 after InternVL3.5).
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GEMMA_PID=${GEMMA_PID:?}
dump() {  # $1 = GPU(s), $2 = model, $3 = device map
  D="--device-map ${3:-cuda:0}"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 --dump $D > $L/dump_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 --dump --only U $D > $L/dumpU_$2.log 2>&1
  echo "dump $2 DONE $(date)" >> $L/xy_done.txt
}
( dump 0,1 internvl_gptoss auto ) &
( while kill -0 $GEMMA_PID 2>/dev/null; do sleep 60; done
  CUDA_VISIBLE_DEVICES=3 $PY scripts/tracking_algo_fix.py --model gemma3_12b --dump --only U --device-map cuda:0 > $L/dumpU_gemma3_12b.log 2>&1
  echo "dump gemma3_12b DONE $(date)" >> $L/xy_done.txt ) &
( until grep -q -E "dump (internvl_gptoss|internvl35_8b) DONE" $L/xy_done.txt; do sleep 60; done
  G=$(grep -q "dump internvl35_8b DONE" $L/xy_done.txt && echo 2 || echo 0)
  dump $G llava_ov7b ) &
wait
echo "ALL DONE $(date)" >> $L/xy_done.txt
