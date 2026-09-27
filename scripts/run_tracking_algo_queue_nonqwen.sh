#!/usr/bin/env bash
# Non-Qwen LLM backbones (Gemma-3-12B, Idefics3-8B/Llama3, InternVL3.5-GPT-OSS-20B on two GPUs): frames as a timestamped image list (family 'multiimg').
# Per model: native + PCD on all pure-present sets and chess captures; image-only controls (O,M, chess); Addendum U.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # $1 = GPU(s), $2 = model, $3 = device map
  D="--device-map ${3:-cuda:0}"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_anchor.py --model $2 --sets O,M --image-only $D > $L/anchor_imgonly_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_chess_now.py --model $2 --captures --image-only $D > $L/chesscap_$2_img.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 $D > $L/fix_pcd_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 --only U $D > $L/fixU_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_fix.py --model $2 --only U --image-only $D > $L/fixU_img_$2.log 2>&1
  echo "$2 ALL DONE $(date)" >> $L/nonqwen_done.txt
}
run ${GPU_A:-0} gemma3_12b &
run ${GPU_B:-1} idefics3_8b &
run ${GPU_C:-2,3} internvl_gptoss auto &
wait
