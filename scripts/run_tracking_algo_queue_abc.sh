#!/usr/bin/env bash
# Remaining Addendum AB (real frames on a synthetic timeline) and AC (synthetic word tiles) runs, spread over idle GPUs.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
both() {  # $1 = GPU(s), $2 = device map, $3 = model
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_real3.py --model $3 --device-map $2 > $L/real3_$3.log 2>&1; echo "real3 $3 DONE $(date)" >> $L/ab_done.txt
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/tracking_algo_synthword.py --model $3 --device-map $2 > $L/synthword_$3.log 2>&1; echo "synthword $3 DONE $(date)" >> $L/ac_done.txt
}
( both 0 cuda:0 internvl35_8b ) &
( both 1 cuda:0 llava_ov7b ) &
( until grep -q "ALL DONE" $L/aa_done.txt; do sleep 30; done; both 2 cuda:0 gemma3_12b ) &
wait
until ! pgrep -f "tracking_algo_(real2|real3|synthword).py" >/dev/null; do sleep 30; done
both 0,1 auto qwen3vl32b
echo "ALL DONE $(date)" >> $L/ab_done.txt; echo "ALL DONE $(date)" >> $L/ac_done.txt
