#!/usr/bin/env bash
# Addendum AE on GPU1 (sequential); 32B after the other queues release GPUs 0/1.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PY=${PY:-/home/wangyf/miniconda3/envs/nul_staterev_vl/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in qwen3vl8b qwen35_9b internvl35_8b llava_ov7b gemma3_12b; do
  CUDA_VISIBLE_DEVICES=1 $PY scripts/tracking_algo_real4.py --model $m > $L/real4_$m.log 2>&1; echo "real4 $m DONE $(date)" >> $L/ae_done.txt
done
until grep -q "ALL DONE" $L/ac_done.txt 2>/dev/null && ! pgrep -f "tracking_algo_(real3|synthword).py --model qwen3vl32b" >/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=0,1 $PY scripts/tracking_algo_real4.py --model qwen3vl32b --device-map auto > $L/real4_qwen3vl32b.log 2>&1
echo "real4 qwen3vl32b DONE $(date)" >> $L/ae_done.txt; echo "ALL DONE $(date)" >> $L/ae_done.txt
