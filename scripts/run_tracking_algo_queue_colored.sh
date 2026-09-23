#!/usr/bin/env bash
# colored3 runs: small models after phase B (sharing GPUs with reasoning runs), large models after the main queue.
set -u
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
until [ -f $L/phaseB.done ]; do sleep 30; done
i=0
for m in qwen3vl8b qwen35_9b qwen25vl7b llava_video7b; do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/tracking_algo_eval.py --model $m --device-map cuda:0 --sets colored3 --max-new-tokens 0 > $L/colored_$m.log 2>&1 &
  i=$((i+1))
done
wait
until [ -f $L/queue.done ]; do sleep 30; done
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/tracking_algo_eval.py --model qwen36_27b --device-map auto --sets colored3 --max-new-tokens 0 > $L/colored_qwen36_27b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 python3 scripts/tracking_algo_eval.py --model qwen36_35b_a3b --device-map auto --sets colored3 --max-new-tokens 0 > $L/colored_qwen36_35b_a3b.log 2>&1 &
wait
echo done > $L/colored.done
