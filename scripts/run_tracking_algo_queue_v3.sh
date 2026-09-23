#!/usr/bin/env bash
# Queue v3 (35B-A3B dropped for now: with 2x40GB it offloads experts to CPU).
set -u
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
E="python3 scripts/tracking_algo_eval.py"
PAIRS="--data outputs/tracking_algo_v1/data_pairs --out outputs/tracking_algo_v1/eval_pairs --sets mp_base,mp_gt_only,mp_sc_only,mp_neither --variants init --max-new-tokens 0"
( CUDA_VISIBLE_DEVICES=2 $E --model qwen3vl8b --device-map cuda:0 --sets colored3 --max-new-tokens 0 > $L/colored_qwen3vl8b.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $E --model qwen25vl7b --device-map cuda:0 --sets colored3 --max-new-tokens 0 > $L/colored_qwen25vl7b.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $E --model qwen3vl8b --device-map cuda:0 --sets k1 --variants init --hidden --hidden-sets k1 --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_k1h > $L/k1h_qwen3vl8b.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 python3 scripts/tracking_algo_decompose.py --model qwen3vl8b --device-map cuda:0 > $L/decomp_qwen3vl8b.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 $E --model qwen35_9b --device-map cuda:0 --sets colored3 --max-new-tokens 0 > $L/colored_qwen35_9b.log 2>&1
  CUDA_VISIBLE_DEVICES=3 $E --model llava_video7b --device-map cuda:0 --sets colored3 --max-new-tokens 0 > $L/colored_llava_video7b.log 2>&1
  CUDA_VISIBLE_DEVICES=3 $E --model qwen35_9b --device-map cuda:0 --sets k1 --variants init --hidden --hidden-sets k1 --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_k1h > $L/k1h_qwen35_9b.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 python3 scripts/tracking_algo_decompose.py --model qwen35_9b --device-map cuda:0 > $L/decomp_qwen35_9b.log 2>&1 ) &
( while kill -0 1131781 2>/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=0,1 $E --model qwen36_27b --device-map auto $PAIRS > $L/pairs_qwen36_27b.log 2>&1
  CUDA_VISIBLE_DEVICES=0,1 $E --model qwen36_27b --device-map auto --sets colored3 --max-new-tokens 0 > $L/colored_qwen36_27b.log 2>&1 ) &
wait
echo SMALL_DONE > $L/v3_small.done
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $E --model qwen3vl8b_think --think --device-map cuda:0 --sets opaque3,transp3 --variants init --shard $i/4 > $L/qwen3vl8b_think_$i.log 2>&1 &
done
wait
echo QUEUE_DONE > $L/queue_v3.done
