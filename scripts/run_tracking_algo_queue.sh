#!/usr/bin/env bash
# G1 queue (v2): after phase A, run minimal pairs for the small models and the
# large hybrid models, then reasoning-mode generation.
set -u
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
PAIRS="--data outputs/tracking_algo_v1/data_pairs --out outputs/tracking_algo_v1/eval_pairs --sets mp_base,mp_gt_only,mp_sc_only,mp_neither --variants init --max-new-tokens 0"
wait_free() { while pgrep -f "tracking_algo_eval.py --model $1 " >/dev/null; do sleep 30; done; }

( wait_free qwen3vl8b; wait_free qwen35_9b
  CUDA_VISIBLE_DEVICES=0 python3 scripts/tracking_algo_eval.py --model qwen3vl8b --device-map cuda:0 $PAIRS > $L/pairs_qwen3vl8b.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python3 scripts/tracking_algo_eval.py --model qwen35_9b --device-map cuda:0 $PAIRS > $L/pairs_qwen35_9b.log 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=0,1 python3 scripts/tracking_algo_eval.py --model qwen36_27b --device-map auto > $L/qwen36_27b.log 2>&1
  CUDA_VISIBLE_DEVICES=0,1 python3 scripts/tracking_algo_eval.py --model qwen36_27b --device-map auto $PAIRS > $L/pairs_qwen36_27b.log 2>&1 ) &
( wait_free qwen25vl7b; wait_free llava_video7b
  CUDA_VISIBLE_DEVICES=2 python3 scripts/tracking_algo_eval.py --model qwen25vl7b --device-map cuda:0 $PAIRS > $L/pairs_qwen25vl7b.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 python3 scripts/tracking_algo_eval.py --model llava_video7b --device-map cuda:0 $PAIRS > $L/pairs_llava_video7b.log 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=2,3 python3 scripts/tracking_algo_eval.py --model qwen36_35b_a3b --device-map auto > $L/qwen36_35b_a3b.log 2>&1
  CUDA_VISIBLE_DEVICES=2,3 python3 scripts/tracking_algo_eval.py --model qwen36_35b_a3b --device-map auto $PAIRS > $L/pairs_qwen36_35b_a3b.log 2>&1 ) &
wait
echo PHASE_B_DONE > $L/phaseB.done

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/tracking_algo_eval.py --model qwen3vl8b_think --think --device-map cuda:0 \
    --sets opaque3,transp3 --variants init --shard $i/4 > $L/qwen3vl8b_think_$i.log 2>&1 &
done
wait
echo QUEUE_DONE > $L/queue.done
