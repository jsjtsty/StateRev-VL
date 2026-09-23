#!/usr/bin/env bash
# Night queue v4. Args: PIDs to wait for on GPU2 and GPU3 lanes, and the 27B lane PID.
set -u
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
E="python3 scripts/tracking_algo_eval.py"
G2WAIT="$1"; G3WAIT="$2"; G01WAIT="$3"
waitpids() { for p in $1; do while kill -0 $p 2>/dev/null; do sleep 20; done; done; }
( waitpids "$G2WAIT"
  CUDA_VISIBLE_DEVICES=2 python3 scripts/tracking_algo_tokenprobe.py --model qwen3vl8b > $L/tokenprobe_qwen3vl8b.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $E --model qwen3vl8b --device-map cuda:0 --sets k1 --variants init --hidden --hidden-sets k1 --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_k1h > $L/k1h_qwen3vl8b.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $E --model qwen3vl8b_think --think --device-map cuda:0 --sets opaque3 --variants init --shard 0/2 > $L/think_qwen3vl8b_0.log 2>&1 ) &
( waitpids "$G3WAIT"
  CUDA_VISIBLE_DEVICES=3 python3 scripts/tracking_algo_tokenprobe.py --model qwen35_9b > $L/tokenprobe_qwen35_9b.log 2>&1
  CUDA_VISIBLE_DEVICES=3 $E --model qwen35_9b --device-map cuda:0 --sets k1 --variants init --hidden --hidden-sets k1 --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_k1h > $L/k1h_qwen35_9b.log 2>&1
  CUDA_VISIBLE_DEVICES=3 $E --model qwen3vl8b_think --think --device-map cuda:0 --sets opaque3 --variants init --shard 1/2 > $L/think_qwen3vl8b_1.log 2>&1 ) &
( waitpids "$G01WAIT"
  CUDA_VISIBLE_DEVICES=0 $E --model qwen35_9b --think --device-map cuda:0 --sets opaque3 --variants init --shard 0/2 > $L/think_qwen35_9b_0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $E --model qwen35_9b --think --device-map cuda:0 --sets opaque3 --variants init --shard 1/2 > $L/think_qwen35_9b_1.log 2>&1 &
  wait ) &
wait
echo done > $L/queue_v4.done
