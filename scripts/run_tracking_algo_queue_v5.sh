#!/usr/bin/env bash
# queue v5: priorities = horizon token probe (Qwen3-VL) > k1 hidden > reasoning-mode shard 0.
set -u
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
E="python3 scripts/tracking_algo_eval.py"
PAIRS="--data outputs/tracking_algo_v1/data_pairs --out outputs/tracking_algo_v1/eval_pairs --sets mp_base,mp_gt_only,mp_sc_only,mp_neither --variants init --max-new-tokens 0"
waitpids() { for p in $1; do while kill -0 $p 2>/dev/null; do sleep 20; done; done; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( waitpids "$1"   # 27B lane (main + pairs) on GPUs 0,1
  CUDA_VISIBLE_DEVICES=0 python3 scripts/tracking_algo_tokenprobe.py --model qwen3vl8b --data outputs/tracking_algo_v1/data_horizon --sets all --tag _horizon > $L/horizon_tp_qwen3vl8b.log 2>&1 ) &
( waitpids "$1"
  CUDA_VISIBLE_DEVICES=1 $E --model qwen3vl8b --device-map cuda:0 --sets k1 --variants init --hidden --hidden-sets k1 --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_k1h > $L/k1h_qwen3vl8b.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $E --model qwen3vl8b_think --think --device-map cuda:0 --sets opaque3 --variants init --shard 0/2 > $L/think_qwen3vl8b_0.log 2>&1 ) &
wait
echo done > $L/queue_v5.done
