#!/usr/bin/env bash
# night 3 queue: N on GPU0/1 after M; 27B J/K/L/M/N on GPUs 2,3
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
waitgrep() { until grep -q "m=4 D_fin=2.0" $1 2>/dev/null; do sleep 30; done; }
( waitgrep $L/now2_qwen3vl8b_images.log; CUDA_VISIBLE_DEVICES=0 python scripts/tracking_algo_now3.py --model qwen3vl8b > $L/now3_qwen3vl8b.log 2>&1 ) &
( waitgrep $L/now2_qwen35_9b_images.log; CUDA_VISIBLE_DEVICES=1 python scripts/tracking_algo_now3.py --model qwen35_9b > $L/now3_qwen35_9b.log 2>&1 ) &
( until grep -q DONE $L/reid_tp_qwen36_27b.log 2>/dev/null; do sleep 30; done
  D="CUDA_VISIBLE_DEVICES=2,3"
  CUDA_VISIBLE_DEVICES=2,3 python scripts/tracking_algo_now3.py --model qwen36_27b --device-map auto > $L/now3_qwen36_27b.log 2>&1
  CUDA_VISIBLE_DEVICES=2,3 python scripts/tracking_algo_now.py --model qwen36_27b --device-map auto > $L/now_qwen36_27b.log 2>&1
  CUDA_VISIBLE_DEVICES=2,3 python scripts/tracking_algo_now2.py --model qwen36_27b --mode video --device-map auto > $L/now2_qwen36_27b_video.log 2>&1
  CUDA_VISIBLE_DEVICES=2,3 python3 scripts/tracking_algo_eval.py --data outputs/tracking_algo_v1/data_reid2 --sets full,steps,tele,telefinal,blank --variants init --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_reid2 --model qwen36_27b --device-map auto > $L/reid2_ev_qwen36_27b.log 2>&1
) &
wait
