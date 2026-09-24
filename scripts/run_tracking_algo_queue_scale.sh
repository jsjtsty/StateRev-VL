#!/usr/bin/env bash
# scale sweep: Qwen3-VL 2B / 4B (GPU0, GPU1 after the current 7B jobs), 32B (GPUs 2,3 after 27B queue)
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
waitdl() { until grep -q '^DONE' $L/dl_$1.log 2>/dev/null; do sleep 60; done; }
waitlog() { until grep -q "$2" $1 2>/dev/null; do sleep 30; done; }
run() {  # gpu model devmap
  CUDA_VISIBLE_DEVICES=$1 python scripts/tracking_algo_now2.py --model $2 --mode video --device-map $3 > $L/now2_$2_video.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 python scripts/tracking_algo_now4.py --model $2 --device-map $3 > $L/now4_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 python scripts/tracking_algo_anchor.py --model $2 --sets O,M --image-only --device-map $3 > $L/anchor_imgonly_$2.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 python3 scripts/tracking_algo_eval.py --data outputs/tracking_algo_v1/data_reid2 --sets full,steps,tele,telefinal,blank --variants init --max-new-tokens 0 --out outputs/tracking_algo_v1/eval_reid2 --model $2 --device-map $3 > $L/reid2_ev_$2.log 2>&1
}
( waitdl Qwen3-VL-2B-Instruct; waitlog $L/now4_qwen25vl7b.log "obj n=3 D_fin=2.0"; run 0 qwen3vl2b cuda:0 ) &
( waitdl Qwen3-VL-4B-Instruct; waitlog $L/now4_llava_video7b.log "obj n=3 D_fin=2.0"; run 1 qwen3vl4b cuda:0 ) &
( waitdl Qwen3-VL-32B-Instruct; waitlog $L/anchor_qwen36_27b.log "J Ifull"; run 2,3 qwen3vl32b auto ) &
wait
