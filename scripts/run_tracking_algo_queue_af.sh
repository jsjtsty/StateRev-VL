#!/bin/bash
# Addendum AF: frozen-last-frame clips through the video path (GPU0), for models whose AE run is done.
cd "$(dirname "$0")/.."
L=outputs/tracking_algo_v1/logs
for m in qwen3vl8b qwen35_9b internvl35_8b llava_ov7b gemma3_12b; do
  while ! grep -q "real4 $m DONE" $L/ae_done.txt 2>/dev/null; do sleep 60; done
  python scripts/tracking_algo_real5.py --model $m --device-map cuda:0 > $L/real5_$m.log 2>&1 && echo "real5 $m DONE $(date)" >> $L/af_done.txt || echo "real5 $m FAILED $(date)" >> $L/af_done.txt
done
echo "ALL DONE $(date)" >> $L/af_done.txt
