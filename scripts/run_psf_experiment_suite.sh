#!/usr/bin/env bash
set -euo pipefail

# Cache-only PSF training suite. This script never invokes a VLM forward.
# Formal use requires SHA256(text)-keyed frozen text embeddings for each model.
DEVICE="${DEVICE:-cuda:0}"
MODELS="${MODELS:-qwen,llava}"
TASKS="${TASKS:-shell,chess}"
EPOCHS="${EPOCHS:-30}"
OUT_ROOT="${OUT_ROOT:-outputs/psf_v1/formal_joint}"
export PSF_CHESS_QWEN_ROOT="${PSF_CHESS_QWEN_ROOT:-outputs/psf_v1/formal_precheck/chess_qwen_multitap}"
QWEN_TEXT_CACHE="${QWEN_TEXT_CACHE:?set QWEN_TEXT_CACHE to frozen text embedding NPZ}"
LLAVA_TEXT_CACHE="${LLAVA_TEXT_CACHE:?set LLAVA_TEXT_CACHE to frozen text embedding NPZ}"

for ablation in full memory_only observation_only fixed_fusion fixed_gate no_change_loss no_persistence_loss single_layer; do
  python scripts/train_psf.py \
    --train-tasks "$TASKS" \
    --models "$MODELS" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --ablation "$ablation" \
    --text-cache-qwen "$QWEN_TEXT_CACHE" \
    --text-cache-llava "$LLAVA_TEXT_CACHE" \
    --strict-text-cache \
    --out "$OUT_ROOT/$ablation"
done
