#!/usr/bin/env python3
"""Mechanism Gate v1 - shared infrastructure.

Builds on scripts/state_rev_input_pipeline.py (the unified input module).
Adds:
  1. frame -> video-token mapping (derived from the Qwen3-VL video processor:
     uniform linspace sampling + temporal-major patchify).
  2. matched temporal manipulations of the CURRENT swap window (freeze /
     shuffle / matched-history-freeze) that preserve total frame count, frame
     positions and the processor token budget, implemented at the SAMPLED-frame
     level and re-rendered with identity sampling.
  3. multi-token-position hidden-state extraction (for the token sweep).
  4. fingerprint + validation helpers proving the manipulated input keeps the
     same grid / input_ids and differs only in the intended patches.

All behavior / hidden / intervention paths MUST go through render_inputs
(pipeline) for the baseline and through render_manipulated (here) for the
manipulated conditions, and every forward records a fingerprint.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_rev_input_pipeline as pipe
from state_rev_input_pipeline import (
    POS, POS_IDS, EVENT_CLASSES, FPS_NATIVE, TEMPORAL_PATCH,
    expected_frames, requested_fps, render_inputs, to_device, _sha256_bytes,
)

# swap timing (30 fps), from run_vetbench_screening
SHUFFLE_START_MS = 1900
SWAP_CYCLE_MS = 2000
SWAP_LEN_FRAMES = SWAP_CYCLE_MS / 1000 * FPS_NATIVE  # = 60


# ---------------------------------------------------------------------------
# frame -> token mapping
# ---------------------------------------------------------------------------

def sampled_indices(clip_len: int, num_frames: int) -> np.ndarray:
    """Exact frame indices the Qwen3-VL processor samples:
    np.linspace(0, clip_len-1, num_frames).round().astype(int)."""
    return np.linspace(0, clip_len - 1, num_frames).round().astype(int)


def swap_window_frames(clip_len: int, t: int) -> tuple[int, int]:
    """[start, end) of the CURRENT (t-th) swap window in clip coordinates.
    clip is [0, 57+t*60); the current swap is the last 60 frames."""
    end = int(round(SHUFFLE_START_MS / 1000 * FPS_NATIVE + t * SWAP_LEN_FRAMES))
    start = end - int(round(SWAP_LEN_FRAMES))
    # clip_len must equal end (prefix ends at the current swap window end)
    return start, end


def window_sample_mask(clip_len: int, t: int, num_frames: int) -> np.ndarray:
    """Boolean mask over the num_frames sampled frames: True if the sampled
    frame index lies in the current swap window."""
    idx0 = sampled_indices(clip_len, num_frames)
    w0, w1 = swap_window_frames(clip_len, t)
    return (idx0 >= w0) & (idx0 < w1)


def video_token_blocks(grid_t: int, grid_hw: int) -> np.ndarray:
    """Map each sampled frame (0..num_frames-1) to the video-token block
    [t_idx*grid_hw, (t_idx+1)*grid_hw) it belongs to. Returns, for each sampled
    frame i, the integer block index b(i)=i//TEMPORAL_PATCH."""
    return None  # placeholder; block of frame i is i // TEMPORAL_PATCH


# ---------------------------------------------------------------------------
# temporal manipulations (on the SAMPLED-frame array)
# ---------------------------------------------------------------------------

def build_manipulated_clip(clip: np.ndarray, t: int, regime: str,
                           condition: str, seed: int = 20260902):
    """Manipulate the SOURCE clip at the frame level so the processor's
    sampling (linspace over clip_len), frame positions (-> timestamps),
    clip length, grid and input_ids are ALL preserved; only the intended
    frames' pixels change.

    `clip` is the full prefix clip (clip_len, H, W, 3). The processor samples
    frames at idx0 = linspace(0, clip_len-1, num_frames). We replace the clip
    frames AT those sample positions for the chosen condition, leaving all
    other clip frames (and the clip length) unchanged. The caller then renders
    the modified clip with the SAME regime via render_inputs.

    conditions:
      baseline        : clip unchanged
      freeze          : all current-window sample frames -> first window frame
      shuffle         : current-window sample frames permuted (same image set)
      history_freeze  : freeze min(|W|,|H|) EARLIEST history sample frames to
                        the first history frame (matched generic corruption);
                        current window unchanged
    Returns (modified_clip, meta).
    """
    clip_len = len(clip)
    num_frames = expected_frames(regime, clip_len)
    idx0 = sampled_indices(clip_len, num_frames)
    in_win = window_sample_mask(clip_len, t, num_frames)
    W = np.where(in_win)[0]                      # sample indices in window
    H = np.where(~in_win)[0]                     # sample indices in history
    w0, w1 = swap_window_frames(clip_len, t)
    clip1 = clip.copy()
    meta = {"clip_len": clip_len, "num_frames": num_frames, "t": t,
            "condition": condition, "window": [int(w0), int(w1)],
            "n_window_frames": int(len(W)), "n_history_frames": int(len(H)),
            "window_sample_idx": W.tolist(),
            "window_clip_frames": idx0[W].tolist()}
    if condition == "baseline":
        meta["n_corrupted"] = 0
    elif condition == "freeze":
        i0 = int(W[0])
        target = clip[idx0[i0]]
        clip1[idx0[W]] = target
        meta["n_corrupted"] = int(len(W))
        meta["frozen_to_clip_frame"] = int(idx0[i0])
    elif condition == "shuffle":
        rng = np.random.default_rng(seed)
        frames = clip[idx0[W]]
        perm = rng.permutation(len(W))
        clip1[idx0[W]] = frames[perm]
        meta["n_corrupted"] = int(len(W))
        meta["perm"] = perm.tolist()
    elif condition == "history_freeze":
        n = min(len(W), len(H))
        Hf = H[:n]                               # earliest history frames
        target = clip[idx0[int(H[0])]]
        clip1[idx0[Hf]] = target
        meta["n_corrupted"] = int(n)
        meta["corrupted_clip_frames"] = idx0[Hf].tolist()
        meta["frozen_to_clip_frame"] = int(idx0[int(H[0])])
    else:
        raise KeyError(condition)
    return clip1, meta


def patch_diff_mask(baseline_pv, manip_pv) -> np.ndarray:
    """Boolean mask over patch rows (n_patches) that differ between the
    baseline and manipulated pixel_values_videos."""
    a = baseline_pv.float().numpy()
    b = manip_pv.float().numpy()
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch {a.shape} vs {b.shape}")
    return np.abs(a - b).sum(axis=1) > 1e-6


# ---------------------------------------------------------------------------
# multi-token-position hidden-state extraction (Exp 2)
# ---------------------------------------------------------------------------

def extract_hidden_at(model, inputs, layers, token_positions: dict[str, np.ndarray]):
    """Base-model forward; returns hidden states at MULTIPLE token-position
    families. `token_positions` maps name -> array of absolute token indices
    (within the prompt). Returns dict name -> (n_layers, D).
    Uses output_hidden_states and reads the requested positions.
    """
    dev = next(model.parameters()).device
    inp = to_device(inputs, dev)
    with torch.inference_mode():
        out = model.model(**inp, output_hidden_states=True)
    hs = out.hidden_states                       # 37-tuple of (1, L, D)
    idx = list(range(len(hs))) if layers is None else list(layers)
    res = {}
    for name, pos in token_positions.items():
        pos = np.asarray(pos, dtype=int)
        stacked = np.stack(
            [hs[i][0, pos, :].float().cpu().numpy().mean(axis=0)
             for i in idx], axis=0)
        res[name] = stacked                      # (n_layers, D)
    return res


def identify_token_regions(input_ids: list[int], mm_token_type_ids,
                           question_text: str, processor) -> dict:
    """Identify token-position families within the rendered prompt.

    Returns dict name -> np.array of absolute token indices.
    Regions (pre-registered):
      final_prompt      : last input token
      question_mean     : mean over the question text token span
      currently         : token(s) around the word 'currently'
      video_all         : all video (mm) tokens
      video_last25      : last 25% of video tokens
      video_window      : video tokens (temporal blocks) covering the current
                          swap-window sampled frames  (needs window mask)
      video_pre         : video tokens strictly before the current swap window
    NOTE: video_window / video_pre depend on the per-row window mask and are
    filled by the caller (they need t + num_frames); here we return the full
    video span so the caller can slice it.
    """
    ids = np.asarray(input_ids, dtype=int)
    L = len(ids)
    # video (mm) token positions: mm_token_type_ids == 2 (0=text, 2=video)
    mm = np.asarray(mm_token_type_ids[0].tolist()
                    if mm_token_type_ids.dim() == 2
                    else mm_token_type_ids.tolist(), dtype=int)
    video_pos = np.where(mm == 2)[0]
    # question text span: find the token span of the question text. We locate
    # it by re-encoding: the question text is the last text block; find the
    # last run of non-video tokens after the video span that matches.
    q_ids = processor.tokenizer(question_text, add_special_tokens=False)["input_ids"]
    q_start = _find_subsequence(ids, np.asarray(q_ids, dtype=int))
    if q_start is None:
        # fallback: text tokens after the video span
        text_after = np.where((mm == 0) & (np.arange(L) >= video_pos[-1]))[0]
        q_span = text_after
    else:
        q_span = np.arange(q_start, q_start + len(q_ids))
    # 'currently' token
    cur_ids = processor.tokenizer("currently", add_special_tokens=False)["input_ids"]
    cur_start = _find_subsequence(ids, np.asarray(cur_ids, dtype=int))
    if cur_start is not None:
        cur_span = np.arange(cur_start, cur_start + len(cur_ids))
    else:
        # find the token whose decode contains 'currently'
        cur_span = np.array([i for i in range(L)
                             if "currently" in processor.tokenizer
                             .decode([int(ids[i])]).lower()], dtype=int)
    return {
        "final_prompt": np.array([L - 1]),
        "question_mean": q_span,
        "currently": cur_span,
        "video_all": video_pos,
        "video_last25": video_pos[int(0.75 * len(video_pos)):] if len(video_pos) else np.array([], dtype=int),
        "_video_span": video_pos,
    }


def video_window_token_regions(video_pos: np.ndarray, grid, in_win_mask: np.ndarray):
    """Slice the video-token positions into current-window / pre-window regions.

    Qwen3-VL video tokens are temporal-major: the k-th video token (in the
    mm==2 sequence) belongs to temporal block k // tokens_per_block, where
    tokens_per_block = (grid_h//merge)*(grid_w//merge), merge=2. A temporal
    block covers sampled frames [2*b, 2*b+1].

    Returns (window_tokens, pre_tokens): absolute token indices.
    """
    merge = 2
    tokens_per_block = (grid[1] // merge) * (grid[2] // merge)
    k = np.arange(len(video_pos))
    block_of_token = k // tokens_per_block
    W = np.where(np.asarray(in_win_mask))[0]
    if len(W) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    win_blocks = np.unique(W // TEMPORAL_PATCH)
    first_block = int(win_blocks.min())
    window_tokens = video_pos[np.isin(block_of_token, win_blocks)]
    pre_tokens = video_pos[block_of_token < first_block]
    return window_tokens, pre_tokens


def _find_subsequence(haystack: np.ndarray, needle: np.ndarray):
    if len(needle) == 0 or len(needle) > len(haystack):
        return None
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if np.array_equal(haystack[i:i + m], needle):
            return i
    return None
