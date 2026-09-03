#!/usr/bin/env python3
"""Mechanism Gate Experiment 3 - matched causal event patching (GPU).

For each pre-registered (source S -> target T) pair (same t/grid), test whether
the window-video-token activations causally carry the event that drives the
current-state readout.

Conditions (all patch into the TARGET T's forward; metric = P(counterfactual
state St_cf = transition(prev_T, E_S))):
  main       : S window video acts -> T WINDOW positions   (expect shift)
  self       : T window video acts -> T WINDOW positions   (no-op control)
  position   : S window video acts -> T HISTORY positions  (locality control)
  sameevent  : S3 window video acts -> T WINDOW positions  (content control,
               S3 has the SAME event as T -> expect no shift toward St_cf)

Layer sweep: main over all 36 decoder layers; controls over a representative
subset. Patch = replace the decoder layer's output at the target positions with
the source's same-layer, same-position window activations (forward hook).

Outputs: causal_exp3.csv (per pair/condition/layer: P_cf + full state probs +
rescue), causal_exp3_manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_rev_input_pipeline import (
    load_model_and_processor, render_inputs, to_device, POS_IDS,
)
import mg_common as mg
from run_state_rev_audit import state_messages
from run_vetbench_screening import sample_clip

N_DECODER_LAYERS = 36
LAYER_CTRL = [0, 5, 11, 17, 22, 28, 33, 35]
CONDS = ["main", "self", "position", "sameevent", "window_permute"]


def apply_event(prev, event):
    a, b = event.split(" and ")
    return b if prev == a else (a if prev == b else prev)


def clean_capture(model, inp, window_pos, dev):
    """One forward capturing the window-position hidden states at every
    decoder layer's output + the first-token state logprobs."""
    pos_t = torch.as_tensor(window_pos, dtype=torch.long, device=dev)
    captured = {}
    handles = []
    for d in range(N_DECODER_LAYERS):
        layer = model.model.language_model.layers[d]

        def hook(mod, i, out, _d=d, _pos=pos_t):
            h = out if isinstance(out, torch.Tensor) else out[0]
            captured[_d] = h[0, _pos, :].float().cpu().clone()

        handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(**inp, logits_to_keep=1)
        logits = out.logits[0, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    logp = torch.log_softmax(logits, dim=-1)
    state = {n: float(logp[i].item()) for n, i in POS_IDS.items()}
    return state, captured


def patched_state(model, inp, d, patch_pos, patch_act, dev):
    """One forward with layer d's output at patch_pos replaced by patch_act;
    returns first-token state logprobs."""
    layer = model.model.language_model.layers[d]
    pos_t = torch.as_tensor(patch_pos, dtype=torch.long, device=dev)
    act = patch_act.to(dev)

    def hook(mod, i, out):
        h = out if isinstance(out, torch.Tensor) else out[0]
        h = h.clone()
        h[0, pos_t, :] = act.to(h.dtype)
        return h if isinstance(out, torch.Tensor) else (h,) + tuple(out[1:])

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(**inp, logits_to_keep=1)
        logits = out.logits[0, -1, :].float()
    finally:
        handle.remove()
    logp = torch.log_softmax(logits, dim=-1)
    return {n: float(logp[i].item()) for n, i in POS_IDS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--dataset", default="dataset/vetbench/cup")
    ap.add_argument("--out-dir", default="outputs/vetbench/mechanism_gate_v1")
    ap.add_argument("--pairs", default="outputs/vetbench/mechanism_gate_v1/"
                                       "exp3_pairs.json")
    ap.add_argument("--regime", default="controlled_8fps")
    ap.add_argument("--main-layers", default="all",
                    help="'all' or comma list of decoder layer idx")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(args.pairs))["pairs"]
    if args.limit:
        pairs = pairs[:args.limit]
    layer_main = list(range(N_DECODER_LAYERS)) if args.main_layers == "all" \
        else [int(x) for x in args.main_layers.split(",")]

    model, processor = load_model_and_processor(Path(args.model_dir))
    dev = next(model.parameters()).device

    csv_path = out / "causal_exp3.csv"
    new_file = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["pair_id", "t", "condition", "layer", "cf_state",
                    "st_state", "P_cf", "P_Left", "P_Middle", "P_Right",
                    "clean_P_cf", "clean_P_st", "rescue_cf"])
        fh.flush()

    t0 = time.time()
    for pi, pr in enumerate(pairs):
        T, S, S3 = pr["target"], pr["source"], pr["source_sameevent"]
        t = pr["t"]
        cf = pr["counterfactual_state"]
        pair_id = f"{T['traj']}_t{t}"

        def render_winpos(traj, frame_end, initial_state):
            clip = sample_clip(Path(args.dataset) / f"{traj}.mp4", 0,
                               int(frame_end))
            msgs = state_messages(clip, initial_state, t)
            inputs, fp = render_inputs(processor, msgs, clip, args.regime)
            ids = inputs["input_ids"][0].tolist()
            mm = inputs.get("mm_token_type_ids")
            grid = fp["video_grid_thw"]
            num_frames = fp["n_video_frames"]
            in_win = mg.window_sample_mask(int(frame_end), t, num_frames)
            reg = mg.identify_token_regions(ids, mm, "", processor)
            vp = reg["_video_span"]
            win_tok, hist_tok = mg.video_window_token_regions(vp, grid, in_win)
            return inputs, win_tok.tolist(), hist_tok.tolist()

        # same t/grid -> same window-token COUNT; verify absolute positions match
        inputs_T, win_pos, hist_tok = render_winpos(
            T["traj"], T["frame_end"], T["initial_state"])
        inputs_S, win_pos_S, _ = render_winpos(
            S["traj"], S["frame_end"], S["initial_state"])
        inputs_S3, win_pos_S3, _ = render_winpos(
            S3["traj"], S3["frame_end"], S3["initial_state"])
        assert win_pos == win_pos_S == win_pos_S3, \
            f"window pos mismatch T{len(win_pos)} S{len(win_pos_S)} " \
            f"S3{len(win_pos_S3)}"
        n_ctrl = min(len(win_pos), len(hist_tok))
        hist_pos_ctrl = hist_tok[-n_ctrl:]

        inp_T = to_device(inputs_T, dev)
        inp_S = to_device(inputs_S, dev)
        inp_S3 = to_device(inputs_S3, dev)

        # clean captures (window layer outputs)
        P_T, T_win = clean_capture(model, inp_T, win_pos, dev)
        _, S_win = clean_capture(model, inp_S, win_pos, dev)
        _, S3_win = clean_capture(model, inp_S3, win_pos, dev)
        # clean full state distribution (probs)
        clean_p = {n: float(torch.exp(torch.tensor(v)).item())
                   for n, v in P_T.items()}
        clean_P_cf = clean_p[cf]
        clean_P_st = clean_p[T["gt_state"]]
        # fixed within-window position permutation (structure control)
        wperm = np.random.default_rng(20260902).permutation(len(win_pos))

        def record(cond, d, state_lp):
            p = {n: float(torch.exp(torch.tensor(v)).item())
                 for n, v in state_lp.items()}
            w.writerow([pair_id, t, cond, d, cf, T["gt_state"],
                        f"{p[cf]:.5f}", f"{p['Left']:.5f}",
                        f"{p['Middle']:.5f}", f"{p['Right']:.5f}",
                        f"{clean_P_cf:.5f}", f"{clean_P_st:.5f}",
                        f"{p[cf]-clean_P_cf:.5f}"])
            fh.flush()

        # main: all layers, S window -> T window
        for d in layer_main:
            record("main", d, patched_state(model, inp_T, d, win_pos,
                                            S_win[d], dev))
        # controls: representative layers
        for d in LAYER_CTRL:
            record("self", d, patched_state(model, inp_T, d, win_pos,
                                            T_win[d], dev))
            record("position", d, patched_state(model, inp_T, d,
                                                hist_pos_ctrl,
                                                S_win[d][:n_ctrl], dev))
            record("sameevent", d, patched_state(model, inp_T, d, win_pos,
                                                 S3_win[d], dev))
            record("window_permute", d, patched_state(model, inp_T, d,
                                                      win_pos,
                                                      S_win[d][wperm], dev))
        del inputs_T, inputs_S, inputs_S3, inp_T, inp_S, inp_S3
        torch.cuda.empty_cache()
        print(f"[{pi+1}/{len(pairs)}] {pair_id} cf={cf} "
              f"({time.time()-t0:.0f}s)", flush=True)
    fh.close()
    print(f"Exp3 done: {len(pairs)} pairs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
