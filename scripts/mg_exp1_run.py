#!/usr/bin/env python3
"""Mechanism Gate Experiment 1 - Visual grounding of the event code (GPU).

For all 250 prefixes (controlled_8fps primary), render 4 matched temporal
conditions with the original state-question prompt, and for each save:
  - state behavior (first-token native logits + answer)
  - last-input-token hidden states (37 layers)
  - input fingerprint

Conditions: baseline / freeze / shuffle / history_freeze.
Outputs (under --out-dir):
  hidden_exp1_{cond}.npz   (key -> 37x4096)
  behavior_exp1.csv
  fingerprints_exp1.csv
  exp1_manifest.json       (per-row manipulation meta: window, n_corrupted, ...)
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
    load_model_and_processor, render_inputs, extract_hidden_states,
    first_token_logits, POS_IDS,
)
import mg_common as mg
from run_state_rev_audit import state_messages
from run_vetbench_screening import sample_clip

COND_ORDER = ["baseline", "freeze", "shuffle", "history_freeze"]
N_LAYERS, D = 37, 4096


def load_rows(path):
    rows = list(csv.DictReader(open(path, newline="")))
    assert len(rows) == 250, len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--dataset", default="dataset/vetbench/cup")
    ap.add_argument("--out-dir", default="outputs/vetbench/mechanism_gate_v1")
    ap.add_argument("--regime", default="controlled_8fps")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.behavior_csv)
    if args.limit:
        rows = rows[:args.limit]

    model, processor = load_model_and_processor(Path(args.model_dir))
    dev = next(model.parameters()).device

    # pre-allocate hidden buffers per condition
    hid = {c: np.zeros((len(rows), N_LAYERS, D), dtype=np.float32)
           for c in COND_ORDER}
    beh_rows = []
    fp_rows = []
    manifest = []
    done_rows = 0
    t0 = time.time()

    for i, r in enumerate(rows):
        traj = r["trajectory_id"]
        t = int(r["t"])
        clip_end = int(r["frame_end"])
        clip = sample_clip(Path(args.dataset) / f"{traj}.mp4", 0, clip_end)
        row_meta = {}
        for c in COND_ORDER:
            clip_c, meta = mg.build_manipulated_clip(clip, t, args.regime, c)
            row_meta[c] = {k: meta[k] for k in
                           ("n_corrupted", "n_window_frames", "window",
                            "frozen_to_clip_frame", "corrupted_clip_frames")
                           if k in meta}
            msgs = state_messages(clip_c, r["initial_state"], t)
            inputs, fp = render_inputs(processor, msgs, clip_c, args.regime)
            # hidden states at last input token
            hs = extract_hidden_states(model, inputs)   # (37, 4096)
            hid[c][i] = hs
            # behavior
            beh = first_token_logits(model, processor, inputs, POS_IDS)
            beh_rows.append({
                "trajectory_id": traj, "t": t, "condition": c,
                "state_pred": beh["answer"],
                "state_correct": str(beh["answer"] == r["gt_state"]).lower(),
                "logprob_Left": f"{beh['first_logprobs']['Left']:.6f}",
                "logprob_Middle": f"{beh['first_logprobs']['Middle']:.6f}",
                "logprob_Right": f"{beh['first_logprobs']['Right']:.6f}",
                "n_answer_tokens": beh["n_answer_tokens"],
            })
            fp = {**fp, "trajectory_id": traj, "t": t, "condition": c,
                  "question": "state", "regime": args.regime}
            fp_rows.append(fp)
            del inputs
        # per-row manifest (window + per-condition corruption summary)
        manifest.append({"key": f"{traj}_t{t}", "traj": traj, "t": t,
                         "clip_end": clip_end,
                         "initial_state": r["initial_state"],
                         "gt_event": r["gt_event"], "gt_state": r["gt_state"],
                         "gt_prev_state": r["gt_prev_state"],
                         "is_transition": r["is_transition"],
                         "cond": row_meta})
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            # periodic save (crash-safe)
            for c in COND_ORDER:
                np.savez_compressed(out / f"hidden_exp1_{c}.npz.partial",
                                    **{f"row{j}": hid[c][j]
                                       for j in range(i + 1)})
            with open(out / "behavior_exp1.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(beh_rows[0].keys()))
                w.writeheader()
                w.writerows(beh_rows)
            print(f"  [{i+1}/{len(rows)}] {traj}_t{t} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        done_rows = i + 1

    # final save: npz keyed by row key
    keys = [f"{r['trajectory_id']}_t{r['t']}" for r in rows]
    for c in COND_ORDER:
        z = {keys[j]: hid[c][j] for j in range(len(rows))}
        np.savez(out / f"hidden_exp1_{c}.npz", **z)
    with open(out / "behavior_exp1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(beh_rows[0].keys()))
        w.writeheader()
        w.writerows(beh_rows)
    with open(out / "fingerprints_exp1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
        w.writeheader()
        w.writerows(fp_rows)
    with open(out / "exp1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    # remove partials
    for c in COND_ORDER:
        p = out / f"hidden_exp1_{c}.npz.partial"
        if p.exists():
            p.unlink()
    print(f"Exp1 done: {len(rows)} rows x {len(COND_ORDER)} cond in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
