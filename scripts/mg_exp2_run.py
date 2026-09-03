#!/usr/bin/env python3
"""Mechanism Gate Experiment 2 - Token-position representation sweep (GPU).

Question (Gate B): is the current state S_t only "not routed" to the last
input token, i.e. is it linearly present at OTHER tokens/positions?

For all 250 baseline prefixes (controlled_8fps), one forward each, extract
the base-model hidden states (37 layers) at 7 pre-registered position
families, and save per family. The probe (mg_exp2_probe.py) then decodes
event/state/prev_state from each family and compares to the last-input-token
reference.

Position families:
  final_prompt  : last input token            (reference)
  question_mean : mean over the state-question text span
  currently     : the word 'currently'
  video_all     : mean over ALL video tokens
  video_last25  : mean over the last 25% of video tokens
  video_window  : mean over video tokens of the current swap window
  video_pre     : mean over video tokens strictly before the current window

Outputs (under --out-dir): hidden_exp2_{family}.npz (key -> 37x4096),
fingerprints_exp2.csv, exp2_regions.json (per-row region spans).
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
    first_token_logits, POS_IDS, expected_frames,
)
import mg_common as mg
from run_state_rev_audit import state_messages, state_question_text
from run_vetbench_screening import sample_clip

FAMILIES = ["final_prompt", "question_mean", "currently", "video_all",
            "video_last25", "video_window", "video_pre"]
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
    hid = {f: np.zeros((len(rows), N_LAYERS, D), dtype=np.float32)
           for f in FAMILIES}
    fp_rows = []
    regions_log = []
    t0 = time.time()

    for i, r in enumerate(rows):
        traj = r["trajectory_id"]
        t = int(r["t"])
        clip_end = int(r["frame_end"])
        clip = sample_clip(Path(args.dataset) / f"{traj}.mp4", 0, clip_end)
        msgs = state_messages(clip, r["initial_state"], t)
        qtext = state_question_text(r["initial_state"], t)
        inputs, fp = render_inputs(processor, msgs, clip, args.regime)
        ids = inputs["input_ids"][0].tolist()
        mm = inputs.get("mm_token_type_ids")
        grid = fp["video_grid_thw"]
        num_frames = fp["n_video_frames"]
        in_win = mg.window_sample_mask(clip_end, t, num_frames)

        reg = mg.identify_token_regions(ids, mm, qtext, processor)
        vp = reg["_video_span"]
        win_tok, pre_tok = mg.video_window_token_regions(vp, grid, in_win)
        families = {
            "final_prompt": reg["final_prompt"],
            "question_mean": reg["question_mean"],
            "currently": reg["currently"],
            "video_all": reg["video_all"],
            "video_last25": reg["video_last25"],
            "video_window": win_tok,
            "video_pre": pre_tok,
        }
        # skip empty families (record)
        nonempty = {f: p for f, p in families.items() if len(p) > 0}
        hs = mg.extract_hidden_at(model, inputs, None, nonempty)
        for f in FAMILIES:
            if f in hs:
                hid[f][i] = hs[f]
        fp = {**fp, "trajectory_id": traj, "t": t, "question": "state",
              "regime": args.regime}
        fp_rows.append(fp)
        regions_log.append({
            "key": f"{traj}_t{t}", "t": t,
            "n_final": len(families["final_prompt"]),
            "n_question": len(families["question_mean"]),
            "n_currently": len(families["currently"]),
            "n_video_all": len(families["video_all"]),
            "n_video_last25": len(families["video_last25"]),
            "n_video_window": len(families["video_window"]),
            "n_video_pre": len(families["video_pre"]),
        })
        del inputs
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] {traj}_t{t} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    keys = [f"{r['trajectory_id']}_t{r['t']}" for r in rows]
    for f in FAMILIES:
        z = {keys[j]: hid[f][j] for j in range(len(rows))}
        np.savez(out / f"hidden_exp2_{f}.npz", **z)
    with open(out / "fingerprints_exp2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fp_rows[0].keys()))
        w.writeheader()
        w.writerows(fp_rows)
    with open(out / "exp2_regions.json", "w") as fh:
        json.dump(regions_log, fh, indent=1)
    print(f"Exp2 done: {len(rows)} rows x {len(FAMILIES)} families in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
