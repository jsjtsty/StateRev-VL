#!/usr/bin/env python3
"""Mechanism Gate smoke test (3 trajectories, controlled_8fps).

Validates BEFORE any full run:
  1. baseline render reproduces the validity_gate_v1 fingerprint exactly.
  2. each manipulation preserves grid AND input_ids (timestamps intact), and
     changes pixel_values ONLY in the expected temporal blocks.
  3. token-region identification is sane.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_rev_input_pipeline import load_model_and_processor, render_inputs
import mg_common as mg
from run_state_rev_audit import state_messages, state_question_text
from run_vetbench_screening import sample_clip


def main():
    model, processor = load_model_and_processor(
        Path("models/Qwen3-VL-8B-Instruct"))
    traj = "cup_001"
    cases = [(1, 117), (2, 177), (3, 237)]
    ref = {}
    for r in csv.DictReader(open("outputs/vetbench/validity_gate_v1/"
                                 "input_fingerprints.csv")):
        if r["regime"] == "controlled_8fps" and r["question"] == "state":
            ref[f"{r['trajectory_id']}_t{r['t']}"] = r

    all_ok = True
    for t, clip_end in cases:
        clip = sample_clip(Path("dataset/vetbench/cup") / f"{traj}.mp4", 0,
                           clip_end)
        msgs = state_messages(clip, "Left", t)
        qtext = state_question_text("Left", t)
        base_inputs, base_fp = render_inputs(processor, msgs, clip,
                                             "controlled_8fps")
        base_pv = base_inputs["pixel_values_videos"]
        key = f"{traj}_t{t}"
        r = ref.get(key)
        if r:
            ok_ids = base_fp["input_ids_sha256"] == r["input_ids_sha256"]
            ok_pv = base_fp["pixel_values_videos_sha256"] == \
                r["pixel_values_videos_sha256"]
            ok_grid = base_fp["video_grid_thw"] == list(
                map(int, r["video_grid_thw"].strip('"').split(",")))
            print(f"[{key}] baseline fp match: ids={ok_ids} "
                  f"pixel={ok_pv} grid={ok_grid}")
            all_ok &= (ok_ids and ok_pv and ok_grid)
        grid = base_fp["video_grid_thw"]
        grid_hw = grid[1] * grid[2]
        num_frames = base_fp["n_video_frames"]
        in_win = mg.window_sample_mask(clip_end, t, num_frames)
        W = np.where(in_win)[0]
        print(f"[{key}] grid={grid} num_frames={num_frames} "
              f"n_window_frames={len(W)} win={mg.swap_window_frames(clip_end, t)}")

        ids = base_inputs["input_ids"][0].tolist()
        mm = base_inputs.get("mm_token_type_ids")
        regions = mg.identify_token_regions(ids, mm, qtext, processor)
        vp = regions["_video_span"]
        vt, pt = mg.video_window_token_regions(vp, grid, in_win)
        print(f"[{key}] video={len(vp)} window_tok={len(vt)} "
              f"pre_tok={len(pt)} qspan={len(regions['question_mean'])} "
              f"cur={len(regions['currently'])} final={regions['final_prompt'][0]}")

        for cond in ["freeze", "shuffle", "history_freeze"]:
            clip1, meta = mg.build_manipulated_clip(clip, t, "controlled_8fps",
                                                    cond)
            msgs1 = state_messages(clip1, "Left", t)
            m_inputs, m_fp = render_inputs(processor, msgs1, clip1,
                                           "controlled_8fps")
            same_grid = m_fp["video_grid_thw"] == base_fp["video_grid_thw"]
            same_ids = m_fp["input_ids_sha256"] == base_fp["input_ids_sha256"]
            diff = mg.patch_diff_mask(base_pv, m_inputs["pixel_values_videos"])
            if cond in ("freeze", "shuffle"):
                exp_blocks = set((W // 2).tolist())
            else:
                # corrupted clip frames -> their sample index -> block
                cc = np.array(meta["corrupted_clip_frames"])
                si = np.searchsorted(mg.sampled_indices(clip_end, num_frames), cc)
                exp_blocks = set((si // 2).tolist())
            diff_blocks = set((np.where(diff)[0] // grid_hw).tolist())
            extra = diff_blocks - exp_blocks
            missing = exp_blocks - diff_blocks
            print(f"[{key}] {cond:14s} grid_ok={same_grid} ids_ok={same_ids} "
                  f"diff_blocks={len(diff_blocks)} (exp {len(exp_blocks)}) "
                  f"extra={sorted(extra)} missing={sorted(missing)} "
                  f"n_corr={meta['n_corrupted']}")
            all_ok &= (same_grid and same_ids and not extra)
            del m_inputs
        del base_inputs
    print("\nSMOKE", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
