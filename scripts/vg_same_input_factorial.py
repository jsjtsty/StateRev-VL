#!/usr/bin/env python3
"""Validity Gate Stage 2: same-input factorial (behavior + hidden, 2 regimes).

For every one of the 250 rows (50 trajectories x t=1..5) and BOTH regimes
(actual_2fps, controlled_8fps), using ONE shared render_inputs() from
state_rev_input_pipeline so behavior and hidden extraction provably consume
the same input:

  A. state-question behavior  : greedy answer + native first-token
                                logprobs for Left/Middle/Right
  B. event-question behavior  : greedy answer + 3-class teacher-forced
                                logprob + GT margin
  C. state-question hidden    : 37-layer hidden states at the LAST INPUT
                                TOKEN (base model, no lm_head)

The event probe in the downstream analysis MUST use the STATE-question
hidden states (C), never the event-question forward. The state-question
input contains no GT state / GT event.

Outputs (under --out-dir):
  behavior_2fps.csv, behavior_8fps.csv
  hidden_states_2fps.npz, hidden_states_8fps.npz   (traj_t -> 37 x 4096)
  input_fingerprints.csv
  factorial_run_summary.json

Crash-safe: every finished (row, regime) is appended to the CSVs and the
npz is re-saved every row; already-done rows are skipped on restart.
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
import state_rev_input_pipeline as pipe
from state_rev_input_pipeline import (
    EVENT_CLASSES, EVENT_TOKEN_IDS, POS_IDS, render_inputs,
    extract_hidden_states, first_token_logits, teacher_force_logprob,
    load_model_and_processor,
)

REGIMES = ("actual_2fps", "controlled_8fps")
REGIME_TAG = {"actual_2fps": "2fps", "controlled_8fps": "8fps"}


def T(x) -> bool:
    return str(x).strip().lower() == "true"


def parse_event(text: str) -> str | None:
    t = text.lower().replace(",", " ")
    for ev in EVENT_CLASSES:
        a, b = ev.split(" and ")
        if a.lower() in t and b.lower() in t:
            return ev
    return None


def load_rows(behavior_csv: Path):
    rows = list(csv.DictReader(open(behavior_csv, newline="")))
    assert len(rows) == 250
    return rows


def done_set(out_dir: Path, regime: str) -> set:
    tag = REGIME_TAG[regime]
    p = out_dir / f"behavior_{tag}.csv"
    done = set()
    if p.exists():
        for r in csv.DictReader(open(p)):
            done.add(f"{r['trajectory_id']}_t{r['t']}")
    return done


def append_behavior(out_dir: Path, regime: str, rec: dict) -> None:
    tag = REGIME_TAG[regime]
    p = out_dir / f"behavior_{tag}.csv"
    new = not p.exists()
    fields = ["trajectory_id", "t", "regime",
              "state_pred", "state_correct", "state_answer",
              "logprob_Left", "logprob_Middle", "logprob_Right",
              "event_pred", "event_correct", "event_answer",
              "event_logprob_gt", "event_logprob_max_other", "event_gt_margin",
              "n_state_tokens", "n_event_tokens",
              "state_input_len", "event_input_len",
              "state_input_ids_sha", "event_input_ids_sha"]
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in fields})


def append_fp(out_dir: Path, fp: dict, traj: str, t: int, question: str) -> None:
    p = out_dir / "input_fingerprints.csv"
    new = not p.exists()
    fields = ["trajectory_id", "t", "regime", "question", "requested_fps",
              "clip_frames", "n_video_frames", "video_grid_thw",
              "input_ids_len", "input_ids_sha256",
              "pixel_values_videos_shape", "pixel_values_videos_sha256",
              "prompt_text_sha256", "transformers_version", "torch_version"]
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        vgw = ",".join(str(x) for x in fp["video_grid_thw"])
        pv = ",".join(str(x) for x in fp["pixel_values_videos_shape"])
        w.writerow({
            "trajectory_id": traj, "t": t, "regime": fp["regime"],
            "question": question, "requested_fps": fp["requested_fps"],
            "clip_frames": fp["clip_frames"],
            "n_video_frames": fp["n_video_frames"], "video_grid_thw": vgw,
            "input_ids_len": fp["input_ids_len"],
            "input_ids_sha256": fp["input_ids_sha256"],
            "pixel_values_videos_shape": pv,
            "pixel_values_videos_sha256": fp["pixel_values_videos_sha256"],
            "prompt_text_sha256": fp["prompt_text_sha256"],
            "transformers_version": fp["transformers_version"],
            "torch_version": fp["torch_version"],
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--dataset", default="dataset/vetbench/cup")
    ap.add_argument("--out-dir",
                    default="outputs/vetbench/validity_gate_v1")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--limit-trajectories", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = [r for r in args.regimes.split(",") if r]

    from run_state_rev_audit import state_messages, event_messages
    from run_vetbench_screening import sample_clip

    rows = load_rows(Path(args.behavior_csv))
    trajs = sorted({r["trajectory_id"] for r in rows})
    if args.limit_trajectories:
        trajs = trajs[:args.limit_trajectories]
    rows = [r for r in rows if r["trajectory_id"] in set(trajs)]
    print(f"Running {len(rows)} rows x {len(regimes)} regimes ...", flush=True)

    model, processor = load_model_and_processor(Path(args.model_dir))
    dev = next(model.parameters()).device

    npz = {regime: {} for regime in regimes}
    for regime in regimes:
        tag = REGIME_TAG[regime]
        p = out_dir / f"hidden_states_{tag}.npz"
        if p.exists():
            loaded = np.load(p)
            npz[regime] = {k: loaded[k] for k in loaded.files}

    t0 = time.time()
    n_done = 0
    for regime in regimes:
        tag = REGIME_TAG[regime]
        already = done_set(out_dir, regime)
        for i, r in enumerate(rows):
            key = f"{r['trajectory_id']}_t{r['t']}"
            if regime in ("actual_2fps", "controlled_8fps") and key in already:
                continue
            video = Path(args.dataset) / f"{r['trajectory_id']}.mp4"
            clip = sample_clip(video, 0, int(r["frame_end"]))
            n_swaps = int(r["n_swaps_shown"])

            # ---- STATE question: hidden (C) + behavior (A) from ONE input
            s_msgs = state_messages(clip, r["initial_state"], n_swaps)
            s_inputs, s_fp = render_inputs(processor, s_msgs, clip, regime)
            hidden = extract_hidden_states(model, s_inputs)
            s_beh = first_token_logits(model, processor, s_inputs, POS_IDS)
            del s_inputs

            # ---- EVENT question: behavior (B)
            e_msgs = event_messages(clip, n_swaps)
            e_inputs, e_fp = render_inputs(processor, e_msgs, clip, regime)
            e_first = first_token_logits(
                model, processor, e_inputs,
                {"Left": 5415, "Middle": 43935})
            ev_lps = {ev: teacher_force_logprob(model, e_inputs,
                                                 EVENT_TOKEN_IDS[ev])
                      for ev in EVENT_CLASSES}
            del e_inputs

            gt_ev = r["gt_event"]
            lp_gt = ev_lps[gt_ev]
            lp_max_other = max(v for k, v in ev_lps.items() if k != gt_ev)
            ev_pred = parse_event(e_first["answer"])
            st_pred = s_beh["answer"].strip()

            rec = {
                "trajectory_id": r["trajectory_id"], "t": r["t"],
                "regime": regime,
                "state_pred": st_pred,
                "state_correct": "true" if st_pred == r["gt_state"] else "false",
                "state_answer": s_beh["answer"],
                "logprob_Left": f"{s_beh['first_logprobs']['Left']:.6f}",
                "logprob_Middle": f"{s_beh['first_logprobs']['Middle']:.6f}",
                "logprob_Right": f"{s_beh['first_logprobs']['Right']:.6f}",
                "event_pred": ev_pred or "",
                "event_correct": "true" if ev_pred == gt_ev else "false",
                "event_answer": e_first["answer"],
                "event_logprob_gt": f"{lp_gt:.6f}",
                "event_logprob_max_other": f"{lp_max_other:.6f}",
                "event_gt_margin": f"{lp_gt - lp_max_other:.6f}",
                "n_state_tokens": s_beh["n_answer_tokens"],
                "n_event_tokens": e_first["n_answer_tokens"],
                "state_input_len": s_fp["input_ids_len"],
                "event_input_len": e_fp["input_ids_len"],
                "state_input_ids_sha": s_fp["input_ids_sha256"],
                "event_input_ids_sha": e_fp["input_ids_sha256"],
            }
            append_behavior(out_dir, regime, rec)
            append_fp(out_dir, s_fp, r["trajectory_id"], r["t"], "state")
            append_fp(out_dir, e_fp, r["trajectory_id"], r["t"], "event")
            npz[regime][key] = hidden
            n_done += 1
            if n_done % 5 == 0 or n_done == 1:
                for rg in regimes:
                    np.savez(out_dir / f"hidden_states_{REGIME_TAG[rg]}.npz",
                             **npz[rg])
                el = time.time() - t0
                print(f"  [{n_done}] {regime} {key} "
                      f"state={st_pred} event={ev_pred} "
                      f"({el:.0f}s elapsed)", flush=True)
            torch.cuda.empty_cache()

    # final save
    for rg in regimes:
        np.savez(out_dir / f"hidden_states_{REGIME_TAG[rg]}.npz",
                 **npz[rg])
    summary = {
        "n_rows": len(rows), "regimes": regimes,
        "trajectories": trajs if args.limit_trajectories else None,
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": "state hidden (C) and state behavior (A) share ONE rendered "
                "input; event probe must use the STATE hidden states.",
    }
    (out_dir / "factorial_run_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"DONE. {n_done} new (row,regime) in {time.time()-t0:.0f}s. "
          f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
