#!/usr/bin/env python3
"""Validity Gate Stage 6: rescue control (pre-registered 5 conditions).

Goal: determine whether the state answer follows the VIDEO or follows
injected TEXT cues. Five conditions, all on the same 2 fps actual regime,
all 250 rows (no re-selection on results). Cues use the exact pre-registered
wording; "wrong" cues are deterministic pre-fixed mappings (always != GT).

Conditions
  1 baseline            : original state question + video
  2 prior_correct       : + correct prior cue            + video
  3 prior_event_correct : + correct prior + correct event + video
  4 text_only_correct   : same text as (3), NO video (pure text)
  5 text_only_wrong     : wrong prior + wrong event, NO video

"wrong" pre-fixed mappings (deterministic, always a different value):
  WRONG_PREV    = {Left: Middle, Middle: Right, Right: Left}
  WRONG_EVENT   = {Left and Middle: Middle and Right,
                   Middle and Right: Left and Right,
                   Left and Right: Left and Middle}

For each condition we record:
  pred_state, matches_gt (== S_t), matches_prev (== S_{t-1}),
  text_implied (position implied by the stated prior+event, where defined),
  matches_text_implied (pred == text_implied).

text_implied is defined for conditions with a full prior+event cue:
  (3),(4): apply E_t to S_{t-1} = S_t
  (5):     apply wrong_E to wrong_P

Output (under --out-dir): rescue_control_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_rev_input_pipeline as pipe
from state_rev_input_pipeline import (
    render_inputs, first_token_logits, load_model_and_processor, to_device,
)
from run_state_rev_audit import SYSTEM_PROMPT, state_question_text

POSITIONS = ("Left", "Middle", "Right")
WRONG_PREV = {"Left": "Middle", "Middle": "Right", "Right": "Left"}
WRONG_EVENT = {"Left and Middle": "Middle and Right",
               "Middle and Right": "Left and Right",
               "Left and Right": "Left and Middle"}
INSERT_ANCHOR = ("The ball is under the cup that is currently at one of the "
                 "three positions.")
PRIOR_TMPL = ("In the previous segment, the ball was at the {PREV} position. "
              "This is the verified starting state.")
EVENT_TMPL = ("In this segment, the {A} and {B} positions were swapped.")


def T(x) -> bool:
    return str(x).strip().lower() == "true"


def apply_swap(pos: str, event: str) -> str:
    a, b = event.split(" and ")
    if pos == a:
        return b
    if pos == b:
        return a
    return pos


def _insert(baseline: str, sentences: str) -> str:
    assert baseline.count(INSERT_ANCHOR) == 1
    return baseline.replace(INSERT_ANCHOR, sentences + " " + INSERT_ANCHOR)


def build_conditions(row: dict) -> list[tuple[str, str, bool, str | None]]:
    """Return [(name, text, has_video, text_implied)] for the 5 conditions."""
    baseline = state_question_text(row["initial_state"],
                                   int(row["n_swaps_shown"]))
    a, b = row["gt_event"].split(" and ")
    prior_c = PRIOR_TMPL.format(PREV=row["gt_prev_state"])
    event_c = EVENT_TMPL.format(A=a, B=b)
    prior_w = PRIOR_TMPL.format(PREV=WRONG_PREV[row["gt_prev_state"]])
    event_w = EVENT_TMPL.format(*WRONG_EVENT[row["gt_event"]].split(" and "))
    text3 = _insert(baseline, prior_c + " " + event_c)
    text5 = _insert(baseline, prior_w + " " + event_w)
    return [
        ("baseline", baseline, True, None),
        ("prior_correct", _insert(baseline, prior_c), True, None),
        ("prior_event_correct", text3, True, row["gt_state"]),
        ("text_only_correct", text3, False, row["gt_state"]),
        ("text_only_wrong", text5, False,
         apply_swap(WRONG_PREV[row["gt_prev_state"]],
                    WRONG_EVENT[row["gt_event"]])),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--dataset", default="dataset/vetbench/cup")
    ap.add_argument("--out-dir", default="outputs/vetbench/validity_gate_v1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.behavior_csv, newline="")))
    assert len(rows) == 250
    if args.limit:
        rows = rows[:args.limit]
    from run_vetbench_screening import sample_clip
    model, processor = load_model_and_processor(Path(args.model_dir))
    out_csv = out_dir / "rescue_control_results.csv"
    fields = ["trajectory_id", "t", "condition", "has_video", "pred_state",
              "matches_gt", "matches_prev", "text_implied",
              "matches_text_implied"]
    done = set()
    if out_csv.exists():
        for x in csv.DictReader(open(out_csv)):
            done.add(f"{x['trajectory_id']}_t{x['t']}|{x['condition']}")
    new = not out_csv.exists()
    t0 = time.time()
    for i, r in enumerate(rows):
        key = f"{r['trajectory_id']}_t{r['t']}"
        clip = sample_clip(Path(args.dataset) / f"{r['trajectory_id']}.mp4",
                           0, int(r["frame_end"]))
        for name, text, has_video, text_impl in build_conditions(r):
            k = f"{key}|{name}"
            if k in done:
                continue
            if has_video:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "video", "video": clip},
                            {"type": "text", "text": text}]}]
                inputs, _ = render_inputs(processor, msgs, clip, "actual_2fps")
            else:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text}]
                inputs = processor.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
            first = first_token_logits(model, processor, inputs,
                                       {"Left": 5415, "Middle": 43935})
            del inputs
            pred = first["answer"]
            with open(out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new:
                    w.writeheader()
                    new = False
                w.writerow({
                    "trajectory_id": r["trajectory_id"], "t": r["t"],
                    "condition": name, "has_video": str(has_video).lower(),
                    "pred_state": pred,
                    "matches_gt": str(pred == r["gt_state"]).lower(),
                    "matches_prev": str(pred == r["gt_prev_state"]).lower(),
                    "text_implied": text_impl or "",
                    "matches_text_implied":
                        ("" if text_impl is None
                         else str(pred == text_impl).lower()),
                })
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] {key} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"rescue control done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
