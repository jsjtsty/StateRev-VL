#!/usr/bin/env python3
"""Validity Gate Stage 3: input-sufficiency gate for the 2 fps event.

Two parts:

(A) GPU - three PRE-FIXED, semantically-equivalent event-prompt variants,
    run on the ACTUAL 2 fps regime for all 250 rows. Each variant yields
    the greedy event answer + 3-class teacher-forced logprob + GT margin.
    We do NOT tune prompts to raise accuracy; the variants are fixed before
    looking at results and used only to test answer consistency.

(B) CPU - frame coverage: for each t, compute the exact 2 fps sampled frame
    indices (uniform linspace over [0, frame_end)) and count how many fall
    BEFORE / DURING / AFTER the swap-t window [57+(t-1)*60, 57+t*60). This
    tests whether the model ever sees a settled post-swap state.

C) event_known (pre-registered): a row's event is "known" at threshold m if
    (i) the GT event is predicted correctly by the primary variant,
    (ii) GT-vs-runner-up margin > m, and (iii) all 3 variants agree on the
    predicted event. m is fixed before subgroup analysis; sensitivity is
    reported at m in {0.5, 1.0, 1.5}.

Outputs (under --out-dir):
  event_variants_2fps.csv
  event_sufficiency.json   (accuracy, agreement, event_known rates,
                            per-t / per-swap-type, frame coverage)
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
    EVENT_CLASSES, EVENT_TOKEN_IDS, render_inputs, first_token_logits,
    teacher_force_logprob, load_model_and_processor,
)

SWAP_T0, SWAP_LEN = 57, 60


def swap_window(t: int):
    return SWAP_T0 + (t - 1) * SWAP_LEN, SWAP_T0 + t * SWAP_LEN


def T(x) -> bool:
    return str(x).strip().lower() == "true"


def parse_event(text: str) -> str | None:
    t = text.lower().replace(",", " ")
    for ev in EVENT_CLASSES:
        a, b = ev.split(" and ")
        if a.lower() in t and b.lower() in t:
            return ev
    return None


# ---- PRE-FIXED event prompt variants (semantically equivalent) ----------
def _ev_text(n: int) -> list[str]:
    opts = ("(A) Left and Middle (B) Middle and Right (C) Left and Right. "
            'Answer with the option text, e.g. "Left and Middle".')
    return [
        # V0 = original (from run_state_rev_audit.event_question_text)
        (f"Three identical cups are at the fixed positions Left, Middle and "
         f"Right. {n} swap(s) have happened in this video. Focus on the swap "
         f"that just happened, i.e. the last (most recent) swap shown in "
         f"this video. Which two positions were swapped in that last swap? "
         + opts),
        # V1
        (f"Three identical cups are at the fixed positions Left, Middle and "
         f"Right. {n} swap(s) have happened in this video. Look at the most "
         f"recent swap only. Which two positions exchanged cups in that "
         f"swap? " + opts),
        # V2
        (f"Three identical cups sit at the fixed positions Left, Middle and "
         f"Right. In this video {n} swap(s) occurred. Consider the final "
         f"swap only. What two positions were involved in the last swap? "
         + opts),
    ]


def frame_coverage(clip_end: int, t: int):
    """Exact 2 fps sampled frame indices (linspace over [0,clip_end)), split
    by before/during/after the swap-t window."""
    n_frames = max(1, int(clip_end / 24.0 * 2))
    idx = np.linspace(0, clip_end - 1, n_frames).round().astype(int)
    w0, w1 = swap_window(t)
    before = int((idx < w0).sum())
    during = int(((idx >= w0) & (idx < w1)).sum())
    after = int((idx >= w1).sum())
    return {"n_sampled": int(n_frames), "before": before, "during": during,
            "after": after, "window": [w0, w1], "clip_end": int(clip_end)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--dataset", default="dataset/vetbench/cup")
    ap.add_argument("--out-dir", default="outputs/vetbench/validity_gate_v1")
    ap.add_argument("--analysis-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.behavior_csv, newline="")))
    assert len(rows) == 250

    if not args.analysis_only:
        from run_state_rev_audit import SYSTEM_PROMPT
        from run_vetbench_screening import sample_clip
        model, processor = load_model_and_processor(Path(args.model_dir))
        nparts = 3
        out_csv = out_dir / "event_variants_2fps.csv"
        fields = ["trajectory_id", "t", "variant", "event_pred",
                  "event_correct", "event_answer", "event_logprob_gt",
                  "event_logprob_max_other", "event_gt_margin"]
        new = not out_csv.exists()
        t0 = time.time()
        for i, r in enumerate(rows):
            key = f"{r['trajectory_id']}_t{r['t']}"
            clip = sample_clip(Path(args.dataset) / f"{r['trajectory_id']}.mp4",
                               0, int(r["frame_end"]))
            n_swaps = int(r["n_swaps_shown"])
            texts = _ev_text(n_swaps)
            for v in range(nparts):
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "video", "video": clip},
                            {"type": "text", "text": texts[v]}]}]
                inputs, fp = render_inputs(processor, msgs, clip,
                                           "actual_2fps")
                first = first_token_logits(model, processor, inputs,
                                           {"Left": 5415, "Middle": 43935})
                lps = {ev: teacher_force_logprob(model, inputs,
                                                  EVENT_TOKEN_IDS[ev])
                       for ev in EVENT_CLASSES}
                del inputs
                gt = r["gt_event"]
                pred = parse_event(first["answer"])
                rec = {"trajectory_id": r["trajectory_id"], "t": r["t"],
                       "variant": v, "event_pred": pred or "",
                       "event_correct": "true" if pred == gt else "false",
                       "event_answer": first["answer"],
                       "event_logprob_gt": f"{lps[gt]:.6f}",
                       "event_logprob_max_other":
                           f"{max(x for k, x in lps.items() if k != gt):.6f}",
                       "event_gt_margin":
                           f"{lps[gt] - max(x for k, x in lps.items()
                                            if k != gt):.6f}"}
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    if new:
                        w.writeheader()
                        new = False
                    w.writerow(rec)
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/250] {key} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"event variants done in {time.time()-t0:.0f}s")

    # ---- analysis (CPU) -------------------------------------------------
    vc = out_dir / "event_variants_2fps.csv"
    result = {"n_rows": len(rows), "variants": 3,
              "thresholds": [0.5, 1.0, 1.5]}
    # frame coverage per t
    cov = {}
    for r in rows:
        t = int(r["t"])
        if t not in cov:
            cov[t] = frame_coverage(int(r["frame_end"]), t)
    result["frame_coverage_2fps"] = {str(k): v for k, v in sorted(cov.items())}

    if vc.exists():
        vr = list(csv.DictReader(open(vc)))
        by_row = {}
        for x in vr:
            by_row.setdefault(f"{x['trajectory_id']}_t{x['t']}", {})[
                int(x["variant"])] = x
        per_var_acc, per_var_margin = {}, []
        for v in range(3):
            sub = [x for x in vr if int(x["variant"]) == v]
            per_var_acc[v] = sum(1 for x in sub if T(x["event_correct"])) / \
                max(len(sub), 1)
            per_var_margin.append(
                float(np.mean([float(x["event_gt_margin"]) for x in sub])))
        result["event_accuracy_by_variant"] = per_var_acc
        result["event_mean_gt_margin_by_variant"] = per_var_margin
        # agreement across variants
        agree_all = agree_v0 = 0
        n = 0
        known = {m: 0 for m in result["thresholds"]}
        per_t = {t: {"n": 0, "v0_correct": 0, "known": {m: 0 for m in result["thresholds"]}}
                 for t in (1, 2, 3, 4, 5)}
        per_swap = {}
        for r in rows:
            key = f"{r['trajectory_id']}_t{r['t']}"
            d = by_row.get(key)
            if not d or len(d) != 3:
                continue
            n += 1
            preds = [d[v]["event_pred"] for v in range(3)]
            t = int(r["t"])
            per_t[t]["n"] += 1
            if d[0]["event_correct"] == "true":
                per_t[t]["v0_correct"] += 1
            if len(set(preds)) == 1:
                agree_all += 1
            if preds[0] == preds[1] == preds[2]:
                agree_v0 += 1
            gt = r["gt_event"]
            m0 = float(d[0]["event_gt_margin"])
            correct0 = (d[0]["event_pred"] == gt)
            allagree = (len(set(preds)) == 1)
            for m in result["thresholds"]:
                if correct0 and m0 > m and allagree:
                    known[m] += 1
                    per_t[t]["known"][m] += 1
            st = r["gt_event"]
            per_swap.setdefault(st, {"n": 0, "v0_correct": 0})
            per_swap[st]["n"] += 1
            if d[0]["event_pred"] == gt:
                per_swap[st]["v0_correct"] += 1
        result["n_analyzed"] = n
        result["all3_agree_rate"] = agree_all / max(n, 1)
        result["event_known_rate"] = {str(m): known[m] / max(n, 1)
                                      for m in result["thresholds"]}
        result["event_known_count"] = {str(m): known[m]
                                       for m in result["thresholds"]}
        result["per_t"] = {str(t): {
            "n": v["n"],
            "v0_acc": v["v0_correct"] / max(v["n"], 1),
            **{f"known_{m}": v["known"][m] / max(v["n"], 1)
               for m in result["thresholds"]}}
            for t, v in per_t.items()}
        result["per_swap_type"] = {k: {"n": v["n"],
                                       "v0_acc": v["v0_correct"] / max(v["n"], 1)}
                                   for k, v in per_swap.items()}
    with open(out_dir / "event_sufficiency.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out_dir / 'event_sufficiency.json'}")
    print(json.dumps({k: result[k] for k in
                      ("frame_coverage_2fps", "event_accuracy_by_variant",
                       "event_known_rate", "all3_agree_rate")
                      if k in result}, indent=1))


if __name__ == "__main__":
    main()
