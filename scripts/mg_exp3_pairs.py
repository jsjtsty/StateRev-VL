#!/usr/bin/env python3
"""Mechanism Gate Experiment 3 - select matched source-target pairs (CPU).

A causal patching pair (S -> T) requires S and T to have the SAME t (same
grid / video-token layout, so window positions match) and the source's event
E_S to produce a COUNTERFACTUAL state St_cf = transition(prev_T, E_S) that
differs from T's own state St_T, so a causal shift is measurable.

Also selects, per target:
  S  : same t, E_S != E_T, St_cf != St_T, different trajectory  (main)
  S3 : same t, E_S3 == E_T, different trajectory                (same-event
       content control: patching a same-event source should NOT shift toward
       the counterfactual state)
Controls implemented in the GPU runner:
  main       : S window video activations  -> T window positions
  self       : T window video activations  -> T window positions (no-op)
  position   : S window video activations  -> T HISTORY positions
  sameevent  : S3 window video activations -> T window positions

Writes exp3_pairs.json (the pre-registered pair manifest).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def apply_event(prev: str, event: str) -> str:
    a, b = event.split(" and ")
    if prev == a:
        return b
    if prev == b:
        return a
    return prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--out", default="outputs/vetbench/mechanism_gate_v1/"
                                     "exp3_pairs.json")
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.behavior_csv, newline="")))
    assert len(rows) == 250, len(rows)
    for r in rows:
        r["t"] = int(r["t"])

    # group true-transition rows by t (targets must be true transitions)
    by_t = {}
    for r in rows:
        if r["is_transition"].strip().lower() == "true":
            by_t.setdefault(r["t"], []).append(r)

    import random
    rng = random.Random(args.seed)

    # generate ALL valid candidate pairs (target must have both S and S3)
    candidates = []
    for t in sorted(by_t.keys()):
        for T in by_t[t]:
            prev_T = T["gt_prev_state"]
            E_T = T["gt_event"]
            St_T = T["gt_state"]
            # main source: same t, different event, valid counterfactual
            S, St_cf = None, None
            for cand in by_t[t]:
                if cand["trajectory_id"] == T["trajectory_id"]:
                    continue
                if cand["gt_event"] == E_T:
                    continue
                st_cf = apply_event(prev_T, cand["gt_event"])
                if st_cf in ("Left", "Middle", "Right") and st_cf != St_T:
                    S, St_cf = cand, st_cf
                    break
            # same-event source: same t, same event, different trajectory
            S3 = None
            for cand in by_t[t]:
                if cand["trajectory_id"] == T["trajectory_id"]:
                    continue
                if cand["gt_event"] == E_T:
                    S3 = cand
                    break
            if S is None or S3 is None:
                continue
            candidates.append({
                "t": t,
                "target": {"traj": T["trajectory_id"], "t": t,
                           "gt_state": St_T, "gt_event": E_T,
                           "prev_state": prev_T,
                           "initial_state": T["initial_state"],
                           "frame_end": T["frame_end"]},
                "source": {"traj": S["trajectory_id"], "t": t,
                           "gt_event": S["gt_event"],
                           "initial_state": S["initial_state"],
                           "frame_end": S["frame_end"]},
                "source_sameevent": {"traj": S3["trajectory_id"], "t": t,
                                     "gt_event": S3["gt_event"],
                                     "initial_state": S3["initial_state"],
                                     "frame_end": S3["frame_end"]},
                "counterfactual_state": St_cf,
            })

    # balance-select n_pairs across t via round-robin
    rng.shuffle(candidates)
    by_t_cand = {}
    for c in candidates:
        by_t_cand.setdefault(c["t"], []).append(c)
    ts = sorted(by_t_cand.keys())
    pairs = []
    while len(pairs) < args.n_pairs:
        added = False
        for t in ts:
            if len(pairs) >= args.n_pairs:
                break
            if by_t_cand[t]:
                pairs.append(by_t_cand[t].pop(0))
                added = True
        if not added:
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_pairs": len(pairs), "seed": args.seed, "pairs": pairs},
                  f, indent=1)
    print(f"selected {len(pairs)} pairs -> {out}")
    # summary
    from collections import Counter
    print("  by t:", dict(Counter(p["t"] for p in pairs)))
    print("  cf states:", dict(Counter(p["counterfactual_state"] for p in pairs)))


if __name__ == "__main__":
    main()
