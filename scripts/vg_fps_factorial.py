#!/usr/bin/env python3
"""Validity Gate Stage 4 + 5: same-input fps factorial contrast + dissociation
re-examination (CPU-only; consumes Stage 2 behavior CSVs and Stage 2b probe
results).

Stage 4 - 2 fps vs 8 fps on the SAME rows / SAME prompt / SAME processor:
  * per-regime state & event accuracy (overall + true-transition subset)
  * paired per-row contrast: how many rows FLIP state / event between regimes
  * Pattern A/B/C/D/E classification (pre-registered)
  * trajectory-bootstrap CI for the regime accuracy differences

Stage 5 - event/state dissociation RE-EXAMINED on the same regime (NOT using
  an independent event prompt as an "internal" property):
  * behavioral: event_correct & state-wrong within each regime; stale vs other
  * representational: probe decodability of E_t vs S_t vs S_{t-1} from the
    STATE-question hidden states (Stage 2b), per regime
  * explicit caveat recorded

Pattern definitions (paired, e2/e8 = event_correct @2/8 fps, s2/s8 =
state_correct @2/8 fps):
  C : e2 != e8                                   (event flips across regimes)
  else (event stable):
    B : e==1 and s2 != s8                          (state flips, event stable)
    A : e==1 and s2==s8==1                         (all correct both regimes)
    D : e==1 and s2==s8==0                         (state wrong both regimes)
    E : e==0                                       (event wrong both regimes)

Output: fps_factorial_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def T(x) -> bool:
    return str(x).strip().lower() == "true"


def load_behavior(path: Path) -> dict:
    d = {}
    for r in csv.DictReader(open(path)):
        d[f"{r['trajectory_id']}_t{r['t']}"] = r
    return d


def load_gt(path: Path) -> dict:
    d = {}
    for r in csv.DictReader(open(path)):
        d[f"{r['trajectory_id']}_t{r['t']}"] = r
    return d


def pattern(e2, e8, s2, s8) -> str:
    if e2 != e8:
        return "C_event_flips"
    if e2:
        if s2 != s8:
            return "B_state_flips"
        if s2 and s8:
            return "A_all_correct"
        return "D_state_wrong_both"
    return "E_event_wrong_both"


def traj_bootstrap(trajs: dict, metric, n_boot: int, seed: int):
    """Resample trajectories with replacement; metric takes the list of
    (traj_id -> rows) and returns a float. Returns (point, lo, hi)."""
    keys = list(trajs.keys())
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [trajs[k] for k in rng.choices(keys, k=len(keys))]
        vals.append(metric(sample))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return metric(trajs.values()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/vetbench/validity_gate_v1")
    ap.add_argument("--behavior-2",
                    default="outputs/vetbench/validity_gate_v1/"
                            "behavior_2fps.csv")
    ap.add_argument("--behavior-8",
                    default="outputs/vetbench/validity_gate_v1/"
                            "behavior_8fps.csv")
    ap.add_argument("--gt-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--probe-csv",
                    default="outputs/vetbench/validity_gate_v1/"
                            "same_input_probe_results.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    b2 = load_behavior(Path(args.behavior_2))
    b8 = load_behavior(Path(args.behavior_8))
    gt = load_gt(Path(args.gt_csv))
    keys = sorted(set(b2) & set(b8) & set(gt))
    res = {"n_rows": len(keys), "n_trajectories":
           len({k.rsplit("_t", 1)[0] for k in keys})}

    def _sub(rows):
        return [k for k in rows if T(gt[k]["is_transition"])]

    # ---- per-regime accuracy -------------------------------------------
    def acc(rows, regime_b, field):
        sel = [k for k in rows if k in regime_b]
        if not sel:
            return None
        return sum(1 for k in sel if T(regime_b[k][field])) / len(sel)

    per = {}
    for subset, rows in (("all", keys),
                         ("true_transition", _sub(keys))):
        per[subset] = {
            "n": len(rows),
            "state_acc_2fps": acc(rows, b2, "state_correct"),
            "state_acc_8fps": acc(rows, b8, "state_correct"),
            "event_acc_2fps": acc(rows, b2, "event_correct"),
            "event_acc_8fps": acc(rows, b8, "event_correct"),
        }
    res["accuracy"] = per

    # ---- paired contrast + patterns ------------------------------------
    patterns = defaultdict(int)
    state_flip = event_flip = 0
    state_flip_correct_2 = state_flip_correct_8 = 0
    for k in keys:
        e2, e8 = T(b2[k]["event_correct"]), T(b8[k]["event_correct"])
        s2, s8 = T(b2[k]["state_correct"]), T(b8[k]["state_correct"])
        patterns[pattern(e2, e8, s2, s8)] += 1
        if e2 != e8:
            event_flip += 1
        if s2 != s8:
            state_flip += 1
            if s2:
                state_flip_correct_2 += 1
            if s8:
                state_flip_correct_8 += 1
    res["paired"] = {
        "patterns": dict(patterns),
        "state_flip_rows": state_flip,
        "state_flip_correct_in_2fps": state_flip_correct_2,
        "state_flip_correct_in_8fps": state_flip_correct_8,
        "event_flip_rows": event_flip,
    }

    # ---- trajectory bootstrap CI on accuracy diffs ---------------------
    trajs = defaultdict(list)
    for k in keys:
        trajs[k.rsplit("_t", 1)[0]].append(k)

    def diff_metric(sample_rows, regime_b, field):
        # returns acc_8 - acc_2 restricted to sampled rows
        sel = [k for k in sample_rows if k in b2 and k in b8]
        if not sel:
            return 0.0
        a2 = sum(1 for k in sel if T(b2[k][field])) / len(sel)
        a8 = sum(1 for k in sel if T(b8[k][field])) / len(sel)
        return a8 - a2

    boot = {}
    for field in ("state_correct", "event_correct"):
        pt, lo, hi = traj_bootstrap(trajs,
                                    lambda rows: diff_metric(rows, None, field),
                                    args.n_boot, seed=20260902)
        boot[f"acc_8fps_minus_2fps_{field.replace('_correct','')}"] = {
            "point": pt, "ci95_lo": lo, "ci95_hi": hi}
    res["bootstrap_ci"] = boot

    # ---- Stage 5: dissociation -----------------------------------------
    diss = {}
    for regime, b in (("2fps", b2), ("8fps", b8)):
        ec_sw = 0  # event correct, state wrong
        stale = other = 0
        ec = sc = 0
        for k in keys:
            e = T(b[k]["event_correct"])
            s = T(b[k]["state_correct"])
            ec += e
            sc += s
            if e and not s:
                ec_sw += 1
                if b[k]["state_pred"] == gt[k]["gt_prev_state"]:
                    stale += 1
                else:
                    other += 1
        diss[regime] = {
            "event_correct": ec, "state_correct": sc,
            "event_correct_state_wrong": ec_sw,
            "of_those_stale": stale, "of_those_other": other,
            "event_state_gap": ec_sw / max(ec, 1),
        }
    res["dissociation_behavioral"] = diss

    # representational (probe) - optional
    pcsv = Path(args.probe_csv)
    if pcsv.exists():
        rows = list(csv.DictReader(open(pcsv)))
        # columns: regime, target, subset, layer, bal_acc, ... (see probe)
        rep = {}
        for regime in ("2fps", "8fps"):
            best = {}
            for tgt in ("event", "state", "prev_state"):
                vals = [float(r["bal_acc"]) for r in rows
                        if r.get("regime") == regime and r.get("target") == tgt
                        and r.get("subset") == "all"]
                best[tgt] = max(vals) if vals else None
            rep[regime] = best
        res["dissociation_representational"] = rep
        res["dissociation_caveat"] = (
            "behavioral event_correct comes from an independent event prompt "
            "(same video, different question), NOT from the state forward's "
            "internal representation; the representational comparison uses a "
            "linear probe on the STATE-question hidden states for E_t / S_t / "
            "S_{t-1}. The two are converging evidence, not identical.")
    res["stage5_caveat"] = (
        "The 'event known but state unknown' dissociation is tested WITHIN a "
        "single regime (same video, same processor). We do NOT treat "
        "event_correct from a separate event question as proof of an internal "
        "event representation; the probe-based comparison is the "
        "representation-level check.")

    with open(out / "fps_factorial_summary.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out / 'fps_factorial_summary.json'}")
    print(json.dumps({k: res[k] for k in
                      ("accuracy", "paired", "bootstrap_ci",
                       "dissociation_behavioral") if k in res}, indent=1))


if __name__ == "__main__":
    main()
