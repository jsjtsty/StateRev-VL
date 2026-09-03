#!/usr/bin/env python3
"""Mechanism Gate Experiment 1 - Gate A analysis (CPU).

Question: does the event / current-state / prev-state CODE survive removal of
the CURRENT swap's visual dynamics (freeze / shuffle), relative to a matched
history-corruption control (history_freeze) and the baseline?

Inputs:
  behavior_exp1.csv        (per row x condition: state_correct, ...)
  exp1_probe_summary.json  (per condition: point + maxt)

Outputs:
  exp1_gateA_summary.json  (per condition: behavior acc + probe obs_max/p_maxT,
                            deltas from baseline, trajectory-bootstrap CI)
  exp1_gateA.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COND_ORDER = ["baseline", "freeze", "shuffle", "history_freeze"]
TARGETS = ["event", "state", "prev_state"]
SUBSETS = ["all", "true_transition"]


def boot_ci(values_by_traj, stat="mean", n=10000, seed=7):
    """Trajectory-cluster bootstrap CI for the mean of per-row values."""
    trajs = list(values_by_traj.keys())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        samp = rng.choice(trajs, size=len(trajs), replace=True)
        vals = np.concatenate([values_by_traj[t] for t in samp])
        boots.append(vals.mean())
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/vetbench/mechanism_gate_v1")
    args = ap.parse_args()
    out = Path(args.out_dir)

    # ---- behavior ----
    beh = list(csv.DictReader(open(out / "behavior_exp1.csv", newline="")))
    # transition mask from the GT (use baseline rows; is_transition per row)
    # behavior csv has no is_transition; load it from the source behavior
    src = list(csv.DictReader(open(
        "outputs/vetbench/composition_analysis_v1/transformers_behavior.csv",
        newline="")))
    trans = {f"{r['trajectory_id']}_t{r['t']}":
             r["is_transition"].strip().lower() == "true" for r in src}

    acc_by_cond = {}
    acc_trans_by_cond = {}
    boot = {}
    for c in COND_ORDER:
        rows = [r for r in beh if r["condition"] == c]
        acc = np.array([r["state_correct"] == "true" for r in rows], dtype=float)
        acc_by_cond[c] = float(acc.mean())
        # transition-only
        acc_t = np.array([acc[i] for i, r in enumerate(rows)
                          if trans[f"{r['trajectory_id']}_t{r['t']}"]])
        acc_trans_by_cond[c] = float(acc_t.mean()) if len(acc_t) else float("nan")
        # trajectory-cluster bootstrap (all rows)
        vbt = defaultdict(list)
        for r in rows:
            vbt[r["trajectory_id"]].append(
                1.0 if r["state_correct"] == "true" else 0.0)
        boot[c] = boot_ci(vbt)

    # ---- probe ----
    probe = json.load(open(out / "exp1_probe_summary.json"))
    probe_metric = {}
    for c in COND_ORDER:
        if c not in probe:
            continue
        d = probe[c]
        for tgt in TARGETS:
            for sub in SUBSETS:
                key = f"{tgt}|{sub}"
                m = d["maxt"].get(key)
                if m and not m.get("empty"):
                    probe_metric[(c, tgt, sub)] = {
                        "obs_max": m["obs_max"], "p_maxT": m["p_maxT"]}

    # ---- assemble + deltas ----
    base_acc = acc_by_cond["baseline"]
    summary = {}
    for c in COND_ORDER:
        entry = {
            "state_acc_all": acc_by_cond[c],
            "state_acc_all_delta_vs_base": acc_by_cond[c] - base_acc,
            "state_acc_all_ci95": boot[c],
            "state_acc_transition": acc_trans_by_cond[c],
            "probe": {},
        }
        for tgt in TARGETS:
            for sub in SUBSETS:
                key = f"{tgt}|{sub}"
                cur = probe_metric.get((c, tgt, sub))
                base = probe_metric.get(("baseline", tgt, sub))
                if cur is None:
                    continue
                e = {"obs_max": cur["obs_max"], "p_maxT": cur["p_maxT"]}
                if base is not None:
                    e["obs_max_delta_vs_base"] = cur["obs_max"] - base["obs_max"]
                entry["probe"][key] = e
        summary[c] = entry

    with open(out / "exp1_gateA_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    with open(out / "exp1_gateA.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "metric", "value"])
        for c in COND_ORDER:
            e = summary[c]
            w.writerow([c, "state_acc_all", f"{e['state_acc_all']:.4f}"])
            w.writerow([c, "state_acc_all_delta",
                        f"{e['state_acc_all_delta_vs_base']:+.4f}"])
            w.writerow([c, "state_acc_all_ci95",
                        f"[{e['state_acc_all_ci95'][0]:.4f},"
                        f"{e['state_acc_all_ci95'][1]:.4f}]"])
            w.writerow([c, "state_acc_transition",
                        f"{e['state_acc_transition']:.4f}"])
            for key, v in e["probe"].items():
                w.writerow([c, f"probe {key} obs_max", f"{v['obs_max']:.4f}"])
                w.writerow([c, f"probe {key} p_maxT", f"{v['p_maxT']:.4f}"])
                if "obs_max_delta_vs_base" in v:
                    w.writerow([c, f"probe {key} obs_max_delta",
                                f"{v['obs_max_delta_vs_base']:+.4f}"])
        fh.close()
    print(f"wrote {out}/exp1_gateA_summary.json and exp1_gateA.csv")
    # console summary
    print("\n=== Gate A: state behavior (all rows) ===")
    for c in COND_ORDER:
        e = summary[c]
        lo, hi = e["state_acc_all_ci95"]
        print(f"  {c:15s} acc={e['state_acc_all']:.3f} "
              f"delta={e['state_acc_all_delta_vs_base']:+.3f} "
              f"CI95=[{lo:.3f},{hi:.3f}]")
    print("\n=== Gate A: probe obs_max (best layer) + delta vs baseline ===")
    for tgt in TARGETS:
        print(f"  {tgt}:")
        for sub in SUBSETS:
            key = f"{tgt}|{sub}"
            line = f"    {sub:16s} "
            for c in COND_ORDER:
                v = summary[c]["probe"].get(key)
                if v:
                    d = v.get("obs_max_delta_vs_base", 0.0)
                    line += f"{c}={v['obs_max']:.3f}({d:+.3f}) "
            print(line)


if __name__ == "__main__":
    main()
