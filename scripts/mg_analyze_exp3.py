#!/usr/bin/env python3
"""Mechanism Gate Experiment 3 - Gate C analysis (CPU).

Question: does the encoded visual event causally affect the current-state
readout?

Robust metric (immune to the low-clean_P_cf entropy confound):
  net_shift = (P_patched(cf) - clean_P_cf) + (clean_P_st - P_patched(st))
  = probability mass that moves FROM the target's own state st TO the
    counterfactual state cf (defined by the source's event).
A positive net_shift for `main` (and ~0 for self/sameevent) means the
different-event window activations causally drive the state readout toward the
counterfactual.

Contrasts (apples-to-apples at the 8 control layers):
  main vs sameevent : does the effect require the event CONTENT to differ?
  main vs self      : is the patching a real effect (not a no-op)?
  main vs window_permute : does the effect require the event's structure?

Outputs: exp3_gateC_summary.json, exp3_gateC.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LAYER_CTRL = [0, 5, 11, 17, 22, 28, 33, 35]
CONDS = ["main", "self", "position", "sameevent", "window_permute"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/vetbench/mechanism_gate_v1")
    args = ap.parse_args()
    out = Path(args.out_dir)
    rows = list(csv.DictReader(open(out / "causal_exp3.csv", newline="")))
    for r in rows:
        r["P_cf"] = float(r["P_cf"])
        r["clean_P_cf"] = float(r["clean_P_cf"])
        r["clean_P_st"] = float(r["clean_P_st"])
        p_st = float(r[f"P_{r['st_state']}"])
        r["net_shift"] = (r["P_cf"] - r["clean_P_cf"]) + \
            (r["clean_P_st"] - p_st)

    # per (pair, condition): mean net_shift over its layers
    pc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pc[r["pair_id"]][r["condition"]].append(r["net_shift"])
    pair_cond = {pid: {c: np.mean(v) for c, v in conds.items()}
                 for pid, conds in pc.items()}
    pairs = list(pair_cond.keys())

    # per condition: mean over pairs (all layers)
    summary = {}
    for c in CONDS:
        vals = [pair_cond[p][c] for p in pairs if c in pair_cond[p]]
        summary[c] = {"mean_net_shift": float(np.mean(vals)),
                      "n_pairs": len(vals),
                      "mean_rescue_cf": None}
    # mean rescue_cf per condition (for reference)
    rc = defaultdict(list)
    for r in rows:
        rc[r["condition"]].append(r["P_cf"] - r["clean_P_cf"])
    for c in CONDS:
        if rc[c]:
            summary[c]["mean_rescue_cf"] = float(np.mean(rc[c]))

    # main per-layer (mean over pairs), all 36 layers
    main_by_layer = defaultdict(list)
    for r in rows:
        if r["condition"] == "main":
            main_by_layer[int(r["layer"])].append(r["net_shift"])
    main_layer = {d: float(np.mean(v)) for d, v in sorted(main_by_layer.items())}

    # contrasts at the 8 control layers (apples-to-apples)
    pc_ctrl = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if int(r["layer"]) in LAYER_CTRL:
            pc_ctrl[r["pair_id"]][r["condition"]].append(r["net_shift"])
    pair_ctrl = {pid: {c: np.mean(v) for c, v in conds.items()}
                 for pid, conds in pc_ctrl.items()}
    ctrl_pairs = list(pair_ctrl.keys())

    def contrast(a, b):
        d = [pair_ctrl[p][a] - pair_ctrl[p][b] for p in ctrl_pairs
             if a in pair_ctrl[p] and b in pair_ctrl[p]]
        d = np.array(d)
        return {"mean_diff": float(d.mean()),
                "frac_positive": float((d > 0).mean()),
                "n_pairs": len(d)}

    contrasts = {
        "main_vs_sameevent": contrast("main", "sameevent"),
        "main_vs_self": contrast("main", "self"),
        "main_vs_window_permute": contrast("main", "window_permute"),
        "main_vs_position": contrast("main", "position"),
    }

    result = {"per_condition": summary, "main_by_layer": main_layer,
              "contrasts": contrasts, "n_pairs": len(pairs)}
    with open(out / "exp3_gateC_summary.json", "w") as fh:
        json.dump(result, fh, indent=1)
    with open(out / "exp3_gateC.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "mean_net_shift", "mean_rescue_cf",
                    "n_pairs"])
        for c in CONDS:
            s = summary[c]
            w.writerow([c, f"{s['mean_net_shift']:+.4f}",
                        f"{s['mean_rescue_cf']:+.4f}" if s["mean_rescue_cf"]
                        is not None else "", s["n_pairs"]])
        w.writerow([])
        w.writerow(["contrast", "mean_diff", "frac_positive", "n_pairs"])
        for k, v in contrasts.items():
            w.writerow([k, f"{v['mean_diff']:+.4f}",
                        f"{v['frac_positive']:.2f}", v["n_pairs"]])
        fh.close()

    print(f"wrote {out}/exp3_gateC_summary.json and exp3_gateC.csv")
    print("\n=== Gate C: mean net_shift (mass st->cf) per condition ===")
    for c in CONDS:
        s = summary[c]
        print(f"  {c:15s} net_shift={s['mean_net_shift']:+.4f} "
              f"rescue_cf={s['mean_rescue_cf']:+.4f}")
    print("\n=== contrasts (at 8 control layers) ===")
    for k, v in contrasts.items():
        print(f"  {k:22s} mean_diff={v['mean_diff']:+.4f} "
              f"frac_pos={v['frac_positive']:.2f} n={v['n_pairs']}")
    print("\n=== main net_shift by layer ===")
    print("  " + " ".join(f"L{d}:{v:+.2f}" for d, v in
                          list(main_layer.items())[:36]))


if __name__ == "__main__":
    main()
