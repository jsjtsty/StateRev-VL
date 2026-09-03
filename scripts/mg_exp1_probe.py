#!/usr/bin/env python3
"""Mechanism Gate Experiment 1 - probe (offline, CPU).

For each temporal condition (baseline / freeze / shuffle / history_freeze),
run the same-input linear probe protocol (10-split by trajectory, 3 targets
event/state/prev_state, subsets all/true_transition, 37 layers, maxT=200
within-t permutation) on hidden_exp1_{cond}.npz.

Labels/splits come from the 250-row baseline behavior (GT labels are
condition-invariant). This is the core of Gate A: does the event/state code
survive removal of current-swap visual dynamics?
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vg_same_input_probe import (
    build_index, load_hidden, job_point, job_maxt,
    _job_point_wrap, _job_maxt_wrap,
    N_LAYERS, TARGETS, SUBSETS,
)

COND_ORDER = ["baseline", "freeze", "shuffle", "history_freeze"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/vetbench/mechanism_gate_v1")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-maxt", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(Path(args.behavior_csv))

    from joblib import Parallel, delayed
    results = {}
    for cond in COND_ORDER:
        npz = out / f"hidden_exp1_{cond}.npz"
        if not npz.exists():
            print(f"[{cond}] {npz} missing - skip")
            continue
        H_path = out / f"hidden_exp1_{cond}.npy"
        if not H_path.exists():
            arr = load_hidden(npz, idx)
            np.save(H_path, arr)
            print(f"[{cond}] saved {H_path} {arr.shape}", flush=True)
        args_pt = [(sk, tgt, sub, L)
                   for sk in range(args.n_splits) for tgt in TARGETS
                   for sub in SUBSETS for L in range(N_LAYERS)]
        print(f"[{cond}] point: {len(args_pt)} jobs", flush=True)
        res = Parallel(n_jobs=args.n_proc, backend="loky", batch_size=8)(
            delayed(_job_point_wrap)(idx, str(H_path), a[0], a[1], a[2],
                                     a[3], args.n_splits)
            for a in args_pt)
        pt = {}
        for a, r in zip(args_pt, res):
            pt.setdefault((a[1], a[2], a[3]), []).append(r)
        point = {}
        for (tgt, sub, L), vals in pt.items():
            ok = [v for v in vals if not v.get("empty")]
            if ok:
                point[f"{tgt}|{sub}|L{L:02d}"] = {
                    "bal_acc": float(np.mean([v["bal_acc"] for v in ok])),
                    "macro_f1": float(np.mean([v["macro_f1"] for v in ok])),
                    "acc": float(np.mean([v["acc"] for v in ok])),
                    "majority": float(np.mean([v["majority"] for v in ok])),
                    "n_splits": len(ok),
                }
        args_mt = [(tgt, sub) for tgt in TARGETS for sub in SUBSETS]
        print(f"[{cond}] maxT: {len(args_mt)} jobs", flush=True)
        res_mt = Parallel(n_jobs=max(1, args.n_proc // 2), backend="loky")(
            delayed(_job_maxt_wrap)(idx, str(H_path), a[0], a[1],
                                    args.n_maxt, args.n_splits)
            for a in args_mt)
        maxt = {f"{a[0]}|{a[1]}": r for a, r in zip(args_mt, res_mt)}
        results[cond] = {"point": point, "maxt": maxt}
        with open(out / f"exp1_probe_{cond}.json", "w") as fh:
            json.dump(results[cond], fh, indent=1)
        print(f"[{cond}] wrote exp1_probe_{cond}.json", flush=True)

    with open(out / "exp1_probe_summary.json", "w") as fh:
        json.dump(results, fh, indent=1)
    # CSV
    with open(out / "exp1_probe_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "target", "subset", "layer", "metric",
                    "value"])
        for cond, d in results.items():
            for key, v in d["point"].items():
                tgt, sub, L = key.split("|")
                for m in ("bal_acc", "macro_f1", "acc", "majority"):
                    w.writerow([cond, tgt, sub, L, m, f"{v[m]:.4f}"])
            for key, v in d["maxt"].items():
                tgt, sub = key.split("|")
                if not v.get("empty"):
                    w.writerow([cond, tgt, sub, "maxT", "obs_max",
                                f"{v['obs_max']:.4f}"])
                    w.writerow([cond, tgt, sub, "maxT", "p_maxT",
                                f"{v['p_maxT']:.4f}"])
        fh.close()
    print(f"wrote {out}/exp1_probe_summary.json and results.csv")


if __name__ == "__main__":
    main()
