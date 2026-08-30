"""Strict v2 probe analysis for StateRev-VL (CPU-only; reuses hidden_states.npz).

Reuses the P0-validated hidden states (outputs/vetbench/hidden_state_probe/
hidden_states.npz: 250 rows x 37 layers x 4096; probe position = last input
token before the assistant turn; pre-generation single forward, verified
bit-exact) and re-runs the full probe protocol with stricter evaluation:

Families
  main    target in {state, event}. ONE probe per (seed, target, layer) trained
          on TRAIN trajectories only (StandardScaler fit on train; C by 3-fold
          group CV inside train), evaluated on held-out test trajectories for
          subsets: all / EC+SC / EC+SW / revision_success / revision_failure /
          revision_stale. Metrics: accuracy, balanced accuracy, macro-F1,
          majority baseline. Nulls: 100 full train-label permutations + 100
          within-step (same-t) permutations per (seed, target, layer); the SAME
          permutation sequence is used across all 37 layers (layer-independent
          rng key), which makes the per-permutation max over layers a valid
          max-statistic.
  cross   transition-direction cross-condition generalization (state target).
          A = {L->M, M->R, R->L}, B = {M->L, R->M, L->R} on GT true
          transitions. A_to_B: train on A-rows of TRAIN trajectories, test on
          B-rows of TEST trajectories (trajectory-disjoint). B_to_A: mirror.
          A_to_A / B_to_B: same-direction controls.
  scgen   state-correct -> stale-failure generalization (state target).
          v "sc": train on TRAIN-traj rows with behavioral state_correct;
          v "ecs": train on TRAIN-traj rows with event_correct AND
          state_correct. Test on held-out canonical stale failures and, as a
          reference, held-out revision successes.
  loo     leave-TRAJECTORY-out state probe for each of the 51 canonical rows
          (train on the other 49 trajectories): per-layer revision margin
          = decision(GT S_t) - decision(GT S_{t-1}) (logistic decision
          function at the probe position).

Multiple comparisons: per (family, layer) we report the one-layer p against
the pooled label-permutation null AND a max-statistic p:
  p_maxT(l) = P( max over 37 layers of null acc  >=  seed-mean observed acc(l) )
with the probability over pooled (seed, permutation) draws. One-layer
p < 0.05 is NOT treated as a significance claim.

No trajectory ever appears in both train and test of any fit. No behavioral
prompt is changed; no samples are filtered.

Run from the project root (CPU):
  python scripts/strict_probe_analysis.py [--n-jobs 24] [--n-perms 100]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR
from run_hidden_state_probe import (
    EVENT_INDEX,
    MANIFEST,
    OUT_DIR,
    STATE_CLASSES,
    STATE_INDEX,
    _pad_proba,
    group_split,
    read_manifest,
)

PROBE_DIR = OUT_DIR  # outputs/vetbench/hidden_state_probe
V2_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "probe_analysis_v2"


def stale_row(r: dict) -> bool:
    return bool(r["clean_revision"]) and not r["state_correct"] and r["state_pred"] == r["gt_prev_state"]


SUBSET2 = {
    "all": lambda r: True,
    "ec_sc": lambda r: r["joint_class"] == "event_correct_state_correct",
    "ec_sw": lambda r: r["joint_class"] == "event_correct_state_wrong",
    "rev_success": lambda r: bool(r["clean_revision"]) and r["state_correct"],
    "rev_failure": lambda r: bool(r["clean_revision"]) and not r["state_correct"],
    "rev_stale": stale_row,
}
SUBSETS2 = tuple(SUBSET2)
TRANS_A = {("Left", "Middle"), ("Middle", "Right"), ("Right", "Left")}
TRANS_B = {("Middle", "Left"), ("Right", "Middle"), ("Left", "Right")}


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    """Multi-class balanced accuracy (mean per-class recall); zero division -> 0."""
    recs = []
    for c in np.unique(y):
        sel = y == c
        recs.append(float((pred[sel] == c).mean()) if sel.any() else 0.0)
    return float(np.mean(recs)) if recs else 0.0


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score
    if len(y) == 0:
        return {"n": 0, "acc": float("nan"), "bal_acc": float("nan"),
                "macro_f1": float("nan")}
    return {
        "n": int(len(y)),
        "acc": float((pred == y).mean()),
        "bal_acc": _balanced_acc(y, pred),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


SOLVER = "newton-cg"  # ~4x faster than lbfgs on this data, same convex optimum
                      # (coef agreement to ~1e-5 absmean; documented in summary)


def _fit_lr2(X, y, C, max_iter=1000):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, solver=SOLVER, max_iter=max_iter).fit(X, y)


def _select_c2(X_tr, y_tr, trajs_tr, c_grid, rng_key):
    """Same 3-fold group-CV C selection as the original protocol, but with the
    newton-cg solver."""
    from sklearn.preprocessing import StandardScaler
    rng = random.Random(rng_key)
    order = list(sorted(set(trajs_tr)))
    rng.shuffle(order)
    n_folds = min(3, len(order))
    folds = [set(order[i * len(order) // n_folds:(i + 1) * len(order) // n_folds])
             for i in range(n_folds)]
    best_c, best_score = 1.0, -1.0
    for C in c_grid:
        scores = []
        for hold in folds:
            te = [j for j, tr in enumerate(trajs_tr) if tr in hold]
            tr = [j for j, tr in enumerate(trajs_tr) if tr not in hold]
            if not te:
                continue
            yh, yt = y_tr[te], y_tr[tr]
            if len(set(yh.tolist())) < 2 or len(set(yt.tolist())) < 2:
                continue
            sc = StandardScaler().fit(X_tr[tr])
            acc = float((_fit_lr2(sc.transform(X_tr[tr]), yt, C)
                         .predict(sc.transform(X_tr[te])) == yh).mean())
            scores.append(acc)
        if scores and float(np.mean(scores)) > best_score + 1e-12:
            best_score, best_c = float(np.mean(scores)), C
    return best_c


def _fit(X_tr, y_tr, X_te, c, rows_tr, c_grid, c_key):
    """StandardScaler (fit on train) + L2 LR. If c is None, C is selected by
    3-fold group CV inside the train trajectories. Returns (clf, Xtr_s, Xte_s, c)."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)
    c_use = _select_c2(Xtr_s, y_tr, rows_tr, c_grid, c_key) if c is None else c
    clf = _fit_lr2(Xtr_s, y_tr, c_use)
    return clf, Xtr_s, Xte_s, float(c_use)


def _worker(job: dict) -> dict:
    """One (family, seed) x layer probe job. Hidden states arrive as a
    zero-copy mmap file path."""
    from threadpoolctl import threadpool_limits

    rows, x_path = job["rows"], job["x_path"]
    layer, seed, kind, tag = job["layer"], job["seed"], job["kind"], job["tag"]
    y_all, c_grid, n_perms = job["y_all"], list(job["c_grid"]), job["n_perms"]
    X_all = np.load(x_path, mmap_mode="r")

    train_idx = [i for i in range(len(rows)) if job["train_mask"][i]]
    test_idx = [i for i in range(len(rows)) if job["test_mask"][i]]
    if not train_idx or not test_idx:
        return {"seed": seed, "layer": layer, "empty": True, "kind": kind, "tag": tag}
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    X_tr = np.asarray(X_all[train_idx, layer, :])
    X_te = np.asarray(X_all[test_idx, layer, :])

    out = {"seed": seed, "layer": layer, "empty": False, "kind": kind, "tag": tag,
           "train_n": len(train_idx), "test_n": len(test_idx), "null_modes": []}
    with threadpool_limits(limits=1):
        clf, Xtr_s, Xte_s, c_best = _fit(X_tr, y_tr, X_te, None,
                                         [rows[i]["trajectory_id"] for i in train_idx],
                                         c_grid, job["c_key"])
        out["c"] = c_best
        P = _pad_proba(clf.predict_proba(Xte_s), clf.classes_, 3)
        pred = P.argmax(axis=1)
        test_rows = [rows[i] for i in test_idx]
        if kind == "main":
            out["subsets"] = {}
            for name in SUBSETS2:
                m = np.array([SUBSET2[name](r) for r in test_rows])
                if not m.any():
                    continue
                mm = _metrics(pred[m], y_te[m])
                mm["majority"] = (Counter(y_te[m].tolist()).most_common(1)[0][1]
                                  / int(m.sum()))
                out["subsets"][name] = mm
        else:
            mm = _metrics(pred, y_te)
            mm["majority"] = (Counter(y_te.tolist()).most_common(1)[0][1]
                              / len(y_te))
            out["metrics"] = mm

        if kind == "loo":
            df = clf.decision_function(Xte_s)
            r0 = rows[test_idx[0]]
            cur, prev = STATE_INDEX[r0["gt_state"]], STATE_INDEX[r0["gt_prev_state"]]
            out["margin"] = float(df[0, cur] - df[0, prev])
            out["pred_class"] = STATE_CLASSES[int(pred[0])]
            out["prob_gt"] = float(P[0, cur])
            return out

        # label-permutation nulls; rng keyed WITHOUT the layer so every layer
        # of a (seed, family) sees the identical permutation sequence.
        modes = ("full", "within_t") if kind == "main" else ("full",)
        n_tr = len(y_tr)
        t_of_train = np.array([rows[i]["t"] for i in train_idx])
        for mode in modes:
            rng = random.Random(f"perm-{mode}-{job['null_key']}")
            nulls: dict[str, list[float]] = defaultdict(list)
            for _ in range(n_perms):
                if mode == "full":
                    y_perm = y_tr[rng.sample(range(n_tr), n_tr)]
                else:
                    y_perm = y_tr.copy()
                    for t in np.unique(t_of_train):
                        sel = np.flatnonzero(t_of_train == t)
                        y_perm[sel] = y_perm[sel][rng.sample(range(len(sel)), len(sel))]
                if len(set(y_perm.tolist())) < 2:
                    continue
                clf_n = _fit_lr2(Xtr_s, y_perm, c_best)
                pp = _pad_proba(clf_n.predict_proba(Xte_s), clf_n.classes_, 3).argmax(axis=1)
                if kind == "main":
                    for name in SUBSETS2:
                        m = np.array([SUBSET2[name](r) for r in test_rows])
                        if m.any():
                            nulls[name].append(float((pp[m] == y_te[m]).mean()))
                else:
                    nulls["main"].append(float((pp == y_te).mean()))
            out["null_modes"].append({"mode": mode,
                                      "nulls": {k: v for k, v in nulls.items() if v}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--split-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--c-grid", type=str, default="0.1,1.0,10.0")
    ap.add_argument("--out-dir", type=Path, default=V2_DIR)
    args = ap.parse_args()

    rows = read_manifest(MANIFEST)
    for r in rows:
        r["is_stale"] = stale_row(r)
    traj_ids = sorted({r["trajectory_id"] for r in rows})
    seeds = [int(x) for x in args.split_seeds.split(",")]
    splits = {s: group_split(traj_ids, s, args.train_frac) for s in seeds}
    c_grid = [float(x) for x in args.c_grid.split(",")]

    y_by_target = {
        "state": np.array([STATE_INDEX[r["gt_state"]] for r in rows]),
        "event": np.array([EVENT_INDEX[r["gt_event"]] for r in rows]),
    }

    can_rows = [r for r in rows if r["clean_revision"]]
    n_stale = sum(r["is_stale"] for r in can_rows)
    n_succ = sum(r["state_correct"] for r in can_rows)
    print(f"Canonical rows: {len(can_rows)} (stale {n_stale}, success {n_succ}, "
          f"other-failure {len(can_rows) - n_stale - n_succ})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(PROBE_DIR / "hidden_states.npz") as z:
        X_all = np.stack([z[f"{r['trajectory_id']}_t{r['t']}"] for r in rows],
                         axis=0).astype(np.float32)
    n, L1, D = X_all.shape
    assert n == 250 and L1 == 37, (n, L1)
    tmpdir = Path(tempfile.mkdtemp(prefix="v2probe_X_"))
    x_path = tmpdir / "X_all.npy"
    np.save(x_path, X_all)
    del X_all
    print(f"X_all {n} x {L1} x {D} staged at {x_path}")

    index = [f"{r['trajectory_id']}_t{r['t']}" for r in rows]
    traj_mask = {t: np.array([r["trajectory_id"] == t for r in rows]) for t in traj_ids}
    trans = np.array([bool(r["is_transition"]) for r in rows])
    pair = [(r["gt_prev_state"], r["gt_state"]) for r in rows]
    inA = np.array([p in TRANS_A for p in pair])
    inB = np.array([p in TRANS_B for p in pair])
    sc = np.array([r["state_correct"] for r in rows])
    ecs = np.array([r["event_correct"] and r["state_correct"] for r in rows])
    stale_m = np.array([r["is_stale"] for r in rows])
    succ_m = np.array([bool(r["clean_revision"]) and r["state_correct"] for r in rows])

    all_jobs: list[dict] = []
    for s in seeds:
        train_g, test_g = splits[s]
        train_mask = np.array([r["trajectory_id"] in train_g for r in rows])
        test_mask = np.array([r["trajectory_id"] in test_g for r in rows])
        for target in ("state", "event"):
            for layer in range(L1):
                all_jobs.append({
                    "kind": "main", "tag": f"main/{target}", "seed": s,
                    "layer": layer, "rows": rows, "index": index,
                    "x_path": str(x_path), "y_all": y_by_target[target],
                    "train_mask": train_mask, "test_mask": test_mask,
                    "c_grid": c_grid, "n_perms": args.n_perms,
                    "null_key": f"{s}-{target}", "c_key": f"{s}-{target}-{layer}",
                })
        for direction, tr_sel, te_sel in (
            ("A_to_B", inA, inB), ("B_to_A", inB, inA),
            ("A_to_A_ctrl", inA, inA), ("B_to_B_ctrl", inB, inB),
        ):
            tr_m = train_mask & trans & tr_sel
            te_m = test_mask & trans & te_sel
            if not tr_m.any() or not te_m.any():
                continue
            for layer in range(L1):
                all_jobs.append({
                    "kind": "cross", "tag": f"cross/{direction}", "seed": s,
                    "layer": layer, "rows": rows, "index": index,
                    "x_path": str(x_path), "y_all": y_by_target["state"],
                    "train_mask": tr_m, "test_mask": te_m,
                    "c_grid": c_grid, "n_perms": args.n_perms,
                    "null_key": f"{s}-cross-{direction}",
                    "c_key": f"{s}-cross-{direction}-{layer}",
                })
        for version, tr_sel in (("sc", sc), ("ecs", ecs)):
            for sub, te_sel in (("stale", stale_m), ("success", succ_m)):
                tr_m = train_mask & tr_sel
                te_m = test_mask & te_sel
                if not tr_m.any() or not te_m.any():
                    continue
                for layer in range(L1):
                    all_jobs.append({
                        "kind": "scgen", "tag": f"scgen/{version}/{sub}",
                        "seed": s, "layer": layer, "rows": rows, "index": index,
                        "x_path": str(x_path), "y_all": y_by_target["state"],
                        "train_mask": tr_m, "test_mask": te_m,
                        "c_grid": c_grid, "n_perms": args.n_perms,
                        "null_key": f"{s}-scgen-{version}-{sub}",
                        "c_key": f"{s}-scgen-{version}-{sub}-{layer}",
                    })
    for r_can in can_rows:
        te_m = np.zeros(len(rows), bool)
        te_m[rows.index(r_can)] = True
        tr_m = ~(traj_mask[r_can["trajectory_id"]] | te_m)
        for layer in range(L1):
            all_jobs.append({
                "kind": "loo", "tag": "loo/margin", "seed": -1, "layer": layer,
                "rows": rows, "index": index, "x_path": str(x_path),
                "y_all": y_by_target["state"], "train_mask": tr_m,
                "test_mask": te_m, "c_grid": c_grid, "n_perms": 0,
                "null_key": "loo", "c_key": f"loo-{r_can['trajectory_id']}-{layer}",
            })

    print(f"Total jobs: {len(all_jobs)}; workers: {args.n_jobs}; "
          f"perms/job: {args.n_perms}")
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        for i, res in enumerate(ex.map(_worker, all_jobs, chunksize=8)):
            results.append(res)
            if (i + 1) % 200 == 0:
                print(f"  jobs done: {i + 1}/{len(all_jobs)}", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    layer_name = lambda l: "embedding" if l == 0 else f"layer_{l:02d}"

    def aggregate(fam: str) -> list[dict]:
        """fam like 'main/state/all' | 'cross/A_to_B' | 'scgen/sc/stale'."""
        if fam.startswith("main"):
            fam_tag, sub = fam.rsplit("/", 1)[0], fam.rsplit("/", 1)[1]
        else:
            fam_tag, sub = fam, None
        sel = [r for r in results if not r.get("empty") and r["kind"] != "loo"
               and r["tag"] == fam_tag]
        out = []
        for layer in range(L1):
            ls = [r for r in sel if r["layer"] == layer]
            per_seed = {}
            for r in ls:
                m = r.get("subsets", {}).get(sub) if sub else r.get("metrics")
                if not m or m["n"] == 0:
                    continue
                per_seed[r["seed"]] = m
            if not per_seed:
                continue
            key = sub if sub else "main"
            _full = [np.asarray(d["nulls"][key]) for r in ls
                     for d in r["null_modes"]
                     if d["mode"] == "full" and d["nulls"].get(key)]
            _ws = [np.asarray(d["nulls"][key]) for r in ls
                   for d in r["null_modes"]
                   if d["mode"] == "within_t" and d["nulls"].get(key)]
            nulls_full = np.concatenate(_full) if _full else np.array([])
            nulls_ws = np.concatenate(_ws) if _ws else np.array([])
            accs = [m["acc"] for m in per_seed.values()]
            out.append({
                "family": fam, "layer": layer, "layer_name": layer_name(layer),
                "n_splits": len(accs),
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "bal_acc_mean": float(np.mean([m["bal_acc"] for m in per_seed.values()])),
                "macro_f1_mean": float(np.mean([m["macro_f1"] for m in per_seed.values()])),
                "majority_mean": float(np.mean([m["majority"] for m in per_seed.values()])),
                "null_mean": float(nulls_full.mean()) if len(nulls_full) else None,
                "null_p95": float(np.percentile(nulls_full, 95)) if len(nulls_full) else None,
                "p_one_layer": (float((nulls_full >= float(np.mean(accs))).mean())
                                if len(nulls_full) else None),
                "null_ws_mean": float(nulls_ws.mean()) if len(nulls_ws) else None,
                "null_ws_p95": float(np.percentile(nulls_ws, 95)) if len(nulls_ws) else None,
                "p_one_layer_ws": (float((nulls_ws >= float(np.mean(accs))).mean())
                                   if len(nulls_ws) else None),
                "per_seed_acc": {str(s2): m["acc"] for s2, m in per_seed.items()},
            })
        return out

    tables: dict[str, list[dict]] = {}
    for target in ("state", "event"):
        for sub in SUBSETS2:
            tables[f"main/{target}/{sub}"] = aggregate(f"main/{target}/{sub}")
    for direction in ("A_to_B", "B_to_A", "A_to_A_ctrl", "B_to_B_ctrl"):
        tables[f"cross/{direction}"] = aggregate(f"cross/{direction}")
    for version in ("sc", "ecs"):
        for sub in ("stale", "success"):
            tables[f"scgen/{version}/{sub}"] = aggregate(f"scgen/{version}/{sub}")

    # max-statistic p per family (pooled over seeds and permutations)
    for fam, tab in tables.items():
        if not tab:
            continue
        fam_tag = fam.rsplit("/", 1)[0] if fam.startswith("main") else fam
        key = fam.rsplit("/", 1)[1] if fam.startswith("main") else "main"
        sel = [r for r in results if not r.get("empty") and r["kind"] != "loo"
               and r["tag"] == fam_tag]
        layers = sorted({t["layer"] for t in tab})
        vecs: dict[tuple, dict[int, float]] = defaultdict(dict)
        for r in sel:
            for d in r["null_modes"]:
                if d["mode"] != "full":
                    continue
                vals = d["nulls"].get(key, [])
                if not vals:
                    continue
                for b, v in enumerate(vals):
                    vecs[(r["seed"], b)][r["layer"]] = v
        maxT = np.asarray([max(m.values()) for m in vecs.values()
                           if len(m) == len(layers)])
        dropped = len(vecs) - len(maxT)
        for t in tab:
            t["p_maxT"] = float((maxT >= t["acc_mean"]).mean()) if len(maxT) else None
        if dropped:
            print(f"maxT {fam}: dropped {dropped}/{len(vecs)} incomplete (seed,perm) vecs")

    # ---------------- LOO margin --------------------------------------------
    # jobs were appended as: for each canonical row, 37 consecutive layer jobs;
    # ex.map preserves order, so results line up the same way.
    loo_jobs = [j for j in all_jobs if j["kind"] == "loo"]
    loo_res = [x for x in results if x["kind"] == "loo"]
    assert len(loo_res) == len(loo_jobs) == len(can_rows) * L1, \
        (len(loo_res), len(loo_jobs), len(can_rows) * L1)
    margin_rows = []
    for ri, r_can in enumerate(can_rows):
        job_block = loo_jobs[ri * L1:(ri + 1) * L1]
        res_block = loo_res[ri * L1:(ri + 1) * L1]
        for j, res in zip(job_block, res_block):
            if res.get("empty"):
                continue
            te_i = int(np.flatnonzero(j["test_mask"])[0])
            margin_rows.append({
                "key": index[te_i], "trajectory_id": r_can["trajectory_id"],
                "t": r_can["t"],
                "group": ("stale" if r_can["is_stale"]
                          else "success" if r_can["state_correct"] else "other_failure"),
                "gt_prev": r_can["gt_prev_state"], "gt_cur": r_can["gt_state"],
                "state_pred": r_can["state_pred"],
                "layer": res["layer"], "layer_name": layer_name(res["layer"]),
                "margin": res["margin"], "pred_class": res["pred_class"],
                "prob_gt": res["prob_gt"],
            })
    with open(args.out_dir / "revision_margin_per_sample.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys()))
        w.writeheader()
        for m in margin_rows:
            w.writerow(m)

    groups = {g: [m for m in margin_rows if m["group"] == g] for g in
              ("stale", "success", "other_failure")}
    margin_summary: dict[str, Any] = {}
    rng = random.Random(12345)
    for g, ms in groups.items():
        by_layer = defaultdict(list)
        for m in ms:
            by_layer[m["layer"]].append(m["margin"])
        trajs_g = sorted({m["trajectory_id"] for m in ms})
        rows_of_traj: dict[str, list] = defaultdict(list)
        for m in ms:
            rows_of_traj[m["trajectory_id"]].append(m)
        per_layer = {}
        for layer in sorted(by_layer):
            vals = np.array(by_layer[layer])
            boot = []
            for _ in range(2000):
                sel_tr = [rng.choice(trajs_g) for _ in range(len(trajs_g))]
                boot.append(np.mean([m["margin"] for t in sel_tr
                                     for m in rows_of_traj[t]]))
            per_layer[layer_name(layer)] = {
                "n": int(len(vals)), "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "ci95_lo": float(np.percentile(boot, 2.5)),
                "ci95_hi": float(np.percentile(boot, 97.5)),
                "frac_margin_pos": float((vals > 0).mean()),
            }
        margin_summary[g] = {"n_rows": len(ms), "n_trajectories": len(trajs_g),
                             "per_layer": per_layer}
    st = np.array([m["margin"] for m in groups["stale"]])
    su = np.array([m["margin"] for m in groups["success"]])
    pool = np.concatenate([st, su])
    diff_obs = float(st.mean() - su.mean())
    rng2 = random.Random(777)
    cnt = 0
    for _ in range(10000):
        perm = pool[rng2.sample(range(len(pool)), len(pool))]
        if abs(perm[:len(st)].mean() - perm[len(st):].mean()) >= abs(diff_obs):
            cnt += 1
    margin_summary["stale_vs_success_permutation"] = {
        "statistic": "mean margin stale - mean margin success (pooled rows x layers)",
        "observed_diff": diff_obs, "p_two_sided": cnt / 10000,
        "n_perm": 10000, "n_stale_values": int(len(st)),
        "n_success_values": int(len(su)),
    }

    # ---------------- write outputs ------------------------------------------
    res_csv = []
    for fam, tab in tables.items():
        for t in tab:
            row = {
                "family": fam, "layer": t["layer"], "layer_name": t["layer_name"],
                "n_splits": t["n_splits"],
                "acc_mean": f'{t["acc_mean"]:.6f}', "acc_std": f'{t["acc_std"]:.6f}',
                "bal_acc_mean": f'{t["bal_acc_mean"]:.6f}',
                "macro_f1_mean": f'{t["macro_f1_mean"]:.6f}',
                "majority_mean": f'{t["majority_mean"]:.6f}',
                "null_mean": f'{t["null_mean"]:.6f}' if t["null_mean"] is not None else "",
                "null_p95": f'{t["null_p95"]:.6f}' if t["null_p95"] is not None else "",
                "p_one_layer": (f'{t["p_one_layer"]:.6f}'
                                if t["p_one_layer"] is not None else ""),
                "null_ws_mean": (f'{t["null_ws_mean"]:.6f}'
                                 if t["null_ws_mean"] is not None else ""),
                "null_ws_p95": (f'{t["null_ws_p95"]:.6f}'
                                if t["null_ws_p95"] is not None else ""),
                "p_one_layer_ws": (f'{t["p_one_layer_ws"]:.6f}'
                                   if t["p_one_layer_ws"] is not None else ""),
                "p_maxT": f'{t["p_maxT"]:.6f}' if t.get("p_maxT") is not None else "",
                "per_seed_acc": json.dumps(t["per_seed_acc"]),
            }
            res_csv.append(row)
    with open(args.out_dir / "strict_probe_results.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res_csv[0].keys()))
        w.writeheader()
        for r in res_csv:
            w.writerow(r)

    def layer_mean(fam: str) -> float | None:
        """Mean probe acc over the 36 text layers (embedding excluded)."""
        t = tables.get(fam)
        if not t:
            return None
        vals = [x["acc_mean"] for x in t
                if x["acc_mean"] is not None and int(x["layer"]) != 0]
        return float(np.mean(vals)) if vals else None

    summary = {
        "config": {
            "hidden_states": str(PROBE_DIR / "hidden_states.npz"),
            "probe_position": "last input token before the assistant turn "
                              "(P0-validated: pre-generation single forward, "
                              "bit-exact re-forward match)",
            "n_rows": 250, "n_layers": L1, "n_trajectories": len(traj_ids),
            "hidden_size": D, "seeds": seeds, "train_frac": args.train_frac,
            "c_grid": c_grid, "n_perms": args.n_perms,
            "solver": f"{SOLVER} (max_iter=1000); ~4x faster than the lbfgs "
                      f"used by the v1 run, same convex optimum "
                      f"(coef agreement to ~1e-5 absmean on this data); v1/v2 "
                      f"numbers are directly comparable",
            "group_split": "by trajectory_id; no trajectory in both train and "
                           "test of any fit",
            "subsets": {
                "all": "all held-out test rows",
                "ec_sc": "event_correct & state_correct",
                "ec_sw": "event_correct & state_wrong",
                "rev_success": "canonical revision with state updated correctly",
                "rev_failure": "canonical revision with state not updated",
                "rev_stale": "canonical revision failure with pred_t == "
                             "GT_{t-1} (the 31 canonical stale failures)",
            },
            "transitions": {"A": sorted(TRANS_A), "B": sorted(TRANS_B)},
            "nulls": "100 full train-label permutations + 100 within-step "
                     "(same-t) permutations per (seed, target, layer); C reused "
                     "from the observed fit; identical permutation sequence "
                     "across layers (layer-independent rng key)",
            "p_maxT": "max-statistic over the 37 layers: P( max over layers of "
                      "null acc >= seed-mean observed acc(layer) ), pooled over "
                      "(seed, permutation); family-wise control. One-layer "
                      "p < 0.05 is NOT a significance claim.",
            "loo_margin": "leave-trajectory-out state probe (train on the other "
                          "49 trajectories); margin = decision(GT S_t) - "
                          "decision(GT S_{t-1}) per layer; CI = 2000 "
                          "trajectory-unit bootstrap; stale vs success = 10000 "
                          "label-shuffle permutation test on pooled row x layer "
                          "margins",
        },
        "headline_layermean": {
            "state_all": layer_mean("main/state/all"),
            "event_all": layer_mean("main/event/all"),
            "old_reported_state_all": 0.497,
            "old_reported_event_all": 0.926,
        },
        "canonical_counts": {"total": len(can_rows), "stale": n_stale,
                             "success": n_succ,
                             "other_failure": len(can_rows) - n_stale - n_succ},
        "tables": {k: [{kk: vv for kk, vv in t.items()} for t in v]
                   for k, v in tables.items()},
        "revision_margin": margin_summary,
    }
    (args.out_dir / "strict_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\nWrote {args.out_dir}/strict_probe_results.csv ({len(res_csv)} rows), "
          f"strict_probe_summary.json, revision_margin_per_sample.csv "
          f"({len(margin_rows)} rows)")
    print("Headline layer-mean acc (layers 1-36): state/all =",
          f"{summary['headline_layermean']['state_all']:.3f}",
          ";  event/all =",
          f"{summary['headline_layermean']['event_all']:.3f}")
    for fam in ("main/state/all", "main/event/all", "cross/A_to_B", "cross/B_to_A",
                "cross/A_to_A_ctrl", "cross/B_to_B_ctrl",
                "scgen/sc/stale", "scgen/ecs/stale"):
        tab = tables.get(fam, [])
        if tab:
            sig = [t["layer_name"] for t in tab
                   if t.get("p_maxT") is not None and t["p_maxT"] < 0.05]
            print(f"{fam}: layers with p_maxT<0.05: {sig if sig else 'none'}")


if __name__ == "__main__":
    main()
