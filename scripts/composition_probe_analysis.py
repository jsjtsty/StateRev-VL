"""Composition analysis: previous-state representation, symbolic composition,
route generalization, and canonical stale-failure operand evidence.

Uses the P0-validated hidden states (outputs/vetbench/hidden_state_probe/
hidden_states.npz) and the Transformers-aligned behavior manifest
(outputs/vetbench/composition_analysis_v1/transformers_behavior.csv).

Job kinds (all group CV by trajectory_id; L2 logistic regression, newton-cg):

  main    h_t -> GT S_t (state) and h_t -> GT S_{t-1} (prev_state);
          10 seeds x group split (40/10 trajs) x 37 layers;
          C by 3-fold group CV inside train; 100 full + 100 within-t
          label-permutation nulls; family-wise p_maxT.
  cross   direction families A/B (same as probe_analysis_v2), both targets,
          A_to_B / B_to_A + same-direction controls.
  route   leave-one-directed-route-out: 6 routes (L->M, L->R, M->L, M->R,
          R->L, R->M); train on the other 5 routes' transition rows, test on
          the held-out route. Both targets. 100 full nulls.
  loo     per trajectory x layer: state / prev_state / event decoders fit on
          the other 49 trajectories (10 CV seeds for C, averaged predict
          probabilities); feeds canonical operand evidence AND symbolic
          composition (strictly out-of-fold / held-out trajectory).

Outputs in outputs/vetbench/composition_analysis_v1/:
  previous_state_probe.csv / _summary.json
  route_generalization.csv
  symbolic_composition.csv / _summary.json
  canonical_operand_evidence.csv
  layer_comparison.png (if matplotlib available)

Usage (from project root):
  python scripts/composition_probe_analysis.py [--n-jobs 24] [--n-perms 100]
  smoke: python scripts/composition_probe_analysis.py --n-jobs 8 --n-perms 5 \
      --seeds 0,1 --max-loo-trajs 3 --out-dir /tmp/opencode/compo_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
import time
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
from strict_probe_analysis import SOLVER, _fit_lr2, _select_c2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "composition_analysis_v1"
PROBE_DIR = OUT_DIR  # hidden_state_probe

POS = {"Left": 0, "Middle": 1, "Right": 2}
POS_NAME = ["Left", "Middle", "Right"]
PAIR = {"Left and Middle": (0, 1), "Middle and Right": (1, 2),
        "Left and Right": (0, 2)}
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
ROUTES = [("Left", "Middle"), ("Left", "Right"), ("Middle", "Left"),
          ("Middle", "Right"), ("Right", "Left"), ("Right", "Middle")]
TRANS_A = {("Left", "Middle"), ("Middle", "Right"), ("Right", "Left")}
TRANS_B = {("Middle", "Left"), ("Right", "Middle"), ("Left", "Right")}
TARGETS = ("state", "prev_state")
SUBSETS = ("all", "ec_sc", "ec_sw", "rev_success", "rev_failure", "rev_stale")


def apply_swap(pos: int, pair: tuple[int, int]) -> int:
    a, b = pair
    if pos == a:
        return b
    if pos == b:
        return a
    return pos


def subset_mask(r: dict, name: str) -> bool:
    if name == "all":
        return True
    if name == "ec_sc":
        return r["joint_class"] == "event_correct_state_correct"
    if name == "ec_sw":
        return r["joint_class"] == "event_correct_state_wrong"
    if name == "rev_success":
        return bool(r["clean_revision"]) and r["state_correct"]
    if name == "rev_failure":
        return bool(r["clean_revision"]) and not r["state_correct"]
    if name == "rev_stale":
        return (bool(r["clean_revision"]) and not r["state_correct"]
                and r["state_pred"] == r["gt_prev_state"])
    raise KeyError(name)


def is_stale(r: dict) -> bool:
    return subset_mask(r, "rev_stale")


def balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    recs = []
    for c in np.unique(y):
        sel = y == c
        recs.append(float((pred[sel] == c).mean()) if sel.any() else 0.0)
    return float(np.mean(recs)) if recs else 0.0


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    fs = []
    for c in range(k):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))


# ---------------------------------------------------------------------------
# Workers (forked pool; X shared through an mmap file)
# ---------------------------------------------------------------------------

def _job_main(job: dict) -> dict:
    X_all = np.load(job["x_path"], mmap_mode="r")
    L = job["layer"]
    X_tr_raw = np.asarray(X_all[job["train_rows"], L, :], dtype=np.float32)
    X_te_raw = np.asarray(X_all[job["test_rows"], L, :], dtype=np.float32)
    y_tr = job["y_all"][job["train_rows"]]
    y_te = job["y_all"][job["test_rows"]]
    trajs_tr = [job["traj_of"][i] for i in job["train_rows"]]
    k = int(job["k_classes"])

    from sklearn.preprocessing import StandardScaler
    clf, Xtr_s, Xte_s, c_best = _fit_main(X_tr_raw, y_tr, X_te_raw,
                                          trajs_tr, job["c_grid"], job["c_key"])
    pred_te = clf.predict(Xte_s)
    proba_te = _pad_proba(clf.predict_proba(Xte_s), clf.classes_, k)
    out: dict[str, Any] = {
        "kind": job["kind"], "tag": job["tag"], "target": job["target"],
        "seed": job["seed"], "layer": job["layer"], "C": c_best,
        "n_train": len(y_tr), "n_test": len(y_te),
        "test_index": job["index"][job["test_rows"]],
        "proba_te": proba_te.tolist(),
        "y_te": y_te.tolist(),
    }
    # subset metrics (post-hoc filter of test rows)
    for name in SUBSETS:
        m = np.array([subset_mask(job["rows"][i], name) for i in job["test_rows"]])
        if m.sum() < 2:
            out[f"acc_{name}"] = None
            continue
        yt, pt = y_te[m], pred_te[m]
        out[f"acc_{name}"] = float((pt == yt).mean())
        out[f"bal_{name}"] = balanced_acc(yt, pt)
        out[f"f1_{name}"] = macro_f1(yt, pt, k)
        out[f"maj_{name}"] = float(np.bincount(yt, minlength=k).max() / len(yt))
    # nulls
    rng = random.Random(job["rng_key"])
    null_full: dict[str, list[float]] = defaultdict(list)
    null_ws: dict[str, list[float]] = defaultdict(list)
    t_of_train = np.array([job["t_of"][i] for i in job["train_rows"]])
    for b in range(job["n_perms"]):
        y_perm = y_tr.copy()
        y_perm[rng.sample(range(len(y_perm)), len(y_perm))] = y_perm
        if len(set(y_perm.tolist())) < 2:
            continue
        clf_n = _fit_lr2(Xtr_s, y_perm, c_best)
        pn = clf_n.predict(Xte_s)
        for name in SUBSETS:
            m = np.array([subset_mask(job["rows"][i], name) for i in job["test_rows"]])
            if m.sum() < 2:
                continue
            null_full[name].append(float((pn[m] == y_te[m]).mean()))
        y_perm_ws = y_tr.copy()
        for t in np.unique(t_of_train):
            sel = np.flatnonzero(t_of_train == t)
            y_perm_ws[sel] = y_perm_ws[sel][rng.sample(range(len(sel)), len(sel))]
        if len(set(y_perm_ws.tolist())) < 2:
            continue
        clf_w = _fit_lr2(Xtr_s, y_perm_ws, c_best)
        pw = clf_w.predict(Xte_s)
        for name in SUBSETS:
            m = np.array([subset_mask(job["rows"][i], name) for i in job["test_rows"]])
            if m.sum() < 2:
                continue
            null_ws[name].append(float((pw[m] == y_te[m]).mean()))
    out["null_full"] = {k2: v for k2, v in null_full.items()}
    out["null_ws"] = {k2: v for k2, v in null_ws.items()}
    return out


def _fit_main(X_tr_raw, y_tr, X_te_raw, trajs_tr, c_grid, c_key):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr_raw)
    Xtr_s, Xte_s = sc.transform(X_tr_raw), sc.transform(X_te_raw)
    c_use = _select_c2(Xtr_s, y_tr, trajs_tr, c_grid, c_key)
    clf = _fit_lr2(Xtr_s, y_tr, c_use)
    return clf, Xtr_s, Xte_s, float(c_use)


def _job_cross(job: dict) -> dict:
    X_all = np.load(job["x_path"], mmap_mode="r")
    tr = np.flatnonzero(job["train_mask"])
    te = np.flatnonzero(job["test_mask"])
    L = job["layer"]
    X_tr_raw = np.asarray(X_all[tr, L, :], dtype=np.float32)
    X_te_raw = np.asarray(X_all[te, L, :], dtype=np.float32)
    y_tr = job["y_all"][tr]
    y_te = job["y_all"][te]
    trajs_tr = [job["traj_of"][i] for i in tr]
    k = int(job["k_classes"])
    from sklearn.preprocessing import StandardScaler
    clf, Xtr_s, Xte_s, c_best = _fit_main(X_tr_raw, y_tr, X_te_raw,
                                          trajs_tr, job["c_grid"], job["c_key"])
    pred_te = clf.predict(Xte_s)
    out: dict[str, Any] = {
        "kind": job["kind"], "tag": job["tag"], "target": job["target"],
        "direction": job["direction"], "seed": job["seed"], "layer": job["layer"],
        "C": c_best, "n_train": len(y_tr), "n_test": len(y_te),
        "acc_all": float((pred_te == y_te).mean()),
        "bal_all": balanced_acc(y_te, pred_te),
        "f1_all": macro_f1(y_te, pred_te, k),
        "maj_all": float(np.bincount(y_te, minlength=k).max() / len(y_te)),
    }
    rng = random.Random(job["rng_key"])
    nulls = []
    for b in range(job["n_perms"]):
        y_perm = y_tr.copy()
        y_perm[rng.sample(range(len(y_perm)), len(y_perm))] = y_perm
        if len(set(y_perm.tolist())) < 2:
            continue
        pn = _fit_lr2(Xtr_s, y_perm, c_best).predict(Xte_s)
        nulls.append(float((pn == y_te).mean()))
    out["null_full"] = nulls
    return out


def _job_route(job: dict) -> dict:
    X_all = np.load(job["x_path"], mmap_mode="r")
    tr = np.flatnonzero(job["train_mask"])
    te = np.flatnonzero(job["test_mask"])
    L = job["layer"]
    X_tr_raw = np.asarray(X_all[tr, L, :], dtype=np.float32)
    X_te_raw = np.asarray(X_all[te, L, :], dtype=np.float32)
    y_tr = job["y_all"][tr]
    y_te = job["y_all"][te]
    trajs_tr = [job["traj_of"][i] for i in tr]
    k = int(job["k_classes"])
    from sklearn.preprocessing import StandardScaler
    clf, Xtr_s, Xte_s, c_best = _fit_main(X_tr_raw, y_tr, X_te_raw,
                                          trajs_tr, job["c_grid"], job["c_key"])
    pred_te = clf.predict(Xte_s)
    out: dict[str, Any] = {
        "kind": job["kind"], "tag": job["tag"], "target": job["target"],
        "route": job["route"], "seed": job["seed"], "layer": job["layer"],
        "C": c_best, "n_train": len(y_tr), "n_test": len(y_te),
        "acc_all": float((pred_te == y_te).mean()),
        "bal_all": balanced_acc(y_te, pred_te),
        "f1_all": macro_f1(y_te, pred_te, k),
        "maj_all": float(np.bincount(y_te, minlength=k).max() / len(y_te)),
    }
    rng = random.Random(job["rng_key"])
    nulls = []
    for b in range(job["n_perms"]):
        y_perm = y_tr.copy()
        y_perm[rng.sample(range(len(y_perm)), len(y_perm))] = y_perm
        if len(set(y_perm.tolist())) < 2:
            continue
        pn = _fit_lr2(Xtr_s, y_perm, c_best).predict(Xte_s)
        nulls.append(float((pn == y_te).mean()))
    out["null_full"] = nulls
    return out


def _job_loo(job: dict) -> dict:
    """LOO-trajectory decoders for one (trajectory, layer): 3 targets x 10
    CV seeds; averaged predict probabilities on the held-out trajectory's 5
    rows. No nulls here (composition/evidence significance done separately).
    """
    X_all = np.load(job["x_path"], mmap_mode="r")
    tr = np.flatnonzero(job["train_mask"])
    te = np.flatnonzero(job["test_mask"])
    trajs_tr = [job["traj_of"][i] for i in tr]
    out: dict[str, Any] = {
        "kind": job["kind"], "tag": job["tag"], "traj": job["traj"],
        "layer": job["layer"], "test_index": job["index"][te],
    }
    from sklearn.preprocessing import StandardScaler
    L = job["layer"]
    Xtr_raw = np.asarray(X_all[tr, L, :], dtype=np.float32)
    Xte_raw = np.asarray(X_all[te, L, :], dtype=np.float32)
    sc = StandardScaler().fit(Xtr_raw)
    Xtr_s = sc.transform(Xtr_raw)
    Xte_s = sc.transform(Xte_raw)
    for target in ("state", "prev_state", "event"):
        y_tr = job["y_all"][target][tr]
        k = 3
        probs = np.zeros((len(te), k))
        cs = []
        for s in range(job["n_seeds"]):
            c_key = f"{job['c_key']}-{s}"
            c_use = _select_c2(Xtr_s, y_tr, trajs_tr, job["c_grid"], c_key)
            cs.append(c_use)
            clf = _fit_lr2(Xtr_s, y_tr, c_use)
            probs += _pad_proba(clf.predict_proba(Xte_s), clf.classes_, k)
        out[f"proba_{target}"] = (probs / job["n_seeds"]).tolist()
        out[f"Cs_{target}"] = cs
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _pval(obs, nulls) -> float:
    if not nulls:
        return None
    arr = np.asarray(nulls)
    return float((arr >= obs).mean())


def aggregate_main(results: list[dict], tag: str, target: str) -> list[dict]:
    """Per (layer, subset): seed-mean acc + one-layer p + pooled maxT."""
    L1 = 37
    layers = list(range(L1))
    rows_out = []
    per_seed: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["tag"] != tag:
            continue
        for sub in SUBSETS:
            a = r.get(f"acc_{sub}")
            if a is not None:
                per_seed[r["seed"]][str(r["layer"])][sub] = a
    # maxT vectors
    vecs: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for r in results:
        if r["tag"] != tag:
            continue
        for b, v in enumerate(r.get("null_full", {})):
            pass
        for sub, vals in r.get("null_full", {}).items():
            for b, v in enumerate(vals):
                vecs[(r["seed"], b)][ (r["layer"], "full", sub) ] = v
        for sub, vals in r.get("null_ws", {}).items():
            for b, v in enumerate(vals):
                vecs[(r["seed"], b)][ (r["layer"], "ws", sub) ] = v
    for layer in layers:
        for sub in SUBSETS:
            ps = [per_seed[s][str(layer)][sub] for s in sorted(per_seed)
                  if sub in per_seed[s].get(str(layer), {})]
            if not ps:
                continue
            acc_mean = float(np.mean(ps))
            nulls_full = [m[(layer, "full", sub)] for (s, b), m in vecs.items()
                          if (layer, "full", sub) in m]
            nulls_ws = [m[(layer, "ws", sub)] for (s, b), m in vecs.items()
                        if (layer, "ws", sub) in m]
            rows_out.append({
                "tag": tag, "target": target, "layer": layer,
                "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                "subset": sub, "n_splits": len(ps),
                "acc_mean": acc_mean,
                "acc_std": float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0,
                "p_one_layer_full": _pval(acc_mean, nulls_full),
                "p_one_layer_ws": _pval(acc_mean, nulls_ws),
                "null_mean": float(np.mean(nulls_full)) if nulls_full else None,
                "null_p95": float(np.percentile(nulls_full, 95)) if nulls_full else None,
                "_nulls_full": nulls_full, "_nulls_ws": nulls_ws,
            })
    # family-wise maxT per subset: P(max over layers of null acc >= acc_mean_l)
    for sub in SUBSETS:
        accs = {row["layer"]: row["acc_mean"] for row in rows_out
                if row["subset"] == sub}
        for row in rows_out:
            if row["subset"] != sub:
                continue
            for key in ("full", "ws"):
                maxts = []
                for (s, b), m in vecs.items():
                    vals = [m[(l, key, sub)] for l in accs if (l, key, sub) in m]
                    if len(vals) == len(accs):
                        maxts.append(max(vals))
                row[f"p_maxT_{key}"] = (float(np.mean(np.asarray(maxts)
                                                       >= row["acc_mean"]))
                                        if maxts else None)
    return rows_out


def aggregate_single(results: list[dict], tag: str, extra: dict) -> list[dict]:
    """cross / route: per layer: seed-mean acc + one-layer p + pooled maxT."""
    per_seed: dict[int, dict[int, float]] = defaultdict(dict)
    vecs: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for r in results:
        if r["tag"] != tag:
            continue
        per_seed[r["seed"]][r["layer"]] = r["acc_all"]
        for b, v in enumerate(r.get("null_full", [])):
            vecs[(r["seed"], b)][r["layer"]] = v
    out = []
    all_layers = sorted({l for m in per_seed.values() for l in m})
    n_layers = len(all_layers)
    for layer in all_layers:
        ps = [per_seed[s][layer] for s in sorted(per_seed) if layer in per_seed[s]]
        if not ps:
            continue
        acc_mean = float(np.mean(ps))
        maxts = [max(m.values()) for m in vecs.values() if len(m) == n_layers]
        per_layer_nulls = [m[layer] for m in vecs.values() if layer in m]
        out.append({**extra, "layer": layer,
                    "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                    "n_splits": len(ps), "acc_mean": acc_mean,
                    "p_one_layer": _pval(acc_mean, per_layer_nulls),
                    "p_maxT": (float(np.mean(np.asarray(maxts) >= acc_mean))
                               if maxts else None),
                    "null_mean": (float(np.mean(per_layer_nulls))
                                  if per_layer_nulls else None),
                    "null_p95": (float(np.percentile(per_layer_nulls, 95))
                                 if per_layer_nulls else None),
                    "maj_mean": None,
                    })
    return out


def boot_ci(values_by_traj: dict[str, list], n_boot: int = 2000,
            seed: int = 0) -> tuple[float, float]:
    """Trajectory-unit bootstrap CI: resample trajectories with replacement,
    pool their per-row values, take the mean."""
    trajs = sorted(values_by_traj)
    arrays = [np.asarray(values_by_traj[t], dtype=np.float64) for t in trajs]
    total = sum(len(a) for a in arrays)
    if total == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, len(trajs), size=len(trajs))
        pooled = np.concatenate([arrays[j] for j in sel])
        means[i] = pooled.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=V1_DIR)
    ap.add_argument("--behavior-csv", type=Path,
                    default=V1_DIR / "transformers_behavior.csv")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--c-grid", default="0.1,1,10")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--loo-seeds", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--cache-pkl", type=Path, default=None,
                    help="pickle of raw job results; if set and exists, jobs are "
                         "skipped and results are loaded from it; otherwise "
                         "results are saved there after the pool finishes")
    ap.add_argument("--max-loo-trajs", type=int, default=0)
    ap.add_argument("--skip", default="",
                    help="comma list of: main,cross,route,loo")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    seeds = [int(x) for x in args.seeds.split(",")]
    c_grid = [float(x) for x in args.c_grid.split(",")]

    rows = read_manifest(args.behavior_csv)
    assert len(rows) == 250, len(rows)
    old_rows = read_manifest(MANIFEST)
    assert [f"{r['trajectory_id']}_t{r['t']}" for r in rows] == \
           [f"{r['trajectory_id']}_t{r['t']}" for r in old_rows], \
        "row order mismatch between aligned and old manifest"
    traj_ids = list(dict.fromkeys(r["trajectory_id"] for r in rows))
    assert len(traj_ids) == 50
    index = np.array([f"{r['trajectory_id']}_t{r['t']}" for r in rows])
    traj_of = np.array([r["trajectory_id"] for r in rows])
    t_of = np.array([int(r["t"]) for r in rows])
    y_by_target = {
        "state": np.array([STATE_INDEX[r["gt_state"]] for r in rows]),
        "prev_state": np.array([STATE_INDEX[r["gt_prev_state"]] for r in rows]),
        "event": np.array([EVENT_INDEX[r["gt_event"]] for r in rows]),
    }
    trans = np.array([bool(r["is_transition"]) for r in rows])
    pair = [(r["gt_prev_state"], r["gt_state"]) for r in rows]
    inA = np.array([p in TRANS_A for p in pair])
    inB = np.array([p in TRANS_B for p in pair])
    route_of = np.array([None if not trans[i] else f"{pair[i][0]}->{pair[i][1]}"
                         for i in range(len(rows))])
    route_names = {f"{a}->{b}": (a, b) for a, b in ROUTES}
    route_masks = {rt: np.array([route_of[i] == rt for i in range(len(rows))])
                   for rt in route_names}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(PROBE_DIR / "hidden_states.npz") as z:
        X_all = np.stack([z[k] for k in index], axis=0).astype(np.float32)
    n, L1, D = X_all.shape
    assert (n, L1, D) == (250, 37, 4096)
    tmpdir = Path(tempfile.mkdtemp(prefix="compo_X_"))
    x_path = tmpdir / "X_all.npy"
    np.save(x_path, X_all)
    del X_all
    print(f"X_all {n} x {L1} x {D} staged at {x_path}", flush=True)

    splits = {s: group_split(traj_ids, s, args.train_frac) for s in seeds}
    all_jobs: list[dict] = []
    if "main" not in skip:
        for s in seeds:
            train_g, test_g = splits[s]
            train_mask = np.array([r["trajectory_id"] in train_g for r in rows])
            test_mask = np.array([r["trajectory_id"] in test_g for r in rows])
            for target in TARGETS:
                for layer in range(L1):
                    all_jobs.append({
                        "kind": "main", "tag": f"main/{target}", "target": target,
                        "seed": s, "layer": layer, "rows": rows, "index": index,
                        "traj_of": traj_of, "t_of": t_of, "x_path": str(x_path),
                        "y_all": y_by_target[target],
                        "train_rows": np.flatnonzero(train_mask).tolist(),
                        "test_rows": np.flatnonzero(test_mask).tolist(),
                        "c_grid": c_grid, "n_perms": args.n_perms,
                        "k_classes": 3,
                        "rng_key": f"{s}-{target}-main",
                        "c_key": f"{s}-{target}-main-{layer}",
                    })
    if "cross" not in skip:
        for s in seeds:
            train_g, test_g = splits[s]
            train_mask = np.array([r["trajectory_id"] in train_g for r in rows])
            test_mask = np.array([r["trajectory_id"] in test_g for r in rows])
            for target in TARGETS:
                for direction, tr_sel, te_sel in (
                    ("A_to_B", inA, inB), ("B_to_A", inB, inA),
                    ("A_to_A_ctrl", inA, inA), ("B_to_B_ctrl", inB, inB)):
                    tr_m = train_mask & trans & tr_sel
                    te_m = test_mask & trans & te_sel
                    if not tr_m.any() or not te_m.any():
                        continue
                    for layer in range(L1):
                        all_jobs.append({
                            "kind": "cross", "tag": f"cross/{target}/{direction}",
                            "target": target, "direction": direction, "seed": s,
                            "layer": layer, "rows": rows, "index": index,
                            "traj_of": traj_of, "t_of": t_of, "x_path": str(x_path),
                            "y_all": y_by_target[target],
                            "train_mask": tr_m, "test_mask": te_m,
                            "c_grid": c_grid, "n_perms": args.n_perms,
                            "k_classes": 3,
                            "rng_key": f"{s}-cross-{target}-{direction}",
                            "c_key": f"{s}-cross-{target}-{direction}-{layer}",
                        })
    if "route" not in skip:
        for target in TARGETS:
            for rt in route_names:
                te_m = route_masks[rt]
                tr_m = trans & ~te_m
                for s in seeds:  # seeds drive only the CV rng for C selection
                    for layer in range(L1):
                        all_jobs.append({
                            "kind": "route", "tag": f"route/{target}/{rt}",
                            "target": target, "route": rt, "seed": s,
                            "layer": layer, "rows": rows, "index": index,
                            "traj_of": traj_of, "t_of": t_of, "x_path": str(x_path),
                            "y_all": y_by_target[target],
                            "train_mask": tr_m, "test_mask": te_m,
                            "c_grid": c_grid, "n_perms": args.n_perms,
                            "k_classes": 3,
                            "rng_key": f"{s}-route-{target}-{rt}",
                            "c_key": f"{s}-route-{target}-{rt}-{layer}",
                        })
    loo_trajs = traj_ids
    if args.max_loo_trajs:
        loo_trajs = traj_ids[: args.max_loo_trajs]
    if "loo" not in skip:
        for tr in loo_trajs:
            te_m = np.array([r["trajectory_id"] == tr for r in rows])
            tr_m = ~te_m
            for layer in range(L1):
                all_jobs.append({
                    "kind": "loo", "tag": "loo", "traj": tr, "layer": layer,
                    "rows": rows, "index": index, "traj_of": traj_of,
                    "x_path": str(x_path), "y_all": y_by_target,
                    "train_mask": tr_m, "test_mask": te_m,
                    "c_grid": c_grid, "n_seeds": args.loo_seeds,
                    "c_key": f"loo-{tr}-{layer}",
                })

    n_loo = sum(1 for j in all_jobs if j["kind"] == "loo")
    print(f"Total jobs: {len(all_jobs)} (loo {n_loo}); workers: {args.n_jobs}; "
          f"perms/job: {args.n_perms}", flush=True)

    import pickle
    results: list[dict] = []
    cache = args.cache_pkl
    if cache is not None and cache.exists():
        results = pickle.loads(cache.read_bytes())
        print(f"Loaded {len(results)} cached results from {cache}", flush=True)
    else:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for i, res in enumerate(ex.map(_worker, all_jobs, chunksize=1), 1):
                results.append(res)
                if i % 200 == 0:
                    print(f"jobs done: {i}/{len(all_jobs)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        print(f"All jobs done in {time.time()-t0:.0f}s", flush=True)
        if cache is not None:
            cache.write_bytes(pickle.dumps(results))
            print(f"Cached {len(results)} results to {cache}", flush=True)

    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)

    # ---------------- main tables (state + prev_state) -----------------------
    main_rows = []
    for target in TARGETS:
        main_rows.extend(aggregate_main(by_kind["main"], f"main/{target}", target))
    # attach maj/bal/f1 from per-result storage
    agg_by = {(r["tag"], r["layer"], r["subset"]): r for r in main_rows}
    for r in by_kind["main"]:
        for sub in SUBSETS:
            key = (r["tag"], r["layer"], sub)
            if key not in agg_by:
                continue
            a = r.get(f"acc_{sub}")
            if a is None:
                continue
            d = agg_by[key]
            for pref in ("bal", "f1", "maj"):
                v = r.get(f"{pref}_{sub}")
                if v is not None:
                    d.setdefault(f"_sum_{pref}", []).append(v)
    for d in agg_by.values():
        for pref in ("bal", "f1", "maj"):
            s = d.pop(f"_sum_{pref}", None)
            if s:
                d[f"{pref}_mean"] = float(np.mean(s))

    csv_path = args.out_dir / "previous_state_probe.csv"
    cols = ["tag", "target", "layer", "layer_name", "subset", "n_splits",
            "acc_mean", "acc_std", "bal_mean", "f1_mean", "maj_mean",
            "p_one_layer_full", "p_one_layer_ws", "null_mean", "null_p95",
            "p_maxT_full", "p_maxT_ws"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in sorted(main_rows, key=lambda x: (x["target"], x["subset"], x["layer"])):
            w.writerow(d)
    print(f"Wrote {csv_path} ({len(main_rows)} rows)", flush=True)

    # ---------------- cross + route ------------------------------------------
    route_rows = []
    for target in TARGETS:
        for direction in ("A_to_B", "B_to_A", "A_to_A_ctrl", "B_to_B_ctrl"):
            route_rows.extend(aggregate_single(
                by_kind["cross"], f"cross/{target}/{direction}",
                {"family": "cross", "target": target, "condition": direction}))
        for rt in route_names:
            route_rows.extend(aggregate_single(
                by_kind["route"], f"route/{target}/{rt}",
                {"family": "route", "target": target, "condition": rt}))
    route_cols = ["family", "target", "condition", "layer", "layer_name",
                  "n_splits", "acc_mean", "p_one_layer", "p_maxT",
                  "null_mean", "null_p95"]
    with open(args.out_dir / "route_generalization.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=route_cols, extrasaction="ignore")
        w.writeheader()
        for d in sorted(route_rows, key=lambda x: (x["target"], x["family"],
                                                   x["condition"], x["layer"])):
            w.writerow(d)
    print(f"Wrote {args.out_dir/'route_generalization.csv'} "
          f"({len(route_rows)} rows)", flush=True)

    # ---------------- LOO: composition + canonical evidence ------------------
    loo_res = { (r["traj"], r["layer"]): r for r in by_kind["loo"] }
    comp_rows = []
    for i, r in enumerate(rows):
        tr = r["trajectory_id"]
        per_layer = {}
        j = int(np.sum(traj_of[:i] == tr))  # row index within traj (0..4)
        for layer in range(L1):
            lr = loo_res.get((tr, layer))
            if lr is None:
                continue
            pp = np.asarray(lr["proba_prev_state"])    # (5,3)
            pe = np.asarray(lr["proba_event"])
            ps = np.asarray(lr["proba_state"])
            prev_hat = POS_NAME[int(pp[j].argmax())]
            event_hat = EVENTS[int(pe[j].argmax())]
            cur_hat = POS_NAME[int(ps[j].argmax())]
            symbolic = POS_NAME[apply_swap(POS[prev_hat], PAIR[event_hat])]
            oracle_a = POS_NAME[apply_swap(POS[r["gt_prev_state"]],
                                           PAIR[event_hat])]
            oracle_b = POS_NAME[apply_swap(POS[prev_hat],
                                           PAIR[r["gt_event"]])]
            p_gtprev = float(pp[j][POS[r["gt_prev_state"]]])
            p_gtcur = float(ps[j][POS[r["gt_state"]]])
            p_gtevt = float(pe[j][EVENT_INDEX[r["gt_event"]]])
            per_layer[layer] = {
                "p_prev_gt": p_gtprev,
                "p_prev_margin": p_gtprev -
                float(np.delete(pp[j], POS[r["gt_prev_state"]]).max()),
                "p_cur_gt": p_gtcur,
                "cur_margin": p_gtcur - float(ps[j][POS[r["gt_prev_state"]]]),
                "p_event_gt": p_gtevt,
                "p_event_margin": p_gtevt -
                float(np.delete(pe[j], EVENT_INDEX[r["gt_event"]]).max()),
                "prev_hat": prev_hat, "event_hat": event_hat,
                "symbolic": symbolic, "oracle_a": oracle_a,
                "oracle_b": oracle_b, "cur_hat": cur_hat,
            }
        if not per_layer:
            continue
        comp_rows.append({
            "key": index[i], "trajectory_id": tr, "t": r["t"],
            "gt_prev_state": r["gt_prev_state"], "gt_state": r["gt_state"],
            "gt_event": r["gt_event"], "is_transition": r["is_transition"],
            "clean_revision": r["clean_revision"],
            "rev_success": subset_mask(r, "rev_success"),
            "rev_stale": is_stale(r),
            "state_pred": r["state_pred"], "state_correct": r["state_correct"],
            "native_state_acc": float(r["state_correct"]),
            "prev_state_acc": int(per_layer[36]["prev_hat"] == r["gt_prev_state"]),
            "event_acc": int(per_layer[36]["event_hat"] == r["gt_event"]),
            "symbolic_acc": int(per_layer[36]["symbolic"] == r["gt_state"]),
            "oracle_a_acc": int(per_layer[36]["oracle_a"] == r["gt_state"]),
            "oracle_b_acc": int(per_layer[36]["oracle_b"] == r["gt_state"]),
            "per_layer": per_layer,
        })
    # note: per-row headline uses layer_36 (final layer) for the decoded hats
    flat_cols = ["key", "trajectory_id", "t", "gt_prev_state", "gt_state",
                 "gt_event", "is_transition", "clean_revision", "rev_success",
                 "rev_stale", "state_pred", "state_correct", "native_state_acc",
                 "prev_state_acc", "event_acc", "symbolic_acc", "oracle_a_acc",
                 "oracle_b_acc"]
    with open(args.out_dir / "symbolic_composition.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for d in comp_rows:
            w.writerow(d)
    print(f"Wrote {args.out_dir/'symbolic_composition.csv'} "
          f"({len(comp_rows)} rows)", flush=True)

    # ---------------- canonical operand evidence -----------------------------
    can_rows = [c for c in comp_rows if c["clean_revision"]]
    ev_cols = ["key", "group", "layer", "layer_name", "gt_prev_state",
               "gt_state", "p_prev_gt", "p_prev_margin", "p_cur_gt",
               "cur_margin", "p_event_gt", "p_event_margin"]
    ev_rows = []
    for c in can_rows:
        for layer in range(L1):
            pl = c["per_layer"].get(layer)
            if pl is None:
                continue
            group = ("stale" if c["rev_stale"]
                     else "success" if c["rev_success"] else "other_failure")
            ev_rows.append({
                "key": c["key"], "group": group, "layer": layer,
                "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                "gt_prev_state": c["gt_prev_state"], "gt_state": c["gt_state"],
                "p_prev_gt": pl["p_prev_gt"], "p_prev_margin": pl["p_prev_margin"],
                "p_cur_gt": pl["p_cur_gt"], "cur_margin": pl["cur_margin"],
                "p_event_gt": pl["p_event_gt"],
                "p_event_margin": pl["p_event_margin"],
            })
    with open(args.out_dir / "canonical_operand_evidence.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev_cols)
        w.writeheader()
        for d in ev_rows:
            w.writerow(d)
    print(f"Wrote {args.out_dir/'canonical_operand_evidence.csv'} "
          f"({len(ev_rows)} rows)", flush=True)

    # ---------------- summaries ----------------------------------------------
    summary = {
        "config": {
            "hidden_states": "outputs/vetbench/hidden_state_probe/hidden_states.npz "
                             "(P0-validated, probe = last input token pre-generation)",
            "behavior": "Transformers-aligned "
                        "(outputs/vetbench/composition_analysis_v1/"
                        "transformers_behavior.csv)",
            "solver": f"{SOLVER} max_iter=1000",
            "seeds": seeds, "c_grid": c_grid,
            "n_perms": args.n_perms, "loo_seeds": args.loo_seeds,
            "group_split": "by trajectory_id; LOO = leave-one-trajectory-out",
            "prev_state_target": "GT S_{t-1}; at t=1 this equals the initial "
                                 "state named in the prompt text (caveat for "
                                 "t=1 rows)",
        },
        "main_probe": {
            f"{target}/{sub}": {
                "layers": {d["layer_name"]: {k: d[k] for k in
                               ("acc_mean", "bal_mean", "f1_mean", "maj_mean",
                                "p_one_layer_full", "p_one_layer_ws",
                                "p_maxT_full", "p_maxT_ws")}
                           for d in main_rows
                           if d["target"] == target and d["subset"] == sub},
            }
            for target in TARGETS for sub in SUBSETS
        },
        "cross_route": {
            f"{tgt}/{fam}/{cond}": {
                "peak_acc": max(d2["acc_mean"] for d2 in route_rows
                                if d2["target"] == tgt and d2["family"] == fam
                                and d2["condition"] == cond),
                "min_p_maxT": min((x["p_maxT"] for x in route_rows
                                   if x["target"] == tgt and x["family"] == fam
                                   and x["condition"] == cond
                                   and x["p_maxT"] is not None), default=None),
            }
            for tgt, fam, cond in sorted({(x["target"], x["family"],
                                           x["condition"]) for x in route_rows}
                                         , key=lambda t: (t[0], t[1], t[2]))
        },
    }
    # composition accuracy per subset
    comp_summary = {}
    for sub_name, pred in (
        ("all", lambda c: True),
        ("true_transition", lambda c: c["is_transition"]),
        ("clean_revision", lambda c: c["clean_revision"]),
        ("canonical_stale", lambda c: c["rev_stale"]),
        ("revision_success", lambda c: c["rev_success"]),
    ):
        sel = [c for c in comp_rows if pred(c)]
        if not sel:
            continue
        def rate(key):
            return float(np.mean([c[key] for c in sel]))
        entry = {"n": len(sel), "n_trajectories": len({c["trajectory_id"] for c in sel})}
        for key in ("native_state_acc", "prev_state_acc", "event_acc",
                    "symbolic_acc", "oracle_a_acc", "oracle_b_acc"):
            entry[key] = rate(key)
            vb = defaultdict(list)
            for c in sel:
                vb[c["trajectory_id"]].append(c[key])
            lo, hi = boot_ci({t: np.asarray(v) for t, v in vb.items()})
            entry[f"{key}_boot95"] = [lo, hi]
        gt_counts = Counter(c["gt_state"] for c in sel)
        entry["gt_state_majority"] = max(gt_counts.values()) / len(sel)
        entry["chance"] = 1 / 3
        comp_summary[sub_name] = entry
    summary["symbolic_composition"] = comp_summary
    # canonical evidence group means
    ev_summary = {}
    groups = defaultdict(lambda: defaultdict(list))
    for d in ev_rows:
        for k in ("p_prev_gt", "p_prev_margin", "p_cur_gt", "cur_margin",
                  "p_event_gt"):
            groups[d["group"]][k].append(d[k])
    for g, dd in groups.items():
        ev_summary[g] = {k: {"mean": float(np.mean(v)),
                             "n": len(v)} for k, v in dd.items()}
    # permutation test: cur_margin stale vs success
    stale_m = [d["cur_margin"] for d in ev_rows if d["group"] == "stale"]
    succ_m = [d["cur_margin"] for d in ev_rows if d["group"] == "success"]
    obs = (float(np.mean(stale_m)) - float(np.mean(succ_m))) \
        if stale_m and succ_m else None
    p_perm = None
    if obs is not None:
        allm = np.asarray(stale_m + succ_m, dtype=np.float64)
        rng = random.Random(12345)
        n_perm = 10000
        cnt = 0
        ns = len(stale_m)
        for _ in range(n_perm):
            perm = allm[rng.sample(range(len(allm)), len(allm))]
            diff = float(perm[:ns].mean() - perm[ns:].mean())
            if abs(diff) >= abs(obs):
                cnt += 1
        p_perm = cnt / n_perm
    summary["canonical_evidence"] = {
        "group_means": ev_summary,
        "cur_margin_stale_minus_success": {
            "observed": obs, "p_two_sided": p_perm,
            "n_stale_rows": len({d['key'] for d in ev_rows if d['group'] == 'stale'}),
            "n_success_rows": len({d['key'] for d in ev_rows if d['group'] == 'success'}),
        },
        "note": "decoders are leave-one-trajectory-out (fit on the other 49 "
                "trajectories, 10 CV seeds for C, averaged probabilities); "
                "cur_margin = P(GT S_t) - P(GT S_{t-1}) under the current-state "
                "decoder; p_prev_margin / p_event_margin = P(GT operand) - "
                "max(P of the other two classes)",
    }
    # per-layer composition profile (headline per-row uses the final layer 36;
    # this table shows how composition accuracy varies by layer)
    per_layer_comp = {}
    for layer in range(L1):
        if not comp_rows or 36 not in comp_rows[0]["per_layer"]:
            continue
        def acc_at(key_hat, key_gt):
            vals = [int(c["per_layer"][layer][key_hat] == c[key_gt])
                    for c in comp_rows if layer in c["per_layer"]]
            return float(np.mean(vals)) if vals else None
        per_layer_comp[f"layer_{layer:02d}" if layer else "embedding"] = {
            "prev_state_acc": acc_at("prev_hat", "gt_prev_state"),
            "event_acc": acc_at("event_hat", "gt_event"),
            "symbolic_acc": acc_at("symbolic", "gt_state"),
            "oracle_a_acc": acc_at("oracle_a", "gt_state"),
            "oracle_b_acc": acc_at("oracle_b", "gt_state"),
        }
    summary["composition_per_layer"] = per_layer_comp
    summary["composition_headline_layer"] = (
        "final layer (36): per-row decoded hats in symbolic_composition.csv "
        "use layer_36, the layer the model itself uses for the first token")
    (args.out_dir / "previous_state_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\nWrote {args.out_dir}/previous_state_probe_summary.json")

    for target in TARGETS:
        for sub in ("all",):
            tab = [d for d in main_rows if d["target"] == target
                   and d["subset"] == sub and d["layer"] != 0]
            lm = float(np.mean([d["acc_mean"] for d in tab]))
            best = min((d for d in tab if d["p_maxT_full"] is not None),
                       key=lambda d: d["p_maxT_full"], default=None)
            print(f"{target}/all layer-mean acc = {lm:.3f}; "
                  f"best p_maxT_full = "
                  f"{best['p_maxT_full'] if best else None} "
                  f"({best['layer_name'] if best else '-'})")
    for sub_name, entry in comp_summary.items():
        print(f"composition {sub_name}: n={entry['n']}  "
              f"native={entry['native_state_acc']:.3f}  "
              f"prev={entry['prev_state_acc']:.3f}  "
              f"event={entry['event_acc']:.3f}  "
              f"symbolic={entry['symbolic_acc']:.3f}  "
              f"oracleA={entry['oracle_a_acc']:.3f}  "
              f"oracleB={entry['oracle_b_acc']:.3f}")
    if stale_m and succ_m:
        print(f"canonical cur_margin: stale {np.mean(stale_m):.3f} vs "
              f"success {np.mean(succ_m):.3f} (diff {obs:.3f}, p={p_perm})")
    # layer comparison plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
        for target, style in (("state", "-o"), ("prev_state", "-s")):
            xs = [d["layer"] for d in main_rows
                  if d["target"] == target and d["subset"] == "all"]
            ys = [d["acc_mean"] for d in main_rows
                  if d["target"] == target and d["subset"] == "all"]
            ax.plot(xs, ys, style, label=f"{target} (all)")
        ax.axhline(1 / 3, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out probe acc (10-seed mean)")
        ax.set_title("Current vs previous state decoding per layer")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "layer_comparison.png", dpi=140)
        print(f"Wrote {args.out_dir}/layer_comparison.png")
    except Exception as e:  # matplotlib optional
        print(f"(no plot: {e})")


def _worker(job: dict) -> dict:
    # One BLAS thread per worker: with 24 workers the pool would otherwise
    # oversubscribe the 128-core box (each worker spawns ~5-6 OpenMP threads)
    # and throughput collapses under contention.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if job["kind"] == "main":
            return _job_main(job)
        if job["kind"] == "cross":
            return _job_cross(job)
        if job["kind"] == "route":
            return _job_route(job)
        if job["kind"] == "loo":
            return _job_loo(job)
        raise ValueError(job["kind"])


if __name__ == "__main__":
    main()
