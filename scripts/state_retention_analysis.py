"""Paired pre/post-event state-retention analysis for StateRev-VL.

Question
--------
In canonical stale revision failures, is the previous state S_{t-1}
(a) never stably represented, or (b) represented before the new visual event
but destroyed / re-encoded once the event is shown?

Design
------
For each transition step t (t=1..5) of each trajectory:

    h_pre  = h_{t-1}   hidden state of the (t-1)-prefix  (BEFORE swap t)
    h_post = h_t       hidden state of the t-prefix      (AFTER  swap t)

The frozen pre-event decoder protocol (per seed, per layer l):

  1. trajectory group split (40/10, no overlap);
  2. D_pre^l is trained ONLY on the train trajectories' h_pre rows,
     target = GT S_{t-1} (train-only StandardScaler, L2 logistic regression,
     C by 3-fold group CV inside the train set);
  3. the SAME frozen (D_pre^l, scaler) is then applied to
        (i)  h_pre  of the held-out trajectories  -> "pre" performance
        (ii) h_post of the held-out trajectories  -> "post" performance
     No separate post-event classifier is ever trained.
     retention_drop = performance(h_post) - performance(h_pre).
  4. within-t label-permutation nulls on the pre labels; the refit decoder is
     evaluated on both h_pre and h_post (and the paired drop);
  5. family-wise maxT over the 37 layers (10 seeds x n_perms each).

Additionally, per held-out (pair, layer):

  * P_pre(GT S_{t-1} | h_pre), P_pre(GT S_{t-1} | h_post)
  * pre/post margins, probability_drop, margin_drop
  * raw drift: cosine similarity + L2-normalized distance between h_pre/h_post
    (descriptive only - NOT a state-drift claim)
  * projected_state_drift: change of the 3-d logit vector of the frozen
    D_pre (the state-relevant subspace) between h_pre and h_post
  * an event decoder D_ev^l trained on the train trajectories' h_post rows
    (target GT E_t; the event exists only in the post prefix), evaluated on
    the held-out h_post - "strong event + collapsed prior state" evidence.

The frozen pre decoder is trained on h_pre rows only, so nothing about the
post prefix leaks into its parameters.

Causal-patching interface (reserved, NOT implemented this round): every
output row keeps trajectory_id, t, layer, h_pre_key, h_post_key, the
canonical group and the decoder margins; hidden states are addressable by
key in hidden_states.npz, so future activation-replacement / steering
experiments can consume these files directly. No patching, steering,
attention or MLP intervention is run here.

Outputs (out-dir, default outputs/vetbench/state_retention_analysis_v1/):
    retention_probe_results.csv     per (layer, subset): pre/post acc, drop,
                                    p-values, maxT
    retention_heldout_samples.csv   per (pair, layer): margins, drift, event
    canonical_retention_evidence.csv canonical rows at the headline layer
    retention_summary.json          config, group contrasts, maxT, counts
    retention_report.md             auto-generated data report

Run from the project root (CPU only; hidden states are pre-extracted):
    python scripts/state_retention_analysis.py
Dry run (tiny, for code-path validation):
    python scripts/state_retention_analysis.py --limit-trajectories 4 \
        --seeds 0 --layers 0,5,36 --n-perms 3 \
        --out-dir /tmp/opencode/retention_dryrun
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
    STATE_CLASSES,
    STATE_INDEX,
    _pad_proba,
    group_split,
)
from strict_probe_analysis import SOLVER, _fit_lr2, _select_c2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "state_retention_analysis_v1"
PROBE_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "hidden_state_probe"

N_LAYERS = 37  # embedding + 36
HIDDEN_SIZE = 4096
HEADLINE_LAYER = 36  # final layer: where the model itself emits the first token
SUBSETS = ("all", "clean_revision", "rev_success", "rev_stale",
           "clean_maintenance")
C_GRID_DEFAULT = (0.1, 1.0, 10.0)

CSV_RESULTS = ["layer", "layer_name", "subset", "n_splits",
               "acc_pre_mean", "acc_post_mean", "drop_mean",
               "bal_pre_mean", "bal_post_mean", "f1_pre_mean", "f1_post_mean",
               "maj_mean",
               "p_one_layer_pre", "p_one_layer_post", "p_one_layer_drop",
               "p_maxT_pre", "p_maxT_post", "p_maxT_drop",
               "null_mean", "null_p95"]
CSV_SAMPLES = ["pair_id", "trajectory_id", "t", "h_pre_key", "h_post_key",
               "layer", "layer_name", "canonical_group", "clean_maintenance",
               "gt_prev_state", "gt_state", "gt_event",
               "p_pre_gt", "p_post_gt", "pre_margin", "post_margin",
               "prob_drop", "margin_drop",
               "p_event_gt", "drift_proj", "drift_logit_gt",
               "cos_pre_post", "l2_norm_dist"]
CSV_CANONICAL = ["pair_id", "trajectory_id", "t", "h_pre_key", "h_post_key",
                 "layer", "layer_name", "canonical_group",
                 "gt_prev_state", "gt_state", "gt_event",
                 "p_pre_gt", "p_post_gt", "pre_margin", "post_margin",
                 "margin_drop", "prob_drop", "p_event_gt",
                 "native_state_correct", "native_state_pred",
                 "native_event_correct"]


def subset_mask(r: dict, name: str) -> bool:
    if name == "all":
        return True
    if name == "clean_revision":
        return bool(r["clean_revision"])
    if name == "rev_success":
        return bool(r["clean_revision"]) and bool(r["state_correct"])
    if name == "rev_stale":
        return bool(r["canonical_stale_failure"])
    if name == "clean_maintenance":
        return bool(r["clean_maintenance"])
    raise KeyError(name)


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


def _pval(obs: float, nulls: list[float]) -> float | None:
    if not nulls:
        return None
    arr = np.asarray(nulls)
    return float((arr >= obs).mean())


def _margins(proba: np.ndarray, gt_idx: np.ndarray) -> np.ndarray:
    """Per-row margin P(GT) - max(P(other classes)). Masks the GT column
    per row (np.delete would treat gt_idx as a column SET, not per-row)."""
    masked = proba.copy()
    masked[np.arange(len(gt_idx)), gt_idx] = -np.inf
    return proba[np.arange(len(gt_idx)), gt_idx] - masked.max(axis=1)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _job_retention(job: dict) -> dict:
    """One (seed, layer): frozen pre decoder + event decoder + nulls."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _job_retention_body(job)


def _job_retention_body(job: dict) -> dict:
    from sklearn.preprocessing import StandardScaler
    Xp_all = np.load(job["x_pre_path"], mmap_mode="r")
    Xq_all = np.load(job["x_post_path"], mmap_mode="r")
    L = job["layer"]
    tr, te = job["train_rows"], job["test_rows"]

    Xtr_pre = np.asarray(Xp_all[tr, L, :], dtype=np.float32)
    Xte_pre = np.asarray(Xp_all[te, L, :], dtype=np.float32)
    Xte_post = np.asarray(Xq_all[te, L, :], dtype=np.float32)
    Xtr_post = np.asarray(Xq_all[tr, L, :], dtype=np.float32)
    y_pre_tr = job["y_pre"][tr]
    y_ev_tr = job["y_event"][tr]
    trajs_tr = [job["traj_of"][i] for i in tr]
    t_tr = np.array([job["t_of"][i] for i in tr])
    k = 3

    if len(set(y_pre_tr.tolist())) < 2:
        return {"kind": "ret", "empty": True, "seed": job["seed"],
                "layer": L}

    # ---- frozen pre-event decoder (trained on h_pre ONLY) ----------------
    sc_pre = StandardScaler().fit(Xtr_pre)
    Xtr_pre_s = sc_pre.transform(Xtr_pre)
    Xte_pre_s = sc_pre.transform(Xte_pre)
    Xte_post_s = sc_pre.transform(Xte_post)
    c_pre = _select_c2(Xtr_pre_s, y_pre_tr, trajs_tr, job["c_grid"],
                       job["c_key"])
    clf_pre = _fit_lr2(Xtr_pre_s, y_pre_tr, c_pre)

    def eval_pre(x):
        proba = _pad_proba(clf_pre.predict_proba(x), clf_pre.classes_, k)
        return proba, proba.argmax(axis=1)

    proba_pre, pred_pre = eval_pre(Xte_pre_s)
    proba_post, pred_post = eval_pre(Xte_post_s)
    n_te = len(te)

    # projected drift in the decoder's state-relevant subspace (3-d logits)
    W = clf_pre.coef_                       # (3, D)
    b = clf_pre.intercept_                  # (3,)
    logit_pre = Xte_pre_s @ W.T + b
    logit_post = Xte_post_s @ W.T + b
    drift_proj = np.linalg.norm(logit_post - logit_pre, axis=1)
    gt_idx = np.array([job["gt_prev_te"][i] for i in range(n_te)])
    drift_logit = logit_post[np.arange(n_te), gt_idx] - logit_pre[np.arange(n_te), gt_idx]

    # ---- event decoder (trained on h_post of train rows) -----------------
    sc_ev = StandardScaler().fit(Xtr_post)
    Xtr_ev_s = sc_ev.transform(Xtr_post)
    Xte_ev_s = sc_ev.transform(Xte_post)
    p_event: list[float | None] = [None] * n_te
    if len(set(y_ev_tr.tolist())) >= 2:
        c_ev = _select_c2(Xtr_ev_s, y_ev_tr, trajs_tr, job["c_grid"],
                          job["c_key"] + "-ev")
        clf_ev = _fit_lr2(Xtr_ev_s, y_ev_tr, c_ev)
        p_ev = _pad_proba(clf_ev.predict_proba(Xte_ev_s), clf_ev.classes_, k)
        ev_idx = np.array([job["gt_event_te"][i] for i in range(n_te)])
        p_event = p_ev[np.arange(n_te), ev_idx].tolist()
        c_ev_out = float(c_ev)
    else:
        c_ev_out = None

    # ---- subset metrics: pre / post / drop -------------------------------
    rows_te = job["rows_te"]
    out: dict[str, Any] = {
        "kind": "ret", "empty": False, "seed": job["seed"], "layer": L,
        "C_pre": float(c_pre), "C_event": c_ev_out,
        "n_train": len(tr), "n_test": n_te,
        "test_index": job["test_ids"],
        "samples": {
            "p_pre_gt": proba_pre[np.arange(n_te), gt_idx].tolist(),
            "p_post_gt": proba_post[np.arange(n_te), gt_idx].tolist(),
            "pre_margin": _margins(proba_pre, gt_idx).tolist(),
            "post_margin": _margins(proba_post, gt_idx).tolist(),
            "p_event_gt": p_event,
            "drift_proj": drift_proj.tolist(),
            "drift_logit_gt": drift_logit.tolist(),
            "pred_pre": pred_pre.tolist(),
            "pred_post": pred_post.tolist(),
        },
    }
    for name in SUBSETS:
        m = np.array([subset_mask(rows_te[i], name) for i in range(n_te)])
        if m.sum() < 2:
            out[f"acc_pre_{name}"] = None
            continue
        yt = np.array([job["gt_prev_te"][i] for i in range(n_te)])[m]
        out[f"acc_pre_{name}"] = float((pred_pre[m] == yt).mean())
        out[f"acc_post_{name}"] = float((pred_post[m] == yt).mean())
        out[f"bal_pre_{name}"] = balanced_acc(yt, pred_pre[m])
        out[f"bal_post_{name}"] = balanced_acc(yt, pred_post[m])
        out[f"f1_pre_{name}"] = macro_f1(yt, pred_pre[m], k)
        out[f"f1_post_{name}"] = macro_f1(yt, pred_post[m], k)
        out[f"maj_{name}"] = float(np.bincount(yt, minlength=k).max() / m.sum())

    # ---- within-t label-permutation nulls on the PRE labels --------------
    rng = random.Random(job["rng_key"])
    nulls: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for b in range(job["n_perms"]):
        y_perm = y_pre_tr.copy()
        for t in np.unique(t_tr):
            sel = np.flatnonzero(t_tr == t)
            y_perm[sel] = y_perm[sel][rng.sample(range(len(sel)), len(sel))]
        if len(set(y_perm.tolist())) < 2:
            continue
        clf_n = _fit_lr2(Xtr_pre_s, y_perm, c_pre)
        pn_pre = _pad_proba(clf_n.predict_proba(Xte_pre_s), clf_n.classes_, k)
        pn_post = _pad_proba(clf_n.predict_proba(Xte_post_s), clf_n.classes_, k)
        for name in SUBSETS:
            m = np.array([subset_mask(rows_te[i], name) for i in range(n_te)])
            if m.sum() < 2:
                continue
            yt = np.array([job["gt_prev_te"][i] for i in range(n_te)])[m]
            acc_p = float((pn_pre.argmax(axis=1)[m] == yt).mean())
            acc_q = float((pn_post.argmax(axis=1)[m] == yt).mean())
            nulls[name]["pre"].append(acc_p)
            nulls[name]["post"].append(acc_q)
            nulls[name]["drop"].append(acc_q - acc_p)
    out["nulls"] = {k2: v for k2, v in nulls.items()}
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"] = int(r["t"])
        for k in ("h_pre_available", "h_post_available", "pair_complete",
                  "is_transition", "event_correct", "prev_state_correct",
                  "state_correct", "clean_revision", "revision_success",
                  "canonical_stale_failure", "clean_maintenance"):
            r[k] = r[k] == "true"
    return rows


def load_hidden(path: Path, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return X_pre, X_post of shape (n_pairs, 37, 4096) float32."""
    need_pre = {p["h_pre_key"] for p in pairs}
    need_post = {p["h_post_key"] for p in pairs}
    with np.load(path, allow_pickle=False) as z:
        have = set(z.files)
        missing = (need_pre | need_post) - have
        if missing:
            raise SystemExit(f"hidden states missing for {len(missing)} pair "
                             f"keys (e.g. {sorted(missing)[:3]}); run the "
                             f"t=0 backfill first")
        Xp = np.zeros((len(pairs), N_LAYERS, HIDDEN_SIZE), dtype=np.float32)
        Xq = np.zeros_like(Xp)
        for i, p in enumerate(pairs):
            Xp[i] = np.asarray(z[p["h_pre_key"]], dtype=np.float32)
            Xq[i] = np.asarray(z[p["h_post_key"]], dtype=np.float32)
    return Xp, Xq


def raw_drift(Xp: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per (pair, layer) cosine similarity and L2-normalized distance."""
    a = Xp / (np.linalg.norm(Xp, axis=-1, keepdims=True) + 1e-12)
    b = Xq / (np.linalg.norm(Xq, axis=-1, keepdims=True) + 1e-12)
    cos = (a * b).sum(axis=-1)
    l2 = np.linalg.norm(a - b, axis=-1)
    return cos, l2


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict], n_layers: int) -> list[dict]:
    """Per (layer, subset): seed-mean pre/post/drop + one-layer p + maxT."""
    layers = list(range(n_layers))
    per_seed: dict[int, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    vecs: dict[tuple[int, int], dict[tuple[int, str, str], float]] = \
        defaultdict(dict)
    for r in results:
        if r.get("empty"):
            continue
        for sub in SUBSETS:
            a = r.get(f"acc_pre_{sub}")
            if a is None:
                continue
            per_seed[r["seed"]][r["layer"]][f"pre_{sub}"] = a
            per_seed[r["seed"]][r["layer"]][f"post_{sub}"] = r[f"acc_post_{sub}"]
            per_seed[r["seed"]][r["layer"]][f"drop_{sub}"] = (
                r[f"acc_post_{sub}"] - a)
        for sub, dd in r.get("nulls", {}).items():
            for b, (v_pre, v_post) in enumerate(
                    zip(dd.get("pre", []), dd.get("post", []))):
                v = vecs[(r["seed"], b)]
                v[(r["layer"], "pre", sub)] = v_pre
                v[(r["layer"], "post", sub)] = v_post
                v[(r["layer"], "drop", sub)] = v_post - v_pre
    out = []
    for layer in layers:
        for sub in SUBSETS:
            for stat in ("pre", "post", "drop"):
                ps = [per_seed[s][layer][f"{stat}_{sub}"]
                      for s in sorted(per_seed)
                      if f"{stat}_{sub}" in per_seed[s].get(layer, {})]
                if not ps:
                    continue
                mean = float(np.mean(ps))
                nulls = [m[(layer, stat, sub)] for (s, b), m in vecs.items()
                          if (layer, stat, sub) in m]
                row = {"layer": layer,
                       "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                       "subset": sub, "stat": stat, "n_splits": len(ps),
                       "mean": mean,
                       "p_one_layer": _pval(mean, nulls),
                       "null_mean": float(np.mean(nulls)) if nulls else None,
                       "null_p95": float(np.percentile(nulls, 95)) if nulls else None}
                out.append(row)
    # family-wise maxT per (subset, stat)
    for sub in SUBSETS:
        for stat in ("pre", "post", "drop"):
            means = {row["layer"]: row["mean"] for row in out
                     if row["subset"] == sub and row["stat"] == stat}
            for row in out:
                if row["subset"] != sub or row["stat"] != stat:
                    continue
                maxts = []
                for (s, b), m in vecs.items():
                    vals = [m[(l, stat, sub)] for l in means
                            if (l, stat, sub) in m]
                    if len(vals) == len(means):
                        maxts.append(max(vals))
                row["p_maxT"] = (float(np.mean(np.asarray(maxts) >= row["mean"]))
                                 if maxts else None)
    return out


def _boot_ci(values_by_traj: dict[str, list], n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float]:
    trajs = sorted(values_by_traj)
    arrays = [np.asarray(values_by_traj[t], dtype=np.float64) for t in trajs]
    total = sum(len(a) for a in arrays)
    if total == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(trajs), size=len(trajs))
        means[i] = float(np.concatenate([arrays[j] for j in idx]).mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _group_pred(name: str):
    if name == "stale":
        return lambda s: s["canonical_group"] == "stale"
    if name == "success":
        return lambda s: s["canonical_group"] == "success"
    if name == "other_failure":
        return lambda s: s["canonical_group"] == "other_failure"
    if name == "maintenance":
        return lambda s: bool(s["clean_maintenance"])
    if name == "rest":
        return lambda s: s["canonical_group"] != "stale"
    raise KeyError(name)


def group_contrasts(samples: list[dict], layer: int,
                    value_key: str, g1: str, g2: str,
                    n_perms: int = 10000) -> dict:
    """Row-level permutation test of mean(value) between two groups at one
    layer, plus trajectory-unit bootstrap CIs per group. Rows whose value is
    missing (never held out in any seed) are excluded."""
    p1, p2 = _group_pred(g1), _group_pred(g2)
    sel = [s for s in samples if s["layer"] == layer and (p1(s) or p2(s))]
    a = [s[value_key] for s in sel if p1(s) and s[value_key] is not None]
    b = [s[value_key] for s in sel if p2(s) and s[value_key] is not None
         and not p1(s)]
    out = {"group_a": g1, "group_b": g2, "n_a": len(a), "n_b": len(b),
           "mean_a": float(np.mean(a)) if a else None,
           "mean_b": float(np.mean(b)) if b else None}
    if not a or not b:
        return out
    by_traj_a: dict[str, list] = defaultdict(list)
    by_traj_b: dict[str, list] = defaultdict(list)
    for s in sel:
        v = s[value_key]
        if v is None:
            continue
        if p1(s):
            by_traj_a[s["trajectory_id"]].append(v)
        elif p2(s):
            by_traj_b[s["trajectory_id"]].append(v)
    out["boot95_a"] = _boot_ci(by_traj_a)
    out["boot95_b"] = _boot_ci(by_traj_b)
    aa, bb = np.asarray(a), np.asarray(b)
    obs = float(aa.mean() - bb.mean())
    pooled = np.concatenate([aa, bb])
    rng = np.random.default_rng(0)
    ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(pooled)
        d = perm[:len(aa)].mean() - perm[len(aa):].mean()
        ge += d >= obs
    out["observed_diff_a_minus_b"] = obs
    out["p_two_sided"] = float(2 * min(ge / n_perms, 1 - ge / n_perms))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path,
                    default=V1_DIR / "state_retention_pairs.csv")
    ap.add_argument("--npz", type=Path,
                    default=PROBE_DIR / "hidden_states.npz")
    ap.add_argument("--out-dir", type=Path, default=V1_DIR)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--layers", type=str, default="all",
                    help="'all' or comma-separated layer indices (0..36)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--c-grid", type=str, default="0.1,1.0,10.0")
    ap.add_argument("--n-perms", type=int, default=100,
                    help="within-t label-permutation nulls per (seed, layer)")
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--limit-trajectories", type=int, default=0,
                    help="keep only the first N trajectories (dry-run helper)")
    ap.add_argument("--skip-jobs", action="store_true",
                    help="reuse the cached job results pkl and only re-run "
                         "aggregation/writing")
    ap.add_argument("--cache-pkl", type=Path, default=None,
                    help="job-results cache (default: <out-dir>/retention_results.pkl)")
    args = ap.parse_args()
    if args.cache_pkl is None:
        args.cache_pkl = args.out_dir / "retention_results.pkl"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    c_grid = [float(c) for c in args.c_grid.split(",") if c.strip() != ""]

    pairs = load_pairs(args.pairs)
    complete = [p for p in pairs if p["pair_complete"]]
    if args.limit_trajectories:
        keep = sorted({p["trajectory_id"] for p in complete}
                      )[: args.limit_trajectories]
        complete = [p for p in complete if p["trajectory_id"] in keep]
    if not complete:
        raise SystemExit("no complete pairs (both h_pre and h_post present)")
    complete = sorted(complete, key=lambda p: (p["trajectory_id"], p["t"]))
    trajs = sorted({p["trajectory_id"] for p in complete})
    traj_of = [p["trajectory_id"] for p in complete]
    t_of = [p["t"] for p in complete]
    y_pre = np.array([STATE_INDEX[p["gt_prev_state"]] for p in complete])
    y_event = np.array([EVENT_INDEX[p["gt_event"]] for p in complete])
    gt_prev = [STATE_INDEX[p["gt_prev_state"]] for p in complete]
    pair_ids = [f"{p['trajectory_id']}_t{p['t']}" for p in complete]

    print(f"Complete pairs: {len(complete)} across {len(trajs)} trajectories "
          f"(t values: {sorted(set(t_of))})")
    print(f"Seeds: {seeds}; layers: {layers[0]}..{layers[-1]} "
          f"({len(layers)}); perms: {args.n_perms}")
    for name in SUBSETS:
        n = sum(1 for p in complete if subset_mask(p, name))
        print(f"  subset {name}: {n}")

    # sufficiency verdict (carried into the summary)
    missing_t0 = [p["h_pre_key"] for p in pairs if not p["h_pre_available"]]
    sufficient = not missing_t0

    if not args.skip_jobs:
        Xp, Xq = load_hidden(args.npz, complete)
        cos, l2 = raw_drift(Xp, Xq)
        tmp = Path(tempfile.mkdtemp(prefix="ret_X_"))
        x_pre_file = tmp / "X_pre.npy"
        x_post_file = tmp / "X_post.npy"
        np.save(x_pre_file, Xp)
        np.save(x_post_file, Xq)
        del Xp, Xq

        jobs = []
        for seed in seeds:
            tr_set, te_set = group_split(trajs, seed, args.train_frac)
            tr_rows = [i for i, t in enumerate(traj_of) if t in tr_set]
            te_rows = [i for i, t in enumerate(traj_of) if t in te_set]
            if not tr_rows or not te_rows:
                continue
            for layer in layers:
                jobs.append({
                    "kind": "ret", "seed": seed, "layer": layer,
                    "x_pre_path": str(x_pre_file),
                    "x_post_path": str(x_post_file),
                    "train_rows": tr_rows, "test_rows": te_rows,
                    "y_pre": y_pre, "y_event": y_event,
                    "traj_of": traj_of, "t_of": t_of,
                    "gt_prev_te": [gt_prev[i] for i in te_rows],
                    "gt_event_te": [y_event[i] for i in te_rows],
                    "rows_te": [complete[i] for i in te_rows],
                    "test_ids": [pair_ids[i] for i in te_rows],
                    "c_grid": c_grid, "c_key": f"ret-{seed}-{layer}",
                    "rng_key": f"ret-null-{seed}-{layer}",
                    "n_perms": args.n_perms,
                })
        print(f"Running {len(jobs)} jobs with {args.n_jobs} workers...")
        t0 = time.time()
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for i, r in enumerate(ex.map(_job_retention, jobs, chunksize=1)):
                results.append(r)
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    print(f"jobs done: {i + 1}/{len(jobs)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        args.cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(args.cache_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"Cached {len(results)} results to {args.cache_pkl}")
    else:
        import pickle
        with open(args.cache_pkl, "rb") as f:
            results = pickle.load(f)
        # recompute raw drift (seed-independent)
        Xp, Xq = load_hidden(args.npz, complete)
        cos, l2 = raw_drift(Xp, Xq)
        del Xp, Xq

    # ---- per (pair, layer) held-out sample table -------------------------
    acc = {"pair_id": pair_ids}
    n_layers_eff = max(layers) + 1
    sample_vals: dict[str, list[tuple[int, ...]]] = {}
    for r in results:
        if r.get("empty"):
            continue
        L = r["layer"]
        for j, pid in enumerate(r["test_index"]):
            i = pair_ids.index(pid)
            for name in ("p_pre_gt", "p_post_gt", "pre_margin", "post_margin",
                         "drift_proj", "drift_logit_gt"):
                v = r["samples"][name][j]
                sample_vals.setdefault(f"{name}|{L}", []).append((i, v))
            ev = r["samples"]["p_event_gt"][j]
            if ev is not None:
                sample_vals.setdefault(f"p_event_gt|{L}", []).append((i, ev))
    # average over seeds per (pair, layer)
    sample_rows = []
    for layer in layers:
        for i, p in enumerate(complete):
            def avg(key):
                vals = [v for (ii, v) in sample_vals.get(f"{key}|{layer}", [])
                        if ii == i]
                return float(np.mean(vals)) if vals else None
            row = {
                "pair_id": pair_ids[i],
                "trajectory_id": p["trajectory_id"],
                "t": p["t"],
                "h_pre_key": p["h_pre_key"],
                "h_post_key": p["h_post_key"],
                "layer": layer,
                "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                "canonical_group": p["canonical_group"] or "none",
                "clean_maintenance": p["clean_maintenance"],
                "gt_prev_state": p["gt_prev_state"],
                "gt_state": p["gt_state"],
                "gt_event": p["gt_event"],
                "p_pre_gt": avg("p_pre_gt"),
                "p_post_gt": avg("p_post_gt"),
                "pre_margin": avg("pre_margin"),
                "post_margin": avg("post_margin"),
                "prob_drop": None, "margin_drop": None,
                "p_event_gt": avg("p_event_gt"),
                "drift_proj": avg("drift_proj"),
                "drift_logit_gt": avg("drift_logit_gt"),
                "cos_pre_post": float(cos[i, layer]),
                "l2_norm_dist": float(l2[i, layer]),
            }
            if row["p_pre_gt"] is not None and row["p_post_gt"] is not None:
                row["prob_drop"] = row["p_post_gt"] - row["p_pre_gt"]
            if row["pre_margin"] is not None and row["post_margin"] is not None:
                row["margin_drop"] = row["post_margin"] - row["pre_margin"]
            sample_rows.append(row)

    # ---- write CSVs -------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "retention_heldout_samples.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_SAMPLES)
        w.writeheader()
        for row in sample_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    res_rows = aggregate_results(results, n_layers_eff)
    # pivot to the wide CSV layout
    wide: dict[tuple[int, str], dict] = {}
    for row in res_rows:
        key = (row["layer"], row["subset"])
        d = wide.setdefault(key, {"layer": row["layer"],
                                  "layer_name": row["layer_name"],
                                  "subset": row["subset"]})
        d[f"{row['stat']}_mean" if row["stat"] != "drop" else "drop_mean"] = \
            row["mean"]
        d[f"p_one_layer_{row['stat']}"] = row["p_one_layer"]
        d[f"p_maxT_{row['stat']}"] = row["p_maxT"]
        if row["stat"] == "pre":
            d["null_mean"] = row["null_mean"]
            d["null_p95"] = row["null_p95"]
    # majority + n_splits + bal/f1 per subset from results
    for r in results:
        if r.get("empty"):
            continue
        for sub in SUBSETS:
            key = (r["layer"], sub)
            d = wide.setdefault(key, {"layer": r["layer"],
                                      "layer_name": f"layer_{r['layer']:02d}"
                                      if r["layer"] else "embedding",
                                      "subset": sub})
            for stat in ("pre", "post"):
                v = r.get(f"bal_{stat}_{sub}")
                if v is not None:
                    d.setdefault(f"_bal_{stat}", []).append(v)
                v = r.get(f"f1_{stat}_{sub}")
                if v is not None:
                    d.setdefault(f"_f1_{stat}", []).append(v)
            if r.get("maj_" + sub) is not None:
                d.setdefault("_maj", []).append(r["maj_" + sub])
            d.setdefault("_n_splits", set()).add(r["seed"])
    with open(args.out_dir / "retention_probe_results.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_RESULTS)
        w.writeheader()
        for (layer, sub) in sorted(wide):
            d = wide[(layer, sub)]
            out = {
                "layer": layer,
                "layer_name": d.get("layer_name",
                                    f"layer_{layer:02d}" if layer else "embedding"),
                "subset": sub,
                "n_splits": len(d.get("_n_splits", [])) or None,
                "acc_pre_mean": d.get("pre_mean"),
                "acc_post_mean": d.get("post_mean"),
                "drop_mean": d.get("drop_mean"),
                "bal_pre_mean": float(np.mean(d["_bal_pre"])) if d.get("_bal_pre") else None,
                "bal_post_mean": float(np.mean(d["_bal_post"])) if d.get("_bal_post") else None,
                "f1_pre_mean": float(np.mean(d["_f1_pre"])) if d.get("_f1_pre") else None,
                "f1_post_mean": float(np.mean(d["_f1_post"])) if d.get("_f1_post") else None,
                "maj_mean": float(np.mean(d["_maj"])) if d.get("_maj") else None,
                "p_one_layer_pre": d.get("p_one_layer_pre"),
                "p_one_layer_post": d.get("p_one_layer_post"),
                "p_one_layer_drop": d.get("p_one_layer_drop"),
                "p_maxT_pre": d.get("p_maxT_pre"),
                "p_maxT_post": d.get("p_maxT_post"),
                "p_maxT_drop": d.get("p_maxT_drop"),
                "null_mean": d.get("null_mean"),
                "null_p95": d.get("null_p95"),
            }
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in out.items()})

    # ---- canonical evidence (headline layer) -----------------------------
    canon = [s for s in sample_rows if s["layer"] == HEADLINE_LAYER
             and s["canonical_group"] in ("stale", "success", "other_failure")]
    beh = {p["pair_id"] if False else f"{p['trajectory_id']}_t{p['t']}": p
           for p in complete}
    with open(args.out_dir / "canonical_retention_evidence.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_CANONICAL)
        w.writeheader()
        for s in canon:
            p = beh[s["pair_id"]]
            w.writerow({
                "pair_id": s["pair_id"], "trajectory_id": s["trajectory_id"],
                "t": s["t"], "h_pre_key": s["h_pre_key"],
                "h_post_key": s["h_post_key"], "layer": s["layer"],
                "layer_name": s["layer_name"],
                "canonical_group": s["canonical_group"],
                "gt_prev_state": s["gt_prev_state"], "gt_state": s["gt_state"],
                "gt_event": s["gt_event"],
                "p_pre_gt": s["p_pre_gt"], "p_post_gt": s["p_post_gt"],
                "pre_margin": s["pre_margin"], "post_margin": s["post_margin"],
                "margin_drop": s["margin_drop"], "prob_drop": s["prob_drop"],
                "p_event_gt": s["p_event_gt"],
                "native_state_correct": p["state_correct"],
                "native_state_pred": p["state_pred"],
                "native_event_correct": p["event_correct"],
            })

    # ---- group contrasts --------------------------------------------------
    n_perms_contrast = 2000 if len(seeds) < 3 else 10000
    contrasts = {}
    for key in ("margin_drop", "prob_drop"):
        for g2 in ("success", "maintenance", "other_failure", "rest"):
            contrasts[f"{key}:stale_vs_{g2}"] = group_contrasts(
                sample_rows, HEADLINE_LAYER, key, "stale", g2,
                n_perms=n_perms_contrast)
    # layer profile of the mean retention drop (subset all)
    profile = []
    for layer in layers:
        vals = [s["margin_drop"] for s in sample_rows
                if s["layer"] == layer and s["margin_drop"] is not None]
        if vals:
            profile.append({"layer": layer,
                            "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                            "mean_margin_drop_all": float(np.mean(vals)),
                            "mean_prob_drop_all": float(np.mean(
                                [s["prob_drop"] for s in sample_rows
                                 if s["layer"] == layer and s["prob_drop"] is not None])),
                            "mean_cos_pre_post_all": float(np.mean(
                                [s["cos_pre_post"] for s in sample_rows
                                 if s["layer"] == layer]))})

    headline = {}
    for sub in SUBSETS:
        row = next((r for r in wide.values() if r["subset"] == sub
                    and r["layer"] == HEADLINE_LAYER), None)
        if row:
            headline[sub] = {
                "acc_pre": row.get("pre_mean"), "acc_post": row.get("post_mean"),
                "drop": row.get("drop_mean"),
                "p_one_layer_pre": row.get("p_one_layer_pre"),
                "p_one_layer_post": row.get("p_one_layer_post"),
                "p_maxT_pre": row.get("p_maxT_pre"),
                "p_maxT_post": row.get("p_maxT_post"),
                "p_maxT_drop": row.get("p_maxT_drop"),
            }

    summary = {
        "config": {
            "pairs": str(args.pairs), "npz": str(args.npz),
            "seeds": seeds, "layers": [layers[0], layers[-1], len(layers)],
            "train_frac": args.train_frac, "c_grid": c_grid,
            "n_perms": args.n_perms, "n_boot": args.n_boot,
            "headline_layer": HEADLINE_LAYER,
            "limit_trajectories": args.limit_trajectories,
        },
        "existing_hidden_states_sufficient": sufficient,
        "missing_t0_keys": sorted(set(missing_t0))[:3] + [
            f"... +{len(set(missing_t0)) - 3} more" if len(set(missing_t0)) > 3
            else ""] or None,
        "counts": {
            "pairs_total": len(pairs), "pairs_complete": len(complete),
            "trajectories": len(trajs),
            "per_subset": {name: sum(1 for p in complete
                                     if subset_mask(p, name))
                           for name in SUBSETS},
            "canonical_by_t": {str(t): dict(Counter(
                p["canonical_group"] for p in complete if p["t"] == t
                and p["canonical_group"])) for t in sorted(set(t_of))},
        },
        "note_success_group": ("all 17 aligned revision successes occur at t=1, "
                               "whose h_pre (t=0) is missing; the stale-vs-"
                               "success contrast therefore requires the t=0 "
                               "backfill. With the current data the stale "
                               "group (26 rows, t=2..5) can be compared "
                               "against maintenance/other groups only."),
        "headline_layer_results": headline,
        "group_contrasts_headline_layer": contrasts,
        "layer_profile_all": profile,
        "protocol": {
            "frozen_pre_decoder": ("D_pre^l trained ONLY on train-trajectory "
                                   "h_pre rows (target GT S_{t-1}), then the "
                                   "same frozen classifier+scaler evaluated "
                                   "on h_pre AND h_post of held-out "
                                   "trajectories; no post-trained classifier"),
            "retention_drop": "performance(h_post) - performance(h_pre)",
            "nulls": "within-t label permutations of the pre labels; refit "
                     "decoder evaluated on pre and post; paired drop",
            "maxT": "family-wise over 37 layers x seeds x perms",
            "event_decoder": "trained on train-trajectory h_post rows "
                             "(target GT E_t), evaluated on held-out h_post",
            "raw_drift_caveat": "cosine/L2 on raw hidden states is "
                                "descriptive only, NOT a state-drift claim; "
                                "use drift_proj / drift_logit_gt "
                                "(frozen-decoder subspace) for that",
        },
        "causal_patching_interface": {
            "status": "reserved; NO patching/steering/attention/MLP "
                      "intervention implemented or run this round",
            "addressable_hidden_states": str(args.npz),
            "fields_per_row": ["pair_id (trajectory_id_t)", "trajectory_id",
                               "t", "layer", "h_pre_key", "h_post_key",
                               "canonical_group", "pre_margin", "post_margin",
                               "prob_drop", "margin_drop"],
            "usage": "future activation replacement / steering experiments "
                     "can address h_pre_key / h_post_key tensors "
                     "[(layer, 4096)] in hidden_states.npz and use the "
                     "decoder margins as readouts",
        },
    }
    (args.out_dir / "retention_summary.json").write_text(
        json.dumps(summary, indent=2))

    write_report(args.out_dir, summary, headline, contrasts, profile)
    print(f"Wrote {args.out_dir}/retention_probe_results.csv "
          f"({len(wide)} layer-subset rows)")
    print(f"Wrote {args.out_dir}/retention_heldout_samples.csv "
          f"({len(sample_rows)} rows)")
    print(f"Wrote {args.out_dir}/canonical_retention_evidence.csv "
          f"({len(canon)} rows)")
    print(f"Wrote {args.out_dir}/retention_summary.json")
    print(f"Wrote {args.out_dir}/retention_report.md")
    print(f"EXISTING_HIDDEN_STATES_SUFFICIENT={str(sufficient).lower()}")
    if headline:
        for sub, h in headline.items():
            if h.get("acc_pre") is not None:
                print(f"  {sub}: pre={h['acc_pre']:.3f} post={h['acc_post']:.3f} "
                      f"drop={h['drop']:+.3f} (p_maxT drop={h['p_maxT_drop']})")


def write_report(out_dir: Path, summary: dict, headline: dict,
                 contrasts: dict, profile: list) -> None:
    L = []
    L.append("# StateRev-VL — Paired Pre/Post-Event State-Retention Analysis (v1)\n")
    L.append(f"Config: seeds={summary['config']['seeds']}, "
             f"layers={summary['config']['layers'][0]}.."
             f"{summary['config']['layers'][1]} ({summary['config']['layers'][2]}), "
             f"perms={summary['config']['n_perms']}, "
             f"limit_trajectories={summary['config']['limit_trajectories']}.\n")
    suff = summary["existing_hidden_states_sufficient"]
    L.append(f"**EXISTING_HIDDEN_STATES_SUFFICIENT={str(suff).lower()}** — "
             + ("all t=0 pre-states present." if suff else
                "t=0 hidden states (h_pre of t=1 pairs) are missing; backfill "
                "with `python scripts/run_hidden_state_probe.py --include-t0`."))
    L.append("\n## Data\n")
    c = summary["counts"]
    L.append(f"- pairs: {c['pairs_complete']}/{c['pairs_total']} complete "
             f"({c['trajectories']} trajectories)")
    L.append(f"- subsets: {c['per_subset']}")
    L.append(f"- canonical by t: {c['canonical_by_t']}")
    L.append(f"- {summary['note_success_group']}\n")
    L.append("\n## Frozen pre-event decoder, headline layer (36)\n")
    L.append("| subset | acc pre | acc post | drop | p1 pre | p1 post | maxT pre | maxT post | maxT drop |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for sub, h in headline.items():
        if h.get("acc_pre") is None:
            L.append(f"| {sub} | (empty) | | | | | | | |")
            continue
        L.append(f"| {sub} | {h['acc_pre']:.3f} | {h['acc_post']:.3f} | "
                 f"{h['drop']:+.3f} | {h['p_one_layer_pre']} | "
                 f"{h['p_one_layer_post']} | {h['p_maxT_pre']} | "
                 f"{h['p_maxT_post']} | {h['p_maxT_drop']} |")
    L.append("\n## Group contrasts (headline layer)\n")
    for k, v in contrasts.items():
        L.append(f"- `{k}`: {json.dumps(v, default=str)}")
    L.append("\n## Layer profile (subset all)\n")
    L.append("| layer | mean margin drop | mean prob drop | mean cos(pre,post) |")
    L.append("|---|---|---|---|")
    for p in profile:
        L.append(f"| {p['layer_name']} | {p['mean_margin_drop_all']:+.3f} | "
                 f"{p['mean_prob_drop_all']:+.3f} | "
                 f"{p['mean_cos_pre_post_all']:.3f} |")
    L.append("\n## Protocol notes\n")
    for k, v in summary["protocol"].items():
        L.append(f"- **{k}**: {v}")
    L.append("\n## Causal-patching interface\n")
    L.append(json.dumps(summary["causal_patching_interface"], indent=2))
    L.append("\n*Interpretation is deferred to the full run; this file is "
             "auto-generated from retention_summary.json.*")
    (out_dir / "retention_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
