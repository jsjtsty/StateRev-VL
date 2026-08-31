"""Experiment 1 — Nonlinear frozen pre-decoder for stale-origin analysis.

Question
--------
The linear frozen pre-decoder (state_retention_analysis.py) showed that the
pre-event state S_{t-1} is linearly readable before the event and linearly
unreadable (at chance) after it. This script asks whether the OLD state
survives the event in a NONLINEAR / distributed form:

    Does a small MLP, trained ONLY on train-trajectory h_pre rows (target
    GT S_{t-1}), decode S_{t-1} better from h_post than the linear decoder
    does, relative to its own h_pre readout?

Protocol (same leak-free structure as the linear analysis)
----------------------------------------------------------
Per (seed, layer):
  1. trajectory group split (train_frac, no overlap);
  2. StandardScaler (+ optional PCA) fit ONLY on train h_pre;
  3. linear baseline: L2 logistic regression (C by 3-fold group CV inside
     train) - identical protocol to state_retention_analysis.py;
  4. small MLP: 4096 -> hidden(64/128) -> 3, ReLU + dropout, AdamW with
     strong weight decay, early stopping on a validation split carved from
     TRAIN trajectories only (val never touches the test trajectories);
  5. both decoders are FROZEN and evaluated on held-out h_pre AND h_post;
     retention_drop = perf(h_post) - perf(h_pre);
  6. within-t label-permutation nulls: refit each decoder with permuted
     pre labels, evaluate pre/post (paired drop);
  7. family-wise maxT over layers (one-sided LOW tail for drop).

Interpretation guard
--------------------
An MLP that decodes S_{t-1} from h_post only shows the information is
PRESENT in a nonlinear/distributed form. It does NOT show that the model
uses that information at decision time (that requires the causal /
generation experiments, Experiments 2-3).

Outputs (default outputs/vetbench/stale_origin_analysis_v1/):
    nonlinear_retention_results.csv   per (arch, layer, subset)
    nonlinear_retention_samples.csv   per (arch, pair, layer)
    nonlinear_retention_summary.json

Run (CPU only; hidden states pre-extracted):
    python scripts/nonlinear_state_retention.py
Smoke (tiny, no GPU, few jobs):
    python scripts/nonlinear_state_retention.py \
        --limit-trajectories 4 --seeds 0 --layers 5,17,36 \
        --n-perms 5 --n-jobs 2 --out-dir /tmp/opencode/nl_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR
from run_hidden_state_probe import STATE_INDEX, _pad_proba, group_split
from state_retention_analysis import (
    HIDDEN_SIZE,
    N_LAYERS,
    _margins,
    _pval,
    balanced_acc,
    load_pairs,
    macro_f1,
)
from strict_probe_analysis import _fit_lr2, _select_c2

ROOT = Path(__file__).resolve().parent.parent
RET_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "state_retention_analysis_v1"
PROBE_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "hidden_state_probe"
OUT_DIR_DEFAULT = DEFAULT_OUTPUT_DIR / "vetbench" / "stale_origin_analysis_v1"

SUBSETS = ("all", "clean_revision", "rev_stale", "clean_maintenance",
           "rest_t2_5")
K = 3
ARCHS = ("linear", "mlp")
HEADLINE_LAYER = 36


def subset_mask(r: dict, name: str) -> bool:
    if name == "all":
        return True
    if name == "clean_revision":
        return bool(r["clean_revision"])
    if name == "rev_stale":
        return bool(r["canonical_stale_failure"])
    if name == "clean_maintenance":
        return bool(r["clean_maintenance"])
    if name == "rest_t2_5":
        # same-step (t=2..5) non-canonical control for the stale group
        return (r["t"] in (2, 3, 4, 5)
                and not r["clean_revision"] and not r["clean_maintenance"])
    raise KeyError(name)


def npz_keys(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as z:
        return set(z.files)


def load_hidden(pairs: list[dict], path: Path
                ) -> tuple[np.ndarray, np.ndarray]:
    need = {p["h_pre_key"] for p in pairs} | {p["h_post_key"] for p in pairs}
    with np.load(path, allow_pickle=False) as z:
        missing = need - set(z.files)
        if missing:
            raise SystemExit(f"hidden states missing for {len(missing)} keys "
                             f"(e.g. {sorted(missing)[:3]})")
        Xp = np.zeros((len(pairs), N_LAYERS, HIDDEN_SIZE), dtype=np.float32)
        Xq = np.zeros_like(Xp)
        for i, p in enumerate(pairs):
            Xp[i] = np.asarray(z[p["h_pre_key"]], dtype=np.float32)
            Xq[i] = np.asarray(z[p["h_post_key"]], dtype=np.float32)
    return Xp, Xq


# ---------------------------------------------------------------------------
# Small MLP (module-level so it survives process-pool pickling under spawn)
# ---------------------------------------------------------------------------

def _make_mlp(dim: int, hidden: int, dropout: float):
    import torch.nn as nn
    F = nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(dim, hidden)
            self.l2 = nn.Linear(hidden, K)
            self.p = float(dropout)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.dropout(x, p=self.p, training=self.training)
            return self.l2(x)

    return Net()


def _train_mlp(Xtr: np.ndarray, ytr: np.ndarray,
               Xval: np.ndarray | None, yval: np.ndarray | None,
               hidden: int, dropout: float, lr: float, weight_decay: float,
               max_epochs: int, patience: int, batch: int, seed: int
               ) -> Any:
    import torch

    torch.manual_seed(seed)
    net = _make_mlp(Xtr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.int64))
    use_val = (Xval is not None and len(np.unique(yval)) >= 2
               and len(yval) >= 2)
    if use_val:
        Xv = torch.from_numpy(np.ascontiguousarray(Xval, dtype=np.float32))
        yv = torch.from_numpy(np.ascontiguousarray(yval, dtype=np.int64))

    def _val_bacc() -> float:
        net.eval()
        with torch.no_grad():
            pv = net(Xv).argmax(1).numpy()
        net.train()
        return balanced_acc(yval, pv)

    best_w, best_acc, bad = None, -1.0, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xt),
                              generator=torch.Generator().manual_seed(
                                  seed * 100003 + ep))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
        if use_val:
            acc = _val_bacc()
            if acc > best_acc + 1e-4:
                best_acc, bad = acc, 0
                best_w = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_w is not None:
        net.load_state_dict(best_w)
    return net


def _mlp_proba(net: Any, X: np.ndarray) -> np.ndarray:
    import torch

    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        p = torch.softmax(net(x), dim=1)
    return p.numpy()


# ---------------------------------------------------------------------------
# Worker: one (seed, layer) -> both archs
# ---------------------------------------------------------------------------

def _job_nl(job: dict) -> dict:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _job_nl_body(job)


def _subset_evals(proba_pre: np.ndarray, proba_post: np.ndarray,
                  pred_pre: np.ndarray, pred_post: np.ndarray,
                  yt_all: np.ndarray, rows_te: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for name in SUBSETS:
        m = np.array([subset_mask(rows_te[i], name)
                      for i in range(len(rows_te))])
        if m.sum() < 2:
            continue
        yt = yt_all[m]
        out[name] = {
            "acc_pre": float((pred_pre[m] == yt).mean()),
            "acc_post": float((pred_post[m] == yt).mean()),
            "bal_pre": balanced_acc(yt, pred_pre[m]),
            "bal_post": balanced_acc(yt, pred_post[m]),
            "f1_pre": macro_f1(yt, pred_pre[m], K),
            "f1_post": macro_f1(yt, pred_post[m], K),
            "maj": float(np.bincount(yt, minlength=K).max() / m.sum()),
        }
    return out


def _null_subset(proba: np.ndarray, rows_te: list[dict],
                 yt_all: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    out = {}
    for name in SUBSETS:
        m = np.array([subset_mask(rows_te[i], name)
                      for i in range(len(rows_te))])
        if m.sum() < 2:
            continue
        yt = yt_all[m]
        out[name] = float((pred[m] == yt).mean())
    return out


def _job_nl_body(job: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from sklearn.preprocessing import StandardScaler

    Xp_all = np.load(job["x_pre_path"], mmap_mode="r")
    Xq_all = np.load(job["x_post_path"], mmap_mode="r")
    L = job["layer"]
    tr, te = job["train_rows"], job["test_rows"]
    Xtr_pre = np.asarray(Xp_all[tr, L, :], dtype=np.float32)
    Xte_pre = np.asarray(Xp_all[te, L, :], dtype=np.float32)
    Xte_post = np.asarray(Xq_all[te, L, :], dtype=np.float32)
    y_pre_tr = np.asarray(job["y_pre"][tr])
    if len(tr) < 2 or len(te) < 2 or len(set(y_pre_tr.tolist())) < 2:
        return {"kind": "nl", "empty": True, "seed": job["seed"], "layer": L}

    sc = StandardScaler().fit(Xtr_pre)
    Xtr_s = sc.transform(Xtr_pre)
    Xte_pre_s = sc.transform(Xte_pre)
    Xte_post_s = sc.transform(Xte_post)
    if job["pca"] > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(job["pca"], Xtr_s.shape[1])).fit(Xtr_s)
        Xtr_s = pca.transform(Xtr_s)
        Xte_pre_s = pca.transform(Xte_pre_s)
        Xte_post_s = pca.transform(Xte_post_s)

    trajs_tr = [job["traj_of"][i] for i in tr]
    t_tr = np.array([job["t_of"][i] for i in tr])
    yt_all = np.array([job["gt_prev_te"][i] for i in range(len(te))])
    rows_te = job["rows_te"]
    gt_idx = yt_all

    # validation trajectories carved from TRAIN (early stopping only)
    rng_val = random.Random(f"nl-val-{job['seed']}-{L}")
    uniq = sorted(set(trajs_tr))
    rng_val.shuffle(uniq)
    n_val = max(1, int(round(0.2 * len(uniq))))
    val_trajs = set(uniq[:n_val])
    val_mask = np.array([job["traj_of"][i] in val_trajs for i in tr])
    fit_idx = np.flatnonzero(~val_mask)

    out: dict[str, Any] = {
        "kind": "nl", "empty": False, "seed": job["seed"], "layer": L,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_val": int(val_mask.sum()),
        "test_index": job["test_ids"],
        "arch": {}, "nulls": {a: {} for a in ARCHS},
    }

    # ---- linear baseline ---------------------------------------------------
    c_lin = _select_c2(Xtr_s, y_pre_tr, trajs_tr, job["c_grid"],
                       job["c_key"] + "-lin")
    clf_lin = _fit_lr2(Xtr_s, y_pre_tr, c_lin)
    proba_l_pre = _pad_proba(clf_lin.predict_proba(Xte_pre_s),
                             clf_lin.classes_, K)
    proba_l_post = _pad_proba(clf_lin.predict_proba(Xte_post_s),
                              clf_lin.classes_, K)
    out["arch"]["linear"] = {
        "C": float(c_lin),
        "samples": {
            "p_pre_gt": proba_l_pre[np.arange(len(te)), gt_idx].tolist(),
            "p_post_gt": proba_l_post[np.arange(len(te)), gt_idx].tolist(),
            "pre_margin": _margins(proba_l_pre, gt_idx).tolist(),
            "post_margin": _margins(proba_l_post, gt_idx).tolist(),
            "pred_pre": proba_l_pre.argmax(1).tolist(),
            "pred_post": proba_l_post.argmax(1).tolist(),
        },
        "subsets": _subset_evals(proba_l_pre, proba_l_post,
                                 proba_l_pre.argmax(1),
                                 proba_l_post.argmax(1), yt_all, rows_te),
    }

    # ---- small MLP ---------------------------------------------------------
    mlp_seed = job["seed"] * 1000003 + L
    net = _train_mlp(Xtr_s[fit_idx], y_pre_tr[fit_idx],
                     Xtr_s[val_mask], y_pre_tr[val_mask],
                     job["hidden"], job["dropout"], job["lr"],
                     job["weight_decay"], job["max_epochs"],
                     job["patience"], job["batch"], mlp_seed)
    proba_m_pre = _mlp_proba(net, Xte_pre_s)
    proba_m_post = _mlp_proba(net, Xte_post_s)
    out["arch"]["mlp"] = {
        "hidden": int(job["hidden"]),
        "samples": {
            "p_pre_gt": proba_m_pre[np.arange(len(te)), gt_idx].tolist(),
            "p_post_gt": proba_m_post[np.arange(len(te)), gt_idx].tolist(),
            "pre_margin": _margins(proba_m_pre, gt_idx).tolist(),
            "post_margin": _margins(proba_m_post, gt_idx).tolist(),
            "pred_pre": proba_m_pre.argmax(1).tolist(),
            "pred_post": proba_m_post.argmax(1).tolist(),
        },
        "subsets": _subset_evals(proba_m_pre, proba_m_post,
                                 proba_m_pre.argmax(1),
                                 proba_m_post.argmax(1), yt_all, rows_te),
    }

    # ---- within-t label-permutation nulls (per arch) ------------------------
    # nulls[arch] = {"pre": {sub: [val per perm]}, "post": {...},
    #                "drop_rows": [(ppre_row, ppost_row) per perm]}
    # all three structures are aligned by perm index (skipped perms are
    # skipped identically for every arch/subset within a job).
    out["nulls"] = {a: {"pre": {}, "post": {}, "drop_rows": []}
                    for a in ARCHS}
    rng = random.Random(job["rng_key"])
    for b in range(job["n_perms"]):
        y_perm = y_pre_tr.copy()
        for t in np.unique(t_tr):
            sel = np.flatnonzero(t_tr == t)
            y_perm[sel] = y_perm[sel][rng.sample(range(len(sel)), len(sel))]
        if len(set(y_perm.tolist())) < 2:
            continue
        # linear null (refit at the selected C, same protocol)
        clf_n = _fit_lr2(Xtr_s, y_perm, c_lin)
        pn_l_pre = _pad_proba(clf_n.predict_proba(Xte_pre_s),
                              clf_n.classes_, K)
        pn_l_post = _pad_proba(clf_n.predict_proba(Xte_post_s),
                               clf_n.classes_, K)
        nl = out["nulls"]["linear"]
        nl["drop_rows"].append(
            (pn_l_pre[np.arange(len(te)), gt_idx],
             pn_l_post[np.arange(len(te)), gt_idx]))
        for sub, v in _null_subset(pn_l_pre, rows_te, yt_all).items():
            nl["pre"].setdefault(sub, []).append(v)
        for sub, v in _null_subset(pn_l_post, rows_te, yt_all).items():
            nl["post"].setdefault(sub, []).append(v)
        # mlp null (fixed budget, no early stopping)
        net_n = _train_mlp(Xtr_s, y_perm, None, None,
                           job["hidden"], job["dropout"], job["lr"],
                           job["weight_decay"], job["null_epochs"],
                           patience=10 ** 9, batch=job["batch"],
                           seed=mlp_seed + b)
        pn_m_pre = _mlp_proba(net_n, Xte_pre_s)
        pn_m_post = _mlp_proba(net_n, Xte_post_s)
        nm = out["nulls"]["mlp"]
        nm["drop_rows"].append(
            (pn_m_pre[np.arange(len(te)), gt_idx],
             pn_m_post[np.arange(len(te)), gt_idx]))
        for sub, v in _null_subset(pn_m_pre, rows_te, yt_all).items():
            nm["pre"].setdefault(sub, []).append(v)
        for sub, v in _null_subset(pn_m_post, rows_te, yt_all).items():
            nm["post"].setdefault(sub, []).append(v)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _per_subset_drop_nulls(drop_pairs: list[tuple[np.ndarray, np.ndarray]],
                           rows_te: list[dict]) -> dict[str, list[float]]:
    """Per-subset paired drop for each null permutation."""
    out: dict[str, list[float]] = defaultdict(list)
    for ppre, ppost in drop_pairs:
        for name in SUBSETS:
            m = np.array([subset_mask(rows_te[i], name)
                          for i in range(len(rows_te))])
            if m.sum() < 2:
                continue
            out[name].append(float((ppost[m] - ppre[m]).mean()))
    return dict(out)


def aggregate(results: list[dict], layers: list[int],
              rows_te_by_seed_layer: dict[tuple[int, int], list[dict]]
              ) -> list[dict]:
    """Per (arch, layer, subset, stat): seed-mean, one-layer p (correct tail),
    family-wise maxT (high tail for pre/post, LOW tail for drop)."""
    out_rows: list[dict] = []
    for arch in ARCHS:
        per_seed: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
        null_pool: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        # per (seed, layer): per-subset per-perm drop null scalars
        drop_perm: list[tuple[int, int, dict[str, list[float]]]] = []
        for r in results:
            if r.get("empty") or arch not in r.get("arch", {}):
                continue
            a = r["arch"][arch]
            for sub, m in a["subsets"].items():
                per_seed[(r["seed"], r["layer"])].update({
                    f"pre_{sub}": m["acc_pre"],
                    f"post_{sub}": m["acc_post"],
                    f"drop_{sub}": m["acc_post"] - m["acc_pre"],
                    f"balpre_{sub}": m["bal_pre"],
                    f"balpost_{sub}": m["bal_post"],
                    f"f1pre_{sub}": m["f1_pre"],
                    f"f1post_{sub}": m["f1_post"],
                    f"maj_{sub}": m["maj"],
                })
            rows_te = rows_te_by_seed_layer.get((r["seed"], r["layer"]))
            if rows_te is None:
                continue
            nl = r.get("nulls", {}).get(arch)
            if not nl:
                continue
            for stat in ("pre", "post"):
                for sub, vals in nl.get(stat, {}).items():
                    null_pool[(stat, r["layer"], sub)].extend(vals)
            nd = _per_subset_drop_nulls(nl.get("drop_rows", []), rows_te)
            for sub, vals in nd.items():
                null_pool[("drop", r["layer"], sub)].extend(vals)
            drop_perm.append((r["seed"], r["layer"], nd))

        for layer in layers:
            for sub in SUBSETS:
                for stat in ("pre", "post", "drop"):
                    key = f"{stat}_{sub}"
                    ps = [per_seed[(s, layer)][key]
                          for s in sorted({k[0] for k in per_seed})
                          if key in per_seed.get((s, layer), {})]
                    if not ps:
                        continue
                    mean = float(np.mean(ps))
                    nulls = null_pool.get((stat, layer, sub), [])
                    if stat == "drop":
                        p1 = (float(np.mean(np.asarray(nulls) <= mean))
                              if nulls else None)
                    else:
                        p1 = _pval(mean, nulls) if nulls else None
                    row: dict[str, Any] = {
                        "arch": arch, "layer": layer,
                        "layer_name": (f"layer_{layer:02d}"
                                       if layer else "embedding"),
                        "subset": sub, "stat": stat, "n_splits": len(ps),
                        "mean": mean, "p_one_layer": p1,
                        "null_mean": (float(np.mean(nulls))
                                      if nulls else None),
                        "null_p95": (float(np.percentile(nulls, 95))
                                     if nulls else None)}
                    if stat in ("pre", "post"):
                        bcol = f"bal{stat}_{sub}"
                        fcol = f"f1{stat}_{sub}"
                        ss = sorted({k[0] for k in per_seed})
                        row[f"bal_{stat}_mean"] = float(np.mean(
                            [per_seed[(s, layer)][bcol] for s in ss
                             if bcol in per_seed.get((s, layer), {})]))
                        row[f"f1_{stat}_mean"] = float(np.mean(
                            [per_seed[(s, layer)][fcol] for s in ss
                             if fcol in per_seed.get((s, layer), {})]))
                        row["maj_mean"] = float(np.mean(
                            [per_seed[(s, layer)][f"maj_{sub}"] for s in ss
                             if f"maj_{sub}" in per_seed.get((s, layer), {})]))
                    out_rows.append(row)

        # family-wise maxT per (subset, stat)
        for sub in SUBSETS:
            for stat in ("pre", "post", "drop"):
                layers_with = [row["layer"] for row in out_rows
                               if row["arch"] == arch and row["subset"] == sub
                               and row["stat"] == stat]
                if not layers_with:
                    continue
                extreme = min if stat == "drop" else max
                per_perm: dict[tuple[int, int], dict[int, float]] = {}
                for r in results:
                    if r.get("empty") or arch not in r.get("arch", {}):
                        continue
                    nl = r.get("nulls", {}).get(arch)
                    if not nl:
                        continue
                    rows_te = rows_te_by_seed_layer.get(
                        (r["seed"], r["layer"]))
                    if rows_te is None:
                        continue
                    if stat == "drop":
                        nd = _per_subset_drop_nulls(
                            nl.get("drop_rows", []), rows_te)
                        vals = nd.get(sub)
                    else:
                        vals = nl.get(stat, {}).get(sub)
                    if not vals:
                        continue
                    for b, v in enumerate(vals):
                        per_perm.setdefault((r["seed"], b), {})[
                            r["layer"]] = v
                maxts = [extreme(v.values()) for v in per_perm.values()
                         if set(v) == set(layers_with)]
                for row in out_rows:
                    if row["arch"] != arch or row["subset"] != sub \
                            or row["stat"] != stat:
                        continue
                    if not maxts:
                        row["p_maxT"] = None
                    elif stat == "drop":
                        row["p_maxT"] = float(
                            np.mean(np.asarray(maxts) <= row["mean"]))
                    else:
                        row["p_maxT"] = float(
                            np.mean(np.asarray(maxts) >= row["mean"]))
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path,
                    default=RET_DIR / "state_retention_pairs.csv")
    ap.add_argument("--npz", type=Path,
                    default=PROBE_DIR / "hidden_states.npz")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--layers", type=str, default="all")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--null-epochs", type=int, default=40)
    ap.add_argument("--pca", type=int, default=0,
                    help="PCA components fit on train h_pre (0 = off)")
    ap.add_argument("--c-grid", type=str, default="0.1,1.0,10.0")
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--limit-trajectories", type=int, default=0)
    ap.add_argument("--skip-jobs", action="store_true")
    ap.add_argument("--cache-pkl", type=Path, default=None)
    args = ap.parse_args()
    if args.cache_pkl is None:
        args.cache_pkl = args.out_dir / "nonlinear_results.pkl"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    layers = (list(range(N_LAYERS)) if args.layers == "all"
              else [int(x) for x in args.layers.split(",") if x.strip() != ""])
    c_grid = [float(c) for c in args.c_grid.split(",") if c.strip() != ""]

    pairs = load_pairs(args.pairs)
    keys = npz_keys(args.npz)
    available = [p for p in pairs
                 if p["h_pre_key"] in keys and p["h_post_key"] in keys]
    if args.limit_trajectories:
        keep = sorted({p["trajectory_id"] for p in available}
                      )[: args.limit_trajectories]
        available = [p for p in available
                     if p["trajectory_id"] in keep]
    if not available:
        raise SystemExit("no pairs with both hidden-state keys present")
    available = sorted(available, key=lambda p: (p["trajectory_id"], p["t"]))
    trajs = sorted({p["trajectory_id"] for p in available})
    traj_of = [p["trajectory_id"] for p in available]
    t_of = [p["t"] for p in available]
    y_pre = np.array([STATE_INDEX[p["gt_prev_state"]] for p in available])
    gt_prev = [STATE_INDEX[p["gt_prev_state"]] for p in available]
    pair_ids = [f"{p['trajectory_id']}_t{p['t']}" for p in available]

    print(f"Pairs available (both keys in npz): {len(available)} "
          f"of {len(pairs)} across {len(trajs)} trajectories; "
          f"t values: {sorted(set(t_of))}")
    print(f"Seeds: {seeds}; layers: {layers[0]}..{layers[-1]} "
          f"({len(layers)}); perms: {args.n_perms}")
    for name in SUBSETS:
        print(f"  subset {name}: "
              f"{sum(1 for p in available if subset_mask(p, name))}")

    rows_te_by_seed_layer: dict[tuple[int, int], list[dict]] = {}
    if not args.skip_jobs:
        Xp, Xq = load_hidden(available, args.npz)
        tmp = Path(tempfile.mkdtemp(prefix="nl_X_"))
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
            rows_te = [available[i] for i in te_rows]
            for layer in layers:
                jobs.append({
                    "seed": seed, "layer": layer,
                    "x_pre_path": str(x_pre_file),
                    "x_post_path": str(x_post_file),
                    "train_rows": tr_rows, "test_rows": te_rows,
                    "y_pre": y_pre, "traj_of": traj_of, "t_of": t_of,
                    "gt_prev_te": [gt_prev[i] for i in te_rows],
                    "rows_te": rows_te,
                    "test_ids": [pair_ids[i] for i in te_rows],
                    "c_grid": c_grid, "c_key": f"nl-{seed}-{layer}",
                    "rng_key": f"nl-null-{seed}-{layer}",
                    "n_perms": args.n_perms,
                    "hidden": args.hidden, "dropout": args.dropout,
                    "lr": args.lr, "weight_decay": args.weight_decay,
                    "batch": args.batch, "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "null_epochs": args.null_epochs, "pca": args.pca,
                })
                rows_te_by_seed_layer[(seed, layer)] = rows_te
        print(f"Running {len(jobs)} jobs with {args.n_jobs} workers...")
        t0 = time.time()
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for i, r in enumerate(ex.map(_job_nl, jobs, chunksize=1)):
                results.append(r)
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    print(f"jobs done: {i + 1}/{len(jobs)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(args.cache_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"Cached {len(results)} results to {args.cache_pkl}")
    else:
        import pickle
        with open(args.cache_pkl, "rb") as f:
            results = pickle.load(f)
        for seed in seeds:
            tr_set, te_set = group_split(trajs, seed, args.train_frac)
            te_rows = [i for i, t in enumerate(traj_of) if t in te_set]
            for layer in layers:
                rows_te_by_seed_layer[(seed, layer)] = \
                    [available[i] for i in te_rows]

    # ---- per (arch, pair, layer) sample table ------------------------------
    sample_vals: dict[str, list[tuple[int, ...]]] = {}
    for r in results:
        if r.get("empty"):
            continue
        L = r["layer"]
        for arch in ARCHS:
            a = r.get("arch", {}).get(arch)
            if not a:
                continue
            for j, pid in enumerate(r["test_index"]):
                i = pair_ids.index(pid)
                for name in ("p_pre_gt", "p_post_gt", "pre_margin",
                             "post_margin"):
                    v = a["samples"][name][j]
                    sample_vals.setdefault(f"{arch}|{name}|{L}",
                                           []).append((i, v))

    sample_rows = []
    for layer in layers:
        for i, p in enumerate(available):
            base = {
                "pair_id": pair_ids[i],
                "trajectory_id": p["trajectory_id"], "t": p["t"],
                "h_pre_key": p["h_pre_key"], "h_post_key": p["h_post_key"],
                "layer": layer,
                "layer_name": f"layer_{layer:02d}" if layer else "embedding",
                "canonical_group": p["canonical_group"] or "none",
                "clean_maintenance": p["clean_maintenance"],
                "gt_prev_state": p["gt_prev_state"],
                "gt_state": p["gt_state"], "gt_event": p["gt_event"],
            }
            for arch in ARCHS:
                def avg(key, arch=arch, layer=layer):
                    vals = [v for (ii, v)
                            in sample_vals.get(f"{arch}|{key}|{layer}", [])
                            if ii == i]
                    return float(np.mean(vals)) if vals else None
                row = dict(base)
                row["arch"] = arch
                row["p_pre_gt"] = avg("p_pre_gt")
                row["p_post_gt"] = avg("p_post_gt")
                row["pre_margin"] = avg("pre_margin")
                row["post_margin"] = avg("post_margin")
                row["prob_drop"] = (row["p_post_gt"] - row["p_pre_gt"]
                                    if None not in (row["p_pre_gt"],
                                                    row["p_post_gt"])
                                    else None)
                row["margin_drop"] = (row["post_margin"] - row["pre_margin"]
                                      if None not in (row["pre_margin"],
                                                      row["post_margin"])
                                      else None)
                sample_rows.append(row)

    # ---- aggregate + write --------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    agg_rows = aggregate(results, layers, rows_te_by_seed_layer)

    res_cols = ["arch", "layer", "layer_name", "subset", "stat", "n_splits",
                "mean", "p_one_layer", "p_maxT", "null_mean", "null_p95",
                "bal_pre_mean", "bal_post_mean",
                "f1_pre_mean", "f1_post_mean", "maj_mean"]
    with open(args.out_dir / "nonlinear_retention_results.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(res_cols)
        for row in agg_rows:
            w.writerow([row.get(c, "") for c in res_cols])
    print(f"Wrote {args.out_dir / 'nonlinear_retention_results.csv'} "
          f"({len(agg_rows)} rows)")

    with open(args.out_dir / "nonlinear_retention_samples.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader()
        for row in sample_rows:
            w.writerow({k: ("" if v is None else v)
                        for k, v in row.items()})
    print(f"Wrote {args.out_dir / 'nonlinear_retention_samples.csv'} "
          f"({len(sample_rows)} rows)")

    headline = {}
    for arch in ARCHS:
        headline[arch] = {}
        for sub in SUBSETS:
            def _pick(stat, arch=arch, sub=sub):
                return next((r for r in agg_rows
                             if r["arch"] == arch and r["subset"] == sub
                             and r["stat"] == stat
                             and r["layer"] == HEADLINE_LAYER), None)
            pre, post, drop = _pick("pre"), _pick("post"), _pick("drop")
            headline[arch][sub] = {
                "acc_pre": pre["mean"] if pre else None,
                "acc_post": post["mean"] if post else None,
                "drop": drop["mean"] if drop else None,
                "p_maxT_pre": pre.get("p_maxT") if pre else None,
                "p_maxT_post": post.get("p_maxT") if post else None,
                "p_maxT_drop": drop.get("p_maxT") if drop else None,
            }

    summary = {
        "config": {
            "pairs": str(args.pairs), "npz": str(args.npz),
            "seeds": seeds, "layers": [layers[0], layers[-1], len(layers)],
            "train_frac": args.train_frac, "hidden": args.hidden,
            "dropout": args.dropout, "lr": args.lr,
            "weight_decay": args.weight_decay, "batch": args.batch,
            "max_epochs": args.max_epochs, "patience": args.patience,
            "null_epochs": args.null_epochs, "pca": args.pca,
            "c_grid": c_grid, "n_perms": args.n_perms,
            "limit_trajectories": args.limit_trajectories,
            "headline_layer": HEADLINE_LAYER,
        },
        "counts": {
            "pairs_total": len(pairs),
            "pairs_available": len(available),
            "trajectories": len(trajs),
            "per_subset": {n: sum(1 for p in available
                                  if subset_mask(p, n)) for n in SUBSETS},
        },
        "headline_layer_results": headline,
        "protocol": {
            "frozen_pre_decoder": (
                "linear (L2 LR, group-CV C) and small MLP "
                "([4096->hidden->3], AdamW strong weight decay, dropout, "
                "early stopping on a val split carved from TRAIN trajectories) "
                "trained ONLY on train h_pre rows (target GT S_{t-1}); the "
                "same frozen decoder is evaluated on held-out h_pre AND "
                "h_post; scaler/PCA fit on train h_pre only"),
            "retention_drop": "performance(h_post) - performance(h_pre)",
            "nulls": "within-t label permutations, refit, evaluated pre/post "
                     "(paired drop)",
            "maxT": "family-wise over layers; one-sided LOW tail for drop",
            "leakage": "test trajectories never used in fitting, CV, "
                       "early stopping or scaling",
        },
        "interpretation_guard": (
            "an MLP that decodes S_{t-1} from h_post only shows the "
            "information is present in a nonlinear/distributed form; it does "
            "NOT show the model uses it at decision time (see "
            "Experiments 2-3 for causal/generation tests)"),
    }
    with open(args.out_dir / "nonlinear_retention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_dir / 'nonlinear_retention_summary.json'}")

    for arch in ARCHS:
        h = headline[arch].get("all", {})
        print(f"  [{arch}] all: pre={h.get('acc_pre')} "
              f"post={h.get('acc_post')} drop={h.get('drop')} "
              f"(p_maxT pre/post/drop = {h.get('p_maxT_pre')}/"
              f"{h.get('p_maxT_post')}/{h.get('p_maxT_drop')})")


if __name__ == "__main__":
    main()
