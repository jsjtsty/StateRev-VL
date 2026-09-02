#!/usr/bin/env python3
"""Validity Gate Stage 2b: offline same-input probe (CPU, multiprocess).

Probes the STATE-question hidden states (from vg_same_input_factorial.py)
for THREE targets, per regime (2fps / 8fps):
  - E_t       : the current event (which two cups swapped at step t)
  - S_t       : the current state (where the ball is after step t)
  - S_{t-1}   : the previous state (where the ball was before step t)

The event probe uses the STATE-question forward's hidden states (NOT the
event-question forward) - this is the whole point of the same-input design.

Protocol (pre-registered):
  - 10 trajectory group splits (40 train / 10 test), fixed seed.
  - StandardScaler fit on TRAIN h only.
  - L2 logistic regression; C in {0.1,1,10} by 3-fold group CV inside train.
  - Metrics: balanced accuracy (primary), macro-F1, accuracy, majority base.
  - Subsets: all (250), true_transition (is_transition=1).
  - Significance: within-t label-permutation null (N_PERM), per
    (target, subset, layer) on a reference split; p = frac(null >= obs).
    maxT/FWER across the 37 layers: on the reference split, N_MAXT
    permutations, each computing the max balanced-accuracy over layers;
    p_maxT = frac(null_max >= obs_max).
  - Point estimate = 10-split mean per layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

POS = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
TARGETS = ("event", "state", "prev_state")
SUBSETS = ("all", "true_transition")
C_GRID = (0.1, 1.0, 10.0)
N_SPLITS, SPLIT_SEED = 10, 20260902
N_PERM, N_MAXT, MAXT_SEED = 200, 200, 7
N_LAYERS = 37


def T(x) -> bool:
    return str(x).strip().lower() == "true"


def encode(labels, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[l] for l in labels])


def balanced_acc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def select_c(X, y, trajs, c_grid, seed_key):
    from sklearn.linear_model import LogisticRegression
    rng = random.Random(seed_key)
    order = sorted(set(trajs))
    rng.shuffle(order)
    n_folds = min(3, len(order))
    folds = [set(order[i * len(order) // n_folds:
                   (i + 1) * len(order) // n_folds]) for i in range(n_folds)]
    best_c, best = 1.0, -1.0
    for C in c_grid:
        sc = _scaler_fit(X)
        Xs = sc.transform(X)
        scores = []
        for hold in folds:
            te = [j for j, tr in enumerate(trajs) if tr in hold]
            tr = [j for j, tr in enumerate(trajs) if tr not in hold]
            if not te or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            clf = LogisticRegression(C=C, max_iter=1000).fit(Xs[tr], y[tr])
            scores.append(float((clf.predict(Xs[te]) == y[te]).mean()))
        if scores and float(np.mean(scores)) > best + 1e-12:
            best, best_c = float(np.mean(scores)), C
    return best_c


def _scaler_fit(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(X)


def build_index(behavior_csv: Path):
    rows = list(csv.DictReader(open(behavior_csv, newline="")))
    assert len(rows) == 250
    idx = []
    for r in rows:
        key = f"{r['trajectory_id']}_t{r['t']}"
        idx.append({
            "key": key, "traj": r["trajectory_id"], "t": int(r["t"]),
            "event": r["gt_event"], "state": r["gt_state"],
            "prev_state": r["gt_prev_state"],
            "is_transition": T(r["is_transition"]),
        })
    return idx


def load_hidden(npz_path: Path, idx) -> np.ndarray:
    """Stack npz (key -> 37x4096) into (N, 37, 4096) aligned to idx order."""
    z = np.load(npz_path)
    N = len(idx)
    arr = np.zeros((N, N_LAYERS, 4096), dtype=np.float32)
    have = 0
    for i, rec in enumerate(idx):
        if rec["key"] in z.files:
            arr[i] = z[rec["key"]]
            have += 1
    assert have == N, f"only {have}/{N} keys in {npz_path}"
    return arr


def make_splits(trajs, n=None, seed=SPLIT_SEED):
    n = min(n or N_SPLITS, len(trajs))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trajs))
    groups, base = [], len(trajs) // n
    for k in range(n):
        lo = k * base + min(k, len(trajs) - n * base)
        hi = lo + base + (1 if k < len(trajs) - n * base else 0)
        groups.append([trajs[int(i)] for i in order[lo:hi]])
    return groups


def subset_mask(idx, name):
    if name == "all":
        return [True] * len(idx)
    if name == "true_transition":
        return [r["is_transition"] for r in idx]
    raise KeyError(name)


def target_labels(idx, target):
    col = {"event": "event", "state": "state",
           "prev_state": "prev_state"}[target]
    return [r[col] for r in idx]


def _job_point_wrap(idx, H_path, split_k, target, subset, layer, n_splits):
    return job_point(idx, H_path, split_k, target, subset, layer, n_splits)


def job_point(idx, H_path, split_k, target, subset, layer, n_splits):
    """Point estimate for one (split, target, subset, layer)."""
    from sklearn.linear_model import LogisticRegression
    H = np.load(H_path, mmap_mode="r")
    trajs = sorted({r["traj"] for r in idx})
    groups = make_splits(trajs, n=n_splits)
    held = set(groups[split_k])
    y_all = encode(target_labels(idx, target),
                   EVENTS if target == "event" else POS)
    mask = subset_mask(idx, subset)
    tr_i = [i for i in range(len(idx))
            if mask[i] and idx[i]["traj"] not in held]
    te_i = [i for i in range(len(idx))
            if mask[i] and idx[i]["traj"] in held]
    if len(tr_i) < 2 or len(te_i) < 1 or len(set(y_all[tr_i])) < 2:
        return {"empty": True}
    Xtr = np.asarray(H[tr_i, layer, :], dtype=np.float32)
    Xte = np.asarray(H[te_i, layer, :], dtype=np.float32)
    trajs_tr = [idx[i]["traj"] for i in tr_i]
    sc = _scaler_fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    c = select_c(Xtr_s, y_all[tr_i], trajs_tr, C_GRID,
                 f"{split_k}-{target}-{subset}-{layer}")
    clf = LogisticRegression(C=c, max_iter=1000).fit(Xtr_s, y_all[tr_i])
    p_te = clf.predict(Xte_s)
    y_te = y_all[te_i]
    from collections import Counter
    maj = Counter(y_te.tolist()).most_common(1)[0][1] / len(y_te)
    return {"empty": False, "n_te": len(te_i),
            "bal_acc": balanced_acc(y_te, p_te),
            "macro_f1": macro_f1(y_te, p_te),
            "acc": float((p_te == y_te).mean()),
            "majority": maj, "C": c}


def _job_maxt_wrap(idx, H_path, target, subset, n_maxt, n_splits):
    return job_maxt(idx, H_path, target, subset, n_maxt, n_splits)


def job_maxt(idx, H_path, target, subset, n_maxt, n_splits):
    """maxT/FWER across layers on reference split (split 0)."""
    from sklearn.linear_model import LogisticRegression
    H = np.load(H_path, mmap_mode="r")
    trajs = sorted({r["traj"] for r in idx})
    groups = make_splits(trajs, n=n_splits)
    held = set(groups[0])
    classes = EVENTS if target == "event" else POS
    y_all = encode(target_labels(idx, target), classes)
    mask = subset_mask(idx, subset)
    tr_i = [i for i in range(len(idx))
            if mask[i] and idx[i]["traj"] not in held]
    te_i = [i for i in range(len(idx))
            if mask[i] and idx[i]["traj"] in held]
    if len(tr_i) < 2 or len(te_i) < 1:
        return {"empty": True}
    Xtr_all = np.asarray(H[tr_i, :, :], dtype=np.float32)
    Xte_all = np.asarray(H[te_i, :, :], dtype=np.float32)
    trajs_tr = [idx[i]["traj"] for i in tr_i]
    t_tr = [idx[i]["t"] for i in tr_i]
    t_te = [idx[i]["t"] for i in te_i]

    # observed per-layer C (from true labels); Xtr_all is (n_train, 37, D)
    per_layer = []
    for L in range(N_LAYERS):
        sc = _scaler_fit(Xtr_all[:, L, :])
        Xs = sc.transform(Xtr_all[:, L, :])
        Xtes = sc.transform(Xte_all[:, L, :])
        c = select_c(Xs, y_all[tr_i], trajs_tr, C_GRID,
                     f"maxt-{target}-{subset}-{L}")
        per_layer.append((Xs, Xtes, c))
    obs = np.zeros(N_LAYERS)
    for L, (Xs, Xtes, c) in enumerate(per_layer):
        clf = LogisticRegression(C=c, max_iter=1000).fit(Xs, y_all[tr_i])
        obs[L] = balanced_acc(y_all[te_i], clf.predict(Xtes))
    obs_max = float(obs.max())

    rng = random.Random(MAXT_SEED)
    null_max = np.zeros(n_maxt)
    for k in range(n_maxt):
        yp = y_all[tr_i].copy()
        # within-t permutation: shuffle labels among same-t train rows
        for tt in sorted(set(t_tr)):
            pos = [j for j, x in enumerate(t_tr) if x == tt]
            lab = yp[pos].copy()
            rng.shuffle(lab.tolist())
            yp[pos] = lab
        if len(set(yp.tolist())) < 2:
            null_max[k] = 0.0
            continue
        best = 0.0
        for L, (Xs, Xtes, c) in enumerate(per_layer):
            clf = LogisticRegression(C=c, max_iter=1000).fit(Xs, yp)
            best = max(best, balanced_acc(y_all[te_i], clf.predict(Xtes)))
        null_max[k] = best
    p_maxT = float((null_max >= obs_max - 1e-12).mean())
    return {"empty": False, "obs_max": obs_max, "p_maxT": p_maxT,
            "obs_per_layer": obs.tolist()}


def main():
    global N_SPLITS, N_PERM, N_MAXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/vetbench/validity_gate_v1")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--regimes", default="2fps,8fps")
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-maxt", type=int, default=N_MAXT)
    args = ap.parse_args()
    N_SPLITS, N_PERM, N_MAXT = args.n_splits, args.n_perm, args.n_maxt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = build_index(Path(args.behavior_csv))
    results = {}
    for tag in [r.strip() for r in args.regimes.split(",") if r.strip()]:
        npz = out_dir / f"hidden_states_{tag}.npz"
        if not npz.exists():
            print(f"[{tag}] {npz} missing - skip")
            continue
        H_path = out_dir / f"hidden_{tag}.npy"
        if not H_path.exists():
            arr = load_hidden(npz, idx)
            np.save(H_path, arr)
            print(f"[{tag}] saved {H_path} {arr.shape}")
        H = np.load(H_path, mmap_mode="r")

        from joblib import Parallel, delayed
        print(f"[{tag}] point estimates: "
              f"{N_SPLITS}x{len(TARGETS)}x{len(SUBSETS)}x{N_LAYERS} jobs")
        args_pt = [(sk, tgt, sub, L)
                   for sk in range(N_SPLITS) for tgt in TARGETS
                   for sub in SUBSETS for L in range(N_LAYERS)]
        res = Parallel(n_jobs=args.n_proc, backend="loky",
                       batch_size=8)(
            delayed(_job_point_wrap)(idx, str(H_path), a[0], a[1], a[2],
                                     a[3], N_SPLITS)
            for a in args_pt)
        pt = {}
        for a, r in zip(args_pt, res):
            pt.setdefault((a[1], a[2], a[3]), []).append(r)
        # aggregate 10-split mean
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
        print(f"[{tag}] maxT/FWER across layers")
        args_mt = [(tgt, sub) for tgt in TARGETS for sub in SUBSETS]
        res_mt = Parallel(n_jobs=max(1, args.n_proc // 2), backend="loky")(
            delayed(_job_maxt_wrap)(idx, str(H_path), a[0], a[1],
                                    N_MAXT, N_SPLITS)
            for a in args_mt)
        maxt = {f"{a[0]}|{a[1]}": r for a, r in zip(args_mt, res_mt)}
        results[tag] = {"point": point, "maxt": maxt}
        with open(out_dir / f"same_input_probe_{tag}.json", "w") as fh:
            json.dump(results[tag], fh, indent=1)
        print(f"[{tag}] wrote same_input_probe_{tag}.json")

    # combined summary
    with open(out_dir / "same_input_probe_summary.json", "w") as fh:
        json.dump(results, fh, indent=1)
    # CSV
    with open(out_dir / "same_input_probe_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "target", "subset", "layer", "metric",
                    "value"])
        for tag, d in results.items():
            for key, v in d["point"].items():
                tgt, sub, L = key.split("|")
                for m in ("bal_acc", "macro_f1", "acc", "majority"):
                    w.writerow([tag, tgt, sub, L, m, f"{v[m]:.4f}"])
            for key, v in d["maxt"].items():
                tgt, sub = key.split("|")
                if not v.get("empty"):
                    w.writerow([tag, tgt, sub, "maxT", "obs_max",
                                f"{v['obs_max']:.4f}"])
                    w.writerow([tag, tgt, sub, "maxT", "p_maxT",
                                f"{v['p_maxT']:.4f}"])
        fh.close()
    print(f"wrote {out_dir}/same_input_probe_results.csv and summary")


if __name__ == "__main__":
    main()
