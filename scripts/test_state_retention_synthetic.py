"""Synthetic unit test for the frozen pre-event retention worker.

Validates, on a small synthetic tensor (no model, no GPU, ~seconds):

  1. FROZEN semantics: the pre decoder is fit on h_pre only, and the SAME
     scaler+classifier is applied to h_post. The worker's returned
     per-sample values must EXACTLY match an independently recomputed
     pipeline (StandardScaler on train h_pre -> C by group CV -> L2 LR ->
     pad_proba -> GT-index / margin / projected-drift extraction).
  2. margin identity: margin = P(GT) - max(P(other classes)).
  3. retention direction: when h_post = h_pre + large noise (the "event
     destroys the prior code" scenario), acc_post < acc_pre; when
     h_post = h_pre (perfect retention), acc_post ~= acc_pre.
  4. event decoder independence: it is fit on train h_post rows with the
     event labels, and its per-row value equals the recomputation.
  5. degenerate train set (single class) -> empty result, no crash.

Run from the project root:
    python scripts/test_state_retention_synthetic.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_retention_analysis import (  # noqa: E402
    SUBSETS,
    _job_retention_body,
)
from strict_probe_analysis import _fit_lr2, _select_c2  # noqa: E402
from run_hidden_state_probe import _pad_proba  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

D = 64
TRJS = [f"trj_{i:03d}" for i in range(8)]
T_STEPS = (2, 3, 4, 5)
RNG = np.random.default_rng(7)


def make_data(preserve: bool):
    """8 trajectories x 4 steps. h_pre encodes a 3-class latent state;
    h_post = h_pre (preserve) or h_pre + heavy noise (destroy)."""
    pairs = []
    Xp = np.zeros((len(TRJS) * len(T_STEPS), 1, D), dtype=np.float32)
    Xq = np.zeros_like(Xp)
    W = RNG.normal(size=(3, D))
    for i, tr in enumerate(TRJS):
        for j, t in enumerate(T_STEPS):
            c = int(RNG.integers(0, 3))
            h = W[c] + RNG.normal(scale=0.3, size=D)
            h = h.astype(np.float32)
            Xp[i * 4 + j, 0] = h
            if preserve:
                Xq[i * 4 + j, 0] = h + RNG.normal(scale=0.05, size=D)
            else:
                Xq[i * 4 + j, 0] = h + RNG.normal(scale=8.0, size=D)
            pairs.append({
                "trajectory_id": tr, "t": t,
                "h_pre_key": f"{tr}_t{t-1}", "h_post_key": f"{tr}_t{t}",
                "pair_complete": True,
                "gt_prev_state": ["Left", "Middle", "Right"][c],
                "gt_state": ["Left", "Middle", "Right"][int(RNG.integers(0, 3))],
                "gt_event": ["Left and Middle", "Middle and Right",
                             "Left and Right"][int(RNG.integers(0, 3))],
                "clean_revision": (c == 0),
                "canonical_stale_failure": (c == 1),
                "state_correct": (c == 2),
                "clean_maintenance": (c == 1 and t == 3),
                "canonical_group": {0: "success", 1: "stale",
                                    2: "other_failure"}[c],
            })
    STATE_INDEX = {"Left": 0, "Middle": 1, "Right": 2}
    EVENT_INDEX = {"Left and Middle": 0, "Middle and Right": 1,
                   "Left and Right": 2}
    y_pre = np.array([STATE_INDEX[p["gt_prev_state"]] for p in pairs])
    y_ev = np.array([EVENT_INDEX[p["gt_event"]] for p in pairs])
    return pairs, Xp, Xq, y_pre, y_ev


def make_job(Xp, Xq, y_pre, y_ev, pairs, tmp: Path, **kw):
    np.save(tmp / "xp.npy", Xp)
    np.save(tmp / "xq.npy", Xq)
    n = len(pairs)
    traj_of = [p["trajectory_id"] for p in pairs]
    t_of = [p["t"] for p in pairs]
    # fixed split: trj_000..005 train, trj_006..007 test
    tr = [i for i in range(n) if traj_of[i] not in ("trj_006", "trj_007")]
    te = [i for i in range(n) if traj_of[i] in ("trj_006", "trj_007")]
    job = {
        "seed": 0, "layer": 0,
        "x_pre_path": str(tmp / "xp.npy"),
        "x_post_path": str(tmp / "xq.npy"),
        "train_rows": tr, "test_rows": te,
        "y_pre": y_pre, "y_event": y_ev,
        "traj_of": traj_of, "t_of": t_of,
        "gt_prev_te": [y_pre[i] for i in te],
        "gt_event_te": [y_ev[i] for i in te],
        "rows_te": [pairs[i] for i in te],
        "test_ids": [f"{pairs[i]['trajectory_id']}_t{pairs[i]['t']}"
                     for i in te],
        "c_grid": [0.1, 1.0, 10.0],
        "c_key": "test-key",
        "rng_key": "test-null-key",
        "n_perms": 5,
    }
    job.update(kw)
    return job


def recompute(Xp, Xq, y_pre, y_ev, job):
    """Independent reimplementation of the worker's math."""
    L = job["layer"]
    tr, te = job["train_rows"], job["test_rows"]
    Xtr_pre = np.asarray(Xp[tr, L, :])
    Xte_pre = np.asarray(Xp[te, L, :])
    Xte_post = np.asarray(Xq[te, L, :])
    Xtr_post = np.asarray(Xq[tr, L, :])
    y_tr = y_pre[tr]
    trajs = [job["traj_of"][i] for i in tr]
    sc = StandardScaler().fit(Xtr_pre)
    Xtr_s, Xte_p_s, Xte_q_s = sc.transform(Xtr_pre), sc.transform(Xte_pre), \
        sc.transform(Xte_post)
    c = _select_c2(Xtr_s, y_tr, trajs, job["c_grid"], job["c_key"])
    clf = _fit_lr2(Xtr_s, y_tr, c)
    proba_p = _pad_proba(clf.predict_proba(Xte_p_s), clf.classes_, 3)
    proba_q = _pad_proba(clf.predict_proba(Xte_q_s), clf.classes_, 3)
    gt = np.array(job["gt_prev_te"])
    W, b = clf.coef_, clf.intercept_
    lp = Xte_p_s @ W.T + b
    lq = Xte_q_s @ W.T + b
    def margins(p, g):
        m = p.copy()
        m[np.arange(len(g)), g] = -np.inf
        return p[np.arange(len(g)), g] - m.max(axis=1)

    out = {
        "p_pre_gt": proba_p[np.arange(len(te)), gt],
        "p_post_gt": proba_q[np.arange(len(te)), gt],
        "pre_margin": margins(proba_p, gt),
        "post_margin": margins(proba_q, gt),
        "drift_proj": np.linalg.norm(lq - lp, axis=1),
        "drift_logit_gt": lq[np.arange(len(te)), gt]
        - lp[np.arange(len(te)), gt],
        "pred_pre": proba_p.argmax(axis=1),
        "pred_post": proba_q.argmax(axis=1),
    }
    # event decoder recompute
    sc_e = StandardScaler().fit(Xtr_post)
    Xtr_e_s = sc_e.transform(Xtr_post)
    Xte_e_s = sc_e.transform(Xte_post)
    c_e = _select_c2(Xtr_e_s, y_ev[tr], trajs, job["c_grid"],
                     job["c_key"] + "-ev")
    clf_e = _fit_lr2(Xtr_e_s, y_ev[tr], c_e)
    pe = _pad_proba(clf_e.predict_proba(Xte_e_s), clf_e.classes_, 3)
    out["p_event_gt"] = pe[np.arange(len(te)), np.array(job["gt_event_te"])]
    out["c_pre"], out["c_ev"] = c, c_e
    return out


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=float),
                       np.asarray(b, dtype=float), atol=tol, rtol=0)


def main() -> int:
    failures = []

    def check(name, cond, extra=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name} {extra}")
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for preserve in (False, True):
            tag = "preserve" if preserve else "destroy"
            pairs, Xp, Xq, y_pre, y_ev = make_data(preserve)
            job = make_job(Xp, Xq, y_pre, y_ev, pairs, tmp)
            res = _job_retention_body(job)
            ref = recompute(Xp, Xq, y_pre, y_ev, job)

            check(f"{tag}: not empty", not res.get("empty"))
            check(f"{tag}: C matches recompute",
                  abs(res["C_pre"] - ref["c_pre"]) < 1e-12,
                  f"({res['C_pre']} vs {ref['c_pre']})")
            for k in ("p_pre_gt", "p_post_gt", "pre_margin", "post_margin",
                      "drift_proj", "drift_logit_gt", "pred_pre",
                      "pred_post"):
                check(f"{tag}: samples.{k} == recompute",
                      close(res["samples"][k], ref[k]),
                      "" if close(res["samples"][k], ref[k])
                      else f"\n  got={res['samples'][k]}\n  ref={ref[k]}")
            check(f"{tag}: event proba == recompute",
                  close(res["samples"]["p_event_gt"], ref["p_event_gt"]))

            # margin identity on the reference full proba matrices
            # (re-derive full matrices to verify the margin formula)
            Xtr_pre = np.asarray(Xp[job["train_rows"], 0, :])
            sc = StandardScaler().fit(Xtr_pre)
            Xte_p_s = sc.transform(np.asarray(Xp[job["test_rows"], 0, :]))
            clf = _fit_lr2(sc.transform(Xtr_pre),
                           y_pre[job["train_rows"]], ref["c_pre"])
            full = _pad_proba(clf.predict_proba(Xte_p_s), clf.classes_, 3)
            gt = np.array(job["gt_prev_te"])
            mfull = full.copy()
            mfull[np.arange(len(gt)), gt] = -np.inf
            margin_ref = full[np.arange(len(gt)), gt] - mfull.max(axis=1)
            check(f"{tag}: margin identity (P_GT - max other)",
                  close(res["samples"]["pre_margin"], margin_ref))

            acc_pre = res["acc_pre_all"]
            acc_post = res["acc_post_all"]
            print(f"        {tag}: acc_pre={acc_pre:.3f} "
                  f"acc_post={acc_post:.3f} drop={acc_post - acc_pre:+.3f}")
            if not preserve:
                check(f"{tag}: acc_pre above chance", acc_pre >= 0.6)
                check(f"{tag}: acc_post < acc_pre (retention drop)",
                      acc_post < acc_pre)
            else:
                check(f"{tag}: acc_pre above chance", acc_pre >= 0.6)
                check(f"{tag}: acc_post ~= acc_pre (retention kept)",
                      abs(acc_post - acc_pre) <= 0.25)

            # nulls: same number of draws for pre/post/drop per subset
            for sub, dd in res.get("nulls", {}).items():
                check(f"{tag}: nulls balanced for {sub}",
                      len(dd["pre"]) == len(dd["post"])
                      == len(dd["drop"]) and len(dd["pre"]) > 0)

        # degenerate: single-class pre labels -> empty
        pairs, Xp, Xq, y_pre, y_ev = make_data(False)
        y_pre = np.zeros_like(y_pre)
        job = make_job(Xp, Xq, y_pre, y_ev, pairs, tmp)
        res = _job_retention_body(job)
        check("degenerate single-class train -> empty", res.get("empty"))

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}")
        return 1
    print("\nALL SYNTHETIC TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
