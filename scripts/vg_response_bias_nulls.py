#!/usr/bin/env python3
"""Validity Gate Stage 0.2: response-bias / null-model audit (CPU only).

Question: can the Transformers-aligned state answers (and in particular the
"stale" pattern pred_t == GT_{t-1} on transition errors) be explained by
simple response bias / answer persistence, without watching the video?

Pre-registered design (fixed before looking at subgroup results):

- Data: outputs/vetbench/composition_analysis_v1/transformers_behavior.csv
  (250 rows = 50 trajectories x t=1..5). "stale" is NOT treated as a
  mechanism claim here; only the response distributions are audited.
- State prompt fixed presentation order: "(A) Left (B) Middle (C) Right",
  with the answer example "Left" (i.e. the first position is the example).
  S0 is stated verbatim in the prompt.
- Null models (parameters fit on TRAIN trajectories only):
    1 categorical   : empirical P_train(pred) over all train rows
    2 most_frequent : always the most frequent train position
    3 always_s0     : pred = S0 of the row (stated in the prompt)
    4 repeat_prev   : pred_t = the model's own pred_{t-1} on the same
                      trajectory; at t=1 (no previous prediction) use S0
    5 markov        : P_train(pred_t | pred_{t-1}) from train trajectories
                      (t=2..5 pairs), Laplace alpha=1; at t=1 use the
                      empirical P_train(pred_1) marginal
- Splits: 10 trajectory splits (40 train / 10 held-out), rng seed 12345;
  pooled held-out rows across splits = all 250 rows, each row scored in the
  one split where its trajectory was held out (no leakage).
- Stochastic nulls (1, 5) are simulated per row with rng seed 1000+split;
  analytic expected accuracy is reported alongside.
- NLL: for stochastic nulls, mean per-row NLL (bits) of the REAL model's
  held-out prediction under the null. For degenerate nulls (2,3,4) the
  mismatch rate against the real prediction is reported instead.
- Metrics per null and for the real model: overall / transition /
  maintenance accuracy, stale share of transition errors
  (1[sim_pred == GT_{t-1}] among is_transition rows where sim_pred !=
  GT_t), canonical-conditional stale rate (eligibility: is_transition &
  event_correct & prev_state_correct), marginal pred distribution.
- CI: trajectory bootstrap, 1000 resamples, rng seed 0, percentiles 2.5/97.5.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

POS = ("Left", "Middle", "Right")
SPLITS, SPLIT_SEED, SIM_BASE, BOOT_N, BOOT_SEED = 10, 12345, 1000, 1000, 0


def T(x: str) -> bool:
    return str(x).strip().lower() == "true"


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    assert len(rows) == 250
    for r in rows:
        r["t"] = int(r["t"])
    return rows


def make_splits(trajs, n_splits=SPLITS, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trajs))
    groups = [[trajs[int(i)] for i in order[k * len(trajs) // n_splits:
                               (k + 1) * len(trajs) // n_splits]]
              for k in range(n_splits)]
    return groups  # split k: held-out = groups[k], train = the rest


def fit_nulls(rows_train):
    tr = [r for r in rows_train if r["state_pred"] in POS]
    cnt = Counter(r["state_pred"] for r in tr)
    p_cat = np.array([cnt.get(p, 0) for p in POS], float)
    p_cat = p_cat / max(p_cat.sum(), 1.0)
    most_freq = POS[int(np.argmax(p_cat))]
    # markov: P(pred_t | pred_{t-1}) from consecutive rows per trajectory
    trans = {a: Counter() for a in POS}
    p1 = Counter()
    for tid in {r["trajectory_id"] for r in tr}:
        seq = sorted((r for r in tr if r["trajectory_id"] == tid),
                     key=lambda r: r["t"])
        for r in seq:
            if r["t"] == 1:
                p1[r["state_pred"]] += 1
        for a, b in zip(seq, seq[1:]):
            if a["state_pred"] in POS and b["state_pred"] in POS:
                trans[a["state_pred"]][b["state_pred"]] += 1
    alpha = 1.0
    markov = {}
    for a in POS:
        v = np.array([trans[a].get(p, 0) + alpha for p in POS], float)
        markov[a] = v / v.sum()
    p1 = np.array([p1.get(p, 0) for p in POS], float)
    if p1.sum() == 0:
        p1 = np.ones(3) / 3
    p1 = p1 / p1.sum()
    return {"categorical": p_cat, "most_frequent": most_freq,
            "markov": markov, "markov_t1": p1}


def null_pred(name, fit, row, prev_pred):
    """Fixed (degenerate) prediction of a null for a row."""
    if name == "categorical" or name == "markov":
        return None  # stochastic, sampled separately
    if name == "most_frequent":
        return fit["most_frequent"]
    if name == "always_s0":
        return row["initial_state"]
    if name == "repeat_prev":
        return prev_pred if prev_pred is not None else row["initial_state"]
    raise KeyError(name)


def sample_null(name, fit, row, prev_pred, rng):
    if name == "categorical":
        p = fit["categorical"]
    elif name == "markov":
        p = fit["markov_t1"] if prev_pred is None else fit["markov"][prev_pred]
    else:
        return null_pred(name, fit, row, prev_pred)
    return POS[int(rng.choice(3, p=p))]


def null_nll(name, fit, row, prev_pred, real_pred):
    if name == "categorical":
        p = fit["categorical"][list(POS).index(real_pred)]
    elif name == "markov":
        dist = fit["markov_t1"] if prev_pred is None else fit["markov"][prev_pred]
        p = dist[list(POS).index(real_pred)]
    else:
        return None
    return -np.log2(max(p, 1e-12))


def metrics(preds, rows, rng_boot=None):
    """preds: dict sample_id -> predicted position (or None)."""
    out = {}
    n_trans = n_maint = 0
    acc_trans = acc_maint = 0
    trans_err = trans_err_stale = 0
    canon = canon_stale = 0
    marg = Counter()
    for r in rows:
        sid = f"{r['trajectory_id']}_t{r['t']}"
        p = preds.get(sid)
        if p is None:
            continue
        marg[p] += 1
        correct = p == r["gt_state"]
        if T(r["is_transition"]):
            n_trans += 1
            acc_trans += correct
            if not correct:
                trans_err += 1
                trans_err_stale += p == r["gt_prev_state"]
        else:
            n_maint += 1
            acc_maint += correct
        if (T(r["is_transition"]) and T(r["event_correct"])
                and T(r["prev_state_correct"])):
            canon += 1
            canon_stale += p == r["gt_prev_state"]
    out["n"] = sum(marg.values())
    out["overall_acc"] = (sum(1 for r in rows
                              if preds.get(f"{r['trajectory_id']}_t{r['t']}")
                              == r["gt_state"]) / max(sum(marg.values()), 1))
    out["transition_acc"] = acc_trans / max(n_trans, 1)
    out["maintenance_acc"] = acc_maint / max(n_maint, 1)
    out["transition_error_stale_rate"] = trans_err_stale / max(trans_err, 1)
    out["n_transition_errors"] = trans_err
    out["canonical_stale_rate"] = canon_stale / max(canon, 1)
    out["n_canonical"] = canon
    out["marginal"] = {p: marg.get(p, 0) / max(sum(marg.values()), 1)
                       for p in POS}
    return out


def bootstrap(rows, preds, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    trajs = sorted({r["trajectory_id"] for r in rows})
    by_traj = {tid: [r for r in rows if r["trajectory_id"] == tid]
               for tid in trajs}
    keys = ("overall_acc", "transition_acc", "maintenance_acc",
            "transition_error_stale_rate", "canonical_stale_rate")
    boot = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.choice(len(trajs), size=len(trajs), replace=True)
        sample = [trajs[i] for i in idx]
        srows = [r for tid in sample for r in by_traj[tid]]
        m = metrics(preds, srows)
        for k in keys:
            boot[k].append(m[k])
    ci = {}
    for k in keys:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        ci[k] = [round(float(lo), 4), round(float(hi), 4)]
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load(args.behavior_csv)
    trajs = sorted({r["trajectory_id"] for r in rows})
    prev_pred = {}
    for r in sorted(rows, key=lambda r: (r["trajectory_id"], r["t"])):
        if r["t"] > 1:
            pp = f"{r['trajectory_id']}_t{r['t']-1}"
            prev_pred[f"{r['trajectory_id']}_t{r['t']}"] = \
                next(x["state_pred"] for x in rows
                     if f"{x['trajectory_id']}_t{x['t']}" == pp
                     and x["state_pred"] in POS) or None

    groups = make_splits(trajs)

    # ---- real-model descriptive tables (all 250 rows) ----
    def dist(subset):
        c = Counter(r["state_pred"] for r in subset)
        n = max(len(subset), 1)
        return {p: round(c.get(p, 0) / n, 4) for p in POS}

    by_t = {str(t): dist([r for r in rows if r["t"] == t]) for t in range(1, 6)}
    by_gtc = {p: dist([r for r in rows if r["gt_state"] == p]) for p in POS}
    by_gtp = {p: dist([r for r in rows if r["gt_prev_state"] == p]) for p in POS}
    by_s0 = {p: dist([r for r in rows if r["initial_state"] == p]) for p in POS}
    conf = {f"gt={g}|pred={p}": 0 for g in POS for p in POS}
    conf_trans = {f"gt={g}|pred={p}": 0 for g in POS for p in POS}
    conf_maint = {f"gt={g}|pred={p}": 0 for g in POS for p in POS}
    for r in rows:
        k = f"gt={r['gt_state']}|pred={r['state_pred']}"
        conf[k] += 1
        (conf_trans if T(r["is_transition"]) else conf_maint)[k] += 1

    real_preds = {f"{r['trajectory_id']}_t{r['t']}": r["state_pred"]
                  for r in rows if r["state_pred"] in POS}
    real_metrics = metrics(real_preds, rows)
    real_ci = bootstrap(rows, real_preds)

    # ---- null models ----
    null_names = ("categorical", "most_frequent", "always_s0",
                  "repeat_prev", "markov")
    results = {}
    for k, held in enumerate(groups):
        train = [t for g in groups if g is not held for t in g]
        rtrain = [r for r in rows if r["trajectory_id"] in set(train)]
        fit = fit_nulls(rtrain)
        nlls = []
        mism = {n: [] for n in null_names}
        for name in null_names:
            rng2 = np.random.default_rng(SIM_BASE + k * 10 +
                                         list(null_names).index(name))
            sim_n = {}
            for r in rows:
                if r["trajectory_id"] not in set(held):
                    continue
                sid = f"{r['trajectory_id']}_t{r['t']}"
                pp = prev_pred.get(sid)
                sim_n[sid] = sample_null(name, fit, r, pp, rng2)
            m = metrics(sim_n, [r for r in rows
                                if r["trajectory_id"] in set(held)])
            d = results.setdefault(name, {"splits": []})
            d["splits"].append(m)
            d.setdefault("sim_pool", {})
            d["sim_pool"].update({sid: sim_n[sid]
                                  for sid in sim_n})
            if name in ("categorical", "markov"):
                for r in rows:
                    if r["trajectory_id"] in set(held):
                        n = null_nll(name, fit, r, prev_pred.get(
                            f"{r['trajectory_id']}_t{r['t']}"),
                            r["state_pred"])
                        if n is not None:
                            nlls.append(n)
            else:
                for r in rows:
                    if r["trajectory_id"] in set(held):
                        sid = f"{r['trajectory_id']}_t{r['t']}"
                        fixed = null_pred(name, fit, r,
                                          prev_pred.get(sid))
                        mism[name].append(
                            0.0 if fixed == r["state_pred"] else 1.0)
            d.setdefault("nll", [])
            if name in ("categorical", "markov"):
                d["nll"] = nlls if k == 0 else d["nll"] + nlls
            d.setdefault("mismatch", [])
            d["mismatch"] = (mism[name] if k == 0
                             else d["mismatch"] + mism[name])

    # ---- pool & summarize ----
    summary = {}
    csv_rows = []
    for name in null_names:
        d = results[name]
        pool = d["sim_pool"]
        m = metrics(pool, rows)
        ci = bootstrap(rows, pool)
        if name in ("categorical", "markov"):
            nll = float(np.mean(d["nll"]))
            extra = {"nll_bits_real_pred": round(nll, 4),
                     "expected_acc_vs_gt": None}
        else:
            nll = None
            extra = {"mismatch_rate_vs_real_pred":
                     round(float(np.mean(d["mismatch"])), 4)}
        entry = {**{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                    for kk, vv in m.items()},
                 "ci_bootstrap": ci, **extra}
        summary[name] = entry
        for kk, vv in entry.items():
            if kk in ("marginal", "ci_bootstrap"):
                continue
            csv_rows.append({"model": name, "metric": kk, "value": vv})
    for kk, vv in real_metrics.items():
        if kk == "marginal":
            continue
        csv_rows.append({"model": "real_model", "metric": kk,
                         "value": round(vv, 4) if isinstance(vv, float)
                         else vv})
    csv_rows.append({"model": "real_model", "metric": "marginal",
                     "value": json.dumps(real_metrics["marginal"])})
    csv_rows.append({"model": "real_model", "metric": "ci_bootstrap",
                     "value": json.dumps(real_ci)})

    payload = {
        "config": {
            "behavior_csv": args.behavior_csv,
            "state_prompt_option_order": "(A) Left (B) Middle (C) Right",
            "state_prompt_example": 'Answer with the option text, e.g. "Left"',
            "s0_stated_in_prompt": True,
            "splits": SPLITS, "split_seed": SPLIT_SEED,
            "sim_seed_base": SIM_BASE,
            "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED,
                          "unit": "trajectory",
                          "ci": "percentile 2.5/97.5"},
            "t1_rule": "repeat_prev falls back to S0 at t=1; markov uses "
                       "empirical P_train(pred_1) marginal at t=1",
            "markov_smoothing": "Laplace alpha=1",
            "canon_eligibility": "is_transition & event_correct & "
                                 "prev_state_correct (row facts from audit)",
        },
        "real_model": {
            "pred_distribution_overall": dist(rows),
            "pred_distribution_by_t": by_t,
            "pred_distribution_by_gt_current": by_gtc,
            "pred_distribution_by_gt_previous": by_gtp,
            "pred_distribution_by_s0": by_s0,
            "confusion_gt_x_pred": conf,
            "confusion_transition": conf_trans,
            "confusion_non_transition": conf_maint,
            "metrics": {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in real_metrics.items()},
            "ci_bootstrap": real_ci,
        },
        "null_models": summary,
        "interpretation_guide": {
            "nll_bits_real_pred": "mean NLL (bits) of the REAL model's "
                                  "held-out answer under the null; "
                                  "log2(3)=1.585 is the uniform-random "
                                  "level. Much lower = the null explains "
                                  "the model's answers.",
            "mismatch_rate_vs_real_pred": "fraction of held-out rows where "
                                          "the degenerate null's fixed "
                                          "answer differs from the real "
                                          "answer (1.0 = no overlap).",
            "transition_error_stale_rate": "share of transition rows the "
                                           "null gets wrong in which the "
                                           "wrong answer equals GT_{t-1}.",
            "canonical_stale_rate": "share of canonical-eligible rows "
                                    "where the null answers GT_{t-1}.",
        },
    }
    with open(out_dir / "response_bias_nulls.json", "w") as f:
        json.dump(payload, f, indent=1)
    with open(out_dir / "response_bias_nulls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {out_dir / 'response_bias_nulls.json'} "
          f"and .csv")
    print("\nreal model metrics:",
          {k: v for k, v in real_metrics.items() if k != "marginal"})
    for name in null_names:
        e = summary[name]
        print(f"{name:14s} acc={e['overall_acc']:.3f} "
              f"trans={e['transition_acc']:.3f} "
              f"maint={e['maintenance_acc']:.3f} "
              f"transErrStale={e['transition_error_stale_rate']:.3f} "
              f"canonStale={e['canonical_stale_rate']:.3f} "
              + (f"nll={e.get('nll_bits_real_pred')}"
                 if e.get('nll_bits_real_pred') is not None else
                 f"mismatch={e.get('mismatch_rate_vs_real_pred')}"))


if __name__ == "__main__":
    main()
