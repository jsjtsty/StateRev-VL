#!/usr/bin/env python3
"""Offline v2 analysis for the completed decisive mechanism run.

This module never loads the VLM. It repairs the pair-key/native-logit join,
reports target-prefix and trajectory-cluster statistics, and evaluates state
hidden states with route-held-out decoders. Pair rows are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
STATES = ("Left", "Middle", "Right")
ROUTES_A = ("Left->Middle", "Middle->Right", "Right->Left")
ROUTES_B = ("Middle->Left", "Right->Middle", "Left->Right")
PAIR_CONDITIONS = ("target_self", "same_event_source", "matched_history_transplant", "source_window_shuffle", "source_current_transplant")


def route(prev: str, state: str) -> str:
    return f"{prev}->{state}"


def load_rows(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    for r in rows:
        r["t"] = int(r["t"])
        r["key"] = f"{r['trajectory_id']}_t{r['t']}"
        r["route"] = route(r["gt_prev_state"], r["gt_state"])
    assert len(rows) == 250
    return rows


def load_npz(path: Path):
    return np.load(path, allow_pickle=False)


def fit_decoder(X, y, traj, c_grid=(0.1, 1.0, 10.0)):
    """Fit train-only StandardScaler/PCA/C selection, returning predict_proba."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ncomp = min(80, max(2, Xs.shape[0] - 1), Xs.shape[1])
    pca = PCA(n_components=ncomp, random_state=0).fit(Xs)
    Z = pca.transform(Xs)
    groups = sorted(set(traj))
    folds = [groups[i::3] for i in range(3)]
    best_c, best = 1.0, -np.inf
    for c in c_grid:
        scores = []
        for held in folds:
            te = np.array([g in held for g in traj])
            tr = ~te
            if tr.sum() < 6 or len(set(y[tr])) < 2 or len(set(y[te])) < 1:
                continue
            clf = LogisticRegression(C=c, max_iter=500).fit(Z[tr], y[tr])
            scores.append(float((clf.predict(Z[te]) == y[te]).mean()))
        if scores and np.mean(scores) > best:
            best, best_c = float(np.mean(scores)), c
    clf = LogisticRegression(C=best_c, max_iter=500).fit(Z, y)

    def predict(Xnew):
        p = np.zeros((len(Xnew), len(STATES)))
        q = clf.predict_proba(pca.transform(scaler.transform(np.asarray(Xnew, dtype=np.float32))))
        for j, cls in enumerate(clf.classes_):
            p[:, int(cls)] = q[:, j]
        return p
    return predict, {"C": best_c, "n_train": int(len(y)), "pca_dim": int(ncomp)}


def cluster_summary(values, groups, seed=20260906, n_boot=4000):
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(float(v))
    cluster_values = np.array([np.mean(v) for v in by.values()], dtype=float)
    if not len(cluster_values):
        return {"mean": None, "ci95": [None, None], "p": None, "n_trajectories": 0, "effect_size": None}
    rng = np.random.default_rng(seed)
    draws = rng.choice(cluster_values, (n_boot, len(cluster_values)), replace=True).mean(axis=1)
    signs = rng.choice([-1.0, 1.0], (n_boot, len(cluster_values)))
    p = float(np.mean(np.abs((cluster_values * signs).mean(axis=1)) >= abs(cluster_values.mean())))
    sd = float(np.std(cluster_values, ddof=1)) if len(cluster_values) > 1 else math.nan
    return {"mean": float(cluster_values.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "p_sign_permutation": p, "n_trajectories": len(cluster_values),
            "effect_size_cluster_d": (float(cluster_values.mean() / sd) if sd and np.isfinite(sd) else None)}


def aggregate_prefix(rows, field):
    by = defaultdict(list)
    for r in rows:
        by[r["target_prefix"]].append(float(r[field]))
    out = []
    for key, xs in by.items():
        out.append({"target_prefix": key, "target_traj": rows[[r["target_prefix"] for r in rows].index(key)]["target_traj"], field: float(np.mean(xs)), "n_source_pairs": len(xs)})
    return out


def native_analysis(out: Path, pairs, source_rows, base_rows):
    bmap = {(r["key"], r["condition"]): r for r in source_rows}
    base = {r["key"]: r for r in base_rows}
    pair_out = []
    for p in pairs:
        pair_key = p["pair_id"]
        target_key = f"{p['target_traj']}_t{p['t']}"
        b = base[target_key]
        h = bmap[(pair_key, "source_current_transplant")]
        st, cf = p["target_state"], p["counterfactual_state"]
        def logits(r): return {s: float(r[f"logprob_{s}"]) for s in STATES}
        bl, hl = logits(b), logits(h)
        def prob(r, s): return math.exp(float(r[f"logprob_{s}"]))
        base_margin = bl[cf] - bl[st]
        hybrid_margin = hl[cf] - hl[st]
        row = {"pair_id": pair_key, "target_prefix": target_key,
               "target_traj": p["target_traj"], "t": p["t"],
               "target_state": st, "counterfactual_state": cf,
               "source_event": p["source_event"], "target_event": p["target_event"],
               "native_cf_shift": hybrid_margin - base_margin,
               "native_cf_probability_change": prob(h, cf) - prob(b, cf),
               "native_cf_prediction": int(h["state_pred"] == cf),
               "base_pred": b["state_pred"], "hybrid_pred": h["state_pred"]}
        for s in STATES:
            row[f"baseline_logit_{s}"] = bl[s]
            row[f"hybrid_logit_{s}"] = hl[s]
            row[f"logit_change_{s}"] = hl[s] - bl[s]
        for c in PAIR_CONDITIONS[:-1]:
            q = bmap[(pair_key, c)]
            qm = float(q[f"logprob_{cf}"]) - float(q[f"logprob_{st}"])
            row[f"control_shift_{c}"] = qm - base_margin
        pair_out.append(row)
    def summarize(field, subset):
        xs = [r for r in pair_out if subset(r)]
        by = defaultdict(list)
        for r in xs: by[r["target_prefix"]].append(r)
        targets = []
        for k, rr in by.items():
            z = {"target_prefix": k, "target_traj": rr[0]["target_traj"], "t": rr[0]["t"]}
            z[field] = float(np.mean([r[field] for r in rr]))
            targets.append(z)
        return cluster_summary([r[field] for r in targets], [r["target_traj"] for r in targets]), targets
    summaries = {}
    for name, pred in [("overall", lambda r: True), ("t1", lambda r: r["t"] == 1),
                       ("t2", lambda r: r["t"] == 2), ("t3", lambda r: r["t"] == 3),
                       ("t4", lambda r: r["t"] == 4), ("t5", lambda r: r["t"] == 5),
                       ("t_ge2", lambda r: r["t"] >= 2), ("t_ge3", lambda r: r["t"] >= 3)]:
        s, targets = summarize("native_cf_shift", pred)
        probs, _ = summarize("native_cf_probability_change", pred)
        pred_rate, _ = summarize("native_cf_prediction", pred)
        controls = {}
        for c in PAIR_CONDITIONS[:-1]:
            vals = [{"target_traj": r["target_traj"], "target_prefix": r["target_prefix"], "v": r["native_cf_shift"] - r[f"control_shift_{c}"]}
                    for r in pair_out if pred(r)]
            by = defaultdict(list)
            for r in vals: by[r["target_prefix"]].append(r)
            tvals = [(np.mean([x["v"] for x in rr]), rr[0]["target_traj"]) for rr in by.values()]
            controls[c] = cluster_summary([x[0] for x in tvals], [x[1] for x in tvals])
        summaries[name] = {"native_cf_shift": s,
                           "native_cf_probability_change": probs,
                           "native_cf_prediction_rate": pred_rate,
                           "paired_control_contrasts": controls,
                           "n_pair_rows_descriptive": sum(pred(r) for r in pair_out),
                           "n_target_prefixes": len({r["target_prefix"] for r in pair_out if pred(r)})}
    return pair_out, summaries


def ood_analysis(out: Path, pairs, rows, hidden):
    index = {r["key"]: r for r in rows}
    H = {k: load_npz(out / f"hidden_{k}.npz") for k in hidden}
    final = lambda z, k: z[k][-1][None, :]
    labels = {r["key"]: STATES.index(r["gt_state"]) for r in rows}
    trajectories = {r["key"]: r["trajectory_id"] for r in rows}
    routes = {r["key"]: r["route"] for r in rows}
    pair_rows = []
    protocols = {"A_to_B": set(ROUTES_A), "B_to_A": set(ROUTES_B), "leave_one_route_out": set(ROUTES_A + ROUTES_B)}
    decoder_cache = {}
    for protocol, train_routes in protocols.items():
        for p in pairs:
            target_key = f"{p['target_traj']}_t{p['t']}"
            target_route = routes[target_key]
            if protocol == "A_to_B" and target_route not in ROUTES_B: continue
            if protocol == "B_to_A" and target_route not in ROUTES_A: continue
            allowed = train_routes - {target_route} if protocol == "leave_one_route_out" else train_routes
            train_keys = [k for k in index if routes[k] in allowed and trajectories[k] != p["target_traj"]]
            if len(train_keys) < 12: continue
            y = np.array([labels[k] for k in train_keys])
            if len(set(y)) < 3: continue
            cache_key = (protocol, p["target_traj"], target_route)
            if cache_key not in decoder_cache:
                decoder_cache[cache_key] = fit_decoder(
                    np.vstack([final(H["baseline"], k) for k in train_keys]), y,
                    [trajectories[k] for k in train_keys])
            pred, meta = decoder_cache[cache_key]
            base_p = pred(final(H["baseline"], target_key))[0]
            hy_p = pred(final(H["source_current_transplant"], p["pair_id"]))[0]
            cf = STATES.index(p["counterfactual_state"]); st = STATES.index(p["target_state"])
            pair_rows.append({"protocol": protocol, "pair_id": p["pair_id"], "target_prefix": target_key,
                              "target_traj": p["target_traj"], "t": p["t"], "target_route": target_route,
                              "counterfactual_state": p["counterfactual_state"], "target_state": p["target_state"],
                              "p_cf_baseline": float(base_p[cf]), "p_cf_hybrid": float(hy_p[cf]),
                              "ood_state_shift": float(hy_p[cf] - base_p[cf]),
                              "ood_margin_shift": float((hy_p[cf] - hy_p[st]) - (base_p[cf] - base_p[st])),
                              "p_target_baseline": float(base_p[st]), "p_target_hybrid": float(hy_p[st]),
                              **meta})
    summaries = {}
    for protocol in sorted({r["protocol"] for r in pair_rows}):
        for name, pred in [("overall", lambda r: True), ("t1", lambda r: r["t"] == 1),
                           ("t2", lambda r: r["t"] == 2), ("t3", lambda r: r["t"] == 3),
                           ("t4", lambda r: r["t"] == 4), ("t5", lambda r: r["t"] == 5),
                           ("t_ge2", lambda r: r["t"] >= 2), ("t_ge3", lambda r: r["t"] >= 3)]:
            xs = [r for r in pair_rows if r["protocol"] == protocol and pred(r)]
            by = defaultdict(list)
            for r in xs: by[r["target_prefix"]].append(r)
            targets = [{"target_prefix": k, "target_traj": rr[0]["target_traj"],
                        "ood_state_shift": float(np.mean([x["ood_state_shift"] for x in rr])),
                        "ood_margin_shift": float(np.mean([x["ood_margin_shift"] for x in rr]))} for k, rr in by.items()]
            summaries[f"{protocol}:{name}"] = {"ood_state_shift": cluster_summary([r["ood_state_shift"] for r in targets], [r["target_traj"] for r in targets]),
                                                "ood_margin_shift": cluster_summary([r["ood_margin_shift"] for r in targets], [r["target_traj"] for r in targets]),
                                                "n_pair_rows_descriptive": len(xs), "n_target_prefixes": len(targets)}
    return pair_rows, summaries


def write_csv(path, rows):
    if not rows: path.write_text(""); return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("outputs/vetbench/mechanism_gate_final_v2")); ap.add_argument("--source", type=Path, default=Path("outputs/vetbench/mechanism_gate_final")); args = ap.parse_args()
    out, src = args.out, args.source; out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"))
    pairs = json.loads((src / "pair_manifest.json").read_text())["pairs"]
    behavior = list(csv.DictReader((src / "behavior.csv").open(newline="")))
    pair_native, native = native_analysis(out, pairs, behavior, [r for r in behavior if r["condition"] == "baseline"])
    write_csv(out / "native_counterfactual_results.csv", pair_native)
    (out / "native_counterfactual_summary.json").write_text(json.dumps({"protocol": "target-prefix aggregation then trajectory-cluster bootstrap/permutation", "n_pairs_descriptive": len(pair_native), "results": native}, indent=2))
    hidden = ("baseline", "source_current_transplant")
    pair_ood, ood = ood_analysis(src, pairs, rows, hidden)
    write_csv(out / "ood_state_decoder_results.csv", pair_ood)
    write_csv(out / "ood_counterfactual_shift.csv", pair_ood)
    (out / "ood_state_decoder_summary.json").write_text(json.dumps({"results": ood, "route_A": ROUTES_A, "route_B": ROUTES_B, "decoder": "final hidden layer; train-only StandardScaler, PCA(80), grouped 3-fold C selection", "n_pairs_descriptive": len(pair_ood)}, indent=2))
    # Use the existing event efficacy gate, but keep all eligible target prefixes primary.
    existing = json.loads((src / "decisive_analysis.json").read_text())
    iid_prefixes = existing["target_prefix_effects"]
    def iid_cluster_for(predicate):
        xs = [x for x in iid_prefixes if predicate(int(x["target_prefix"].rsplit("_t", 1)[1]))]
        by = defaultdict(list)
        for x in xs: by[x["target_traj"]].append(x["state_p_cf_delta"])
        vals = [float(np.mean(v)) for v in by.values()]
        return cluster_summary(vals, list(by))
    iid_ge2 = iid_cluster_for(lambda t: t >= 2)
    iid_ge3 = iid_cluster_for(lambda t: t >= 3)
    native_overall = native["overall"]["native_cf_shift"]
    native_ge2 = native["t_ge2"]["native_cf_shift"]
    native_ge3 = native["t_ge3"]["native_cf_shift"]
    def get(protocol, name): return ood.get(f"{protocol}:{name}", {}).get("ood_state_shift", {})
    i_ge2 = iid_ge2["ci95"]
    i_ge3 = iid_ge3["ci95"]
    o_ge2 = get("leave_one_route_out", "t_ge2")
    if o_ge2.get("ci95") and o_ge2["ci95"][0] > 0: pattern = "IID_and_OOD_state_shift"
    else: pattern = "IID_state_shift_but_OOD_not_confirmed"
    verdict = "GO-restricted-representation" if pattern == "IID_and_OOD_state_shift" else "NO-GO-abstract-state-generalization"
    report = f'''# StateRev-VL mechanism v2 offline report

No model or video forward was run. All results reuse `mechanism_gate_final`.
Pair-level rows are descriptive; all confidence intervals and sign tests use
target-prefix aggregation followed by trajectory-cluster bootstrap/permutation.
The 870 pairs are never treated as independent observations.

## Native state logits

The join is repaired by mapping `target_t_from_source` to `target_t` for the
same target trajectory in the baseline. The primary statistic is
`[(logit(S_cf)-logit(S_target))_hybrid - (logit(S_cf)-logit(S_target))_baseline]`.

| subset | native cf-shift mean | 95% trajectory-cluster CI | sign p | cf prediction rate |
|---|---:|---:|---:|---:|
| overall | {native_overall["mean"]:.4f} | {native_overall["ci95"]} | {native_overall["p_sign_permutation"]:.4g} | {native["overall"]["native_cf_prediction_rate"]["mean"]:.4f} |
| t>=2 | {native_ge2["mean"]:.4f} | {native_ge2["ci95"]} | {native_ge2["p_sign_permutation"]:.4g} | {native["t_ge2"]["native_cf_prediction_rate"]["mean"]:.4f} |
| t>=3 | {native_ge3["mean"]:.4f} | {native_ge3["ci95"]} | {native_ge3["p_sign_permutation"]:.4g} | {native["t_ge3"]["native_cf_prediction_rate"]["mean"]:.4f} |

Full t=1..5 breakdown is in `native_counterfactual_summary.json`. Native logits
are the actual model readout endpoint; no erroneous pair-key accuracy is used.

## OOD current-state decoder

The decoder uses only baseline prefixes for training, with train-only scaler,
PCA and grouped C selection. A/B route splits are `A={ROUTES_A}` and
`B={ROUTES_B}`. Leave-one-route-out excludes the target's directed route from
training. Hybrid evaluation uses the same target prefix and source current
window as the existing efficacy-passing experiment, but all eligible targets
remain primary.

| protocol | subset | OOD state shift mean | 95% trajectory-cluster CI | sign p |
|---|---|---:|---:|---:|
'''
    for key, val in sorted(ood.items()):
        x = val["ood_state_shift"]
        report += f'| {key.split(":")[0]} | {key.split(":")[1]} | {x["mean"]:.4f} | {x["ci95"]} | {x["p_sign_permutation"]:.4g} |\n'
    report += f'''\n## Final interpretation

Existing event efficacy remains the prerequisite: target-cluster efficacy is
`{existing["event_efficacy_rate_target_cluster"]:.4f}` (threshold 0.7).
Existing ordinary IID state-probe shift is `{existing["pair_state_cf_delta_mean_target_cluster"]:.4f}` overall,
with CI `{existing["pair_state_cf_delta_ci95_trajectory_cluster"]}`. Recomputed
from the stored target-prefix rows, its t>=2 CI is `{i_ge2}` and its t>=3 CI
is `{i_ge3}`; these are offline re-aggregations, not new independent estimates.

The mechanical v2 classification is `{pattern}` and recommendation is
`{verdict}`. If leave-one-route-out remains near zero or its CI includes zero,
the evidence supports transition-entangled representation rather than an
abstract current-state code. If it is positive with a trajectory-cluster CI
above zero, IID and OOD movement both support a more abstract representation.
Native movement is an independent axis: positive native shift supports actual
state readout influence; absent native shift despite hidden movement is a
representation-readout gap.

## Files

- `native_counterfactual_results.csv` and `native_counterfactual_summary.json`
- `ood_state_decoder_results.csv`
- `ood_counterfactual_shift.csv`
- `ood_state_decoder_summary.json`
- `mechanism_final_verdict.md` (this report)
'''
    (out / "mechanism_final_verdict.md").write_text(report)
    print(json.dumps({"out": str(out), "native_pairs": len(pair_native), "ood_rows": len(pair_ood), "pattern": pattern, "verdict": verdict}, indent=2))


if __name__ == "__main__": main()
