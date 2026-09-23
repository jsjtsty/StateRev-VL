#!/usr/bin/env python3
"""Offline analysis for the held-out path-dependence experiment.

Rows are descriptive pair observations.  Inference first averages repeated
observations within a target prefix, then treats target trajectory as the
independent cluster.  This script never loads a model.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CONDS = ("target_real", "donor_history_target_current",
         "target_history_donor_current", "donor_real")
SEED = 20260913

def clustered(v, seed=SEED):
    v = pd.Series(v, dtype=float).dropna().to_numpy()
    if not len(v):
        return {"mean": None, "ci95": [None, None], "p_sign_permutation": None,
                "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, (20000, len(v)), replace=True).mean(1)
    null = (v[None, :] * rng.choice([-1., 1.], (20000, len(v)))).mean(1)
    mean = float(v.mean())
    return {"mean": mean,
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "p_sign_permutation": float((np.abs(null) >= abs(mean)).mean()),
            "n_trajectories": int(len(v))}

def prefix_then_trajectory(d, metric):
    p = d.groupby(["target_prefix", "target_trajectory"], as_index=False)[metric].mean()
    return p.groupby("target_trajectory", as_index=False)[metric].mean()

def subset(d, name):
    if name == "overall": return d
    if name == "t_ge3": return d[d.t >= 3]
    if name.startswith("t="): return d[d.t == int(name[2:])]
    raise KeyError(name)

def fmt(x):
    if x["mean"] is None: return "NA"
    a, b = x["ci95"]
    return f"{x['mean']:+.3f} [{a:+.3f},{b:+.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/vetbench/path_dependence_v1"))
    ap.add_argument("--num-shards", type=int, default=8)
    a = ap.parse_args(); root = a.out; manifest = json.loads((root / "path_dependence_manifest.json").read_text())
    files = [root / "shards" / f"path_shard_{i}.csv" for i in range(a.num_shards)]
    missing = [str(x) for x in files if not x.exists()]
    if missing: raise FileNotFoundError("missing path shards: " + ", ".join(missing))
    pair = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    expected = {p["pair_id"] for p in manifest["pairs"]}
    if set(pair.pair_id) != expected: raise AssertionError("pair coverage mismatch")
    if pair.duplicated(["pair_id", "condition"]).any(): raise AssertionError("duplicate condition row")
    if set(pair.condition) != set(CONDS): raise AssertionError("condition coverage mismatch")
    if pair.groupby("pair_id").size().min() != 4: raise AssertionError("incomplete pair")
    split = json.loads(Path(manifest["split"]).read_text())
    if not set(pair.target_trajectory) <= set(split["validation_trajectories"]): raise AssertionError("target split violation")
    if not set(pair.donor_trajectory) <= set(split["discovery_trajectories"]): raise AssertionError("donor split violation")
    if (pair.target_trajectory == pair.donor_trajectory).any(): raise AssertionError("trajectory overlap")
    # Render invariants: changing history/current pixels must not change the
    # textual prompt, sequence length, video grid, or pixel tensor shape.
    for pid, g in pair.groupby("pair_id"):
        for col in ("condition_input_ids_sha256", "condition_video_grid_thw",
                    "condition_pixel_shape", "condition_prompt_text_sha256"):
            if g[col].nunique(dropna=False) != 1:
                raise AssertionError(f"{col} changed within pair {pid}")
        if g.condition_pixel_sha256.nunique(dropna=False) < 2:
            raise AssertionError(f"no rendered visual change in pair {pid}")
    pair.to_csv(root / "path_dependence_results.csv", index=False)

    metrics = ["gt_margin", "delta_gt_margin_vs_target_real", "state_probe_L24_gt",
               "state_probe_L32_gt", "state_probe_L36_gt", "event_probe_L24_current",
               "event_probe_L32_current", "event_probe_L36_current"]
    available = [m for m in metrics if m in pair.columns]
    summaries = []; contrasts = []
    subsets = ["overall", "t=3", "t=4", "t=5", "t_ge3"]
    for ss in subsets:
        d = subset(pair, ss)
        for metric in available:
            for c in CONDS:
                x = prefix_then_trajectory(d[d.condition == c], metric)
                summaries.append({"subset": ss, "condition": c, "metric": metric,
                                  "n_target_prefixes": int(d[d.condition == c].target_prefix.nunique()), **clustered(x[metric])})
        # Two factorial history contrasts, aligned at trajectory level after
        # the required target-prefix averaging.
        for metric in available:
            by = {}
            for c in CONDS:
                by[c] = prefix_then_trajectory(d[d.condition == c], metric).set_index("target_trajectory")[metric]
            for left, right, label in (("donor_history_target_current", "target_real", "donor_history_minus_target_real"),
                                       ("donor_real", "target_history_donor_current", "donor_real_minus_target_history_donor_current")):
                x, y = by[left].align(by[right], join="inner")
                st = clustered(x - y)
                contrasts.append({"subset": ss, "metric": metric, "contrast": label,
                                  "n_target_trajectories": len(x), **st})
    pd.DataFrame(summaries).to_csv(root / "path_dependence_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(root / "path_dependence_contrasts.csv", index=False)
    summary = {"protocol": "target-prefix mean, then target-trajectory cluster bootstrap/sign permutation; pair rows descriptive",
               "counts": manifest["counts"], "conditions": list(CONDS),
               "available_metrics": available, "summaries": summaries, "contrasts": contrasts,
               "split": {"discovery": split["discovery_trajectories"], "validation": split["validation_trajectories"]}}
    (root / "path_dependence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Held-out path dependence", "", "No model was loaded by this analysis.",
             "", f"Fixed held-out manifest: {manifest['counts']['pairs']} pairs, "
             f"{manifest['counts']['target_trajectories']} target trajectories, t>=3.",
             "Pair rows are descriptive. All intervals and sign permutations use target-prefix aggregation followed by target-trajectory clustering.", "",
             "## History contrasts", "",
             "The primary contrasts hold the current-event window fixed: donor-history/target-current minus target-real, and donor-real minus target-history/donor-current.", "",
             "| subset | metric | contrast | estimate [95% CI] | trajectories | p(sign) |", "|---|---|---|---:|---:|---:|"]
    ct = pd.DataFrame(contrasts)
    for _, r in ct.iterrows():
        lines.append(f"| {r['subset']} | {r['metric']} | {r['contrast']} | {fmt(r)} | {int(r['n_trajectories'])} | {r['p_sign_permutation']:.4f} |")
    lines += ["", "## Interpretation", "", "A reproducible non-zero history contrast under both current-window controls would reject a purely Markov readout of (S_prev, E_t) for the tested internal endpoints. A null contrast would support approximate state sufficiency, subject to the small held-out sample and decoder limitations.", "", "The t>=3 result is the primary mechanism test; this manifest intentionally does not use t=1 or t=2."]
    (root / "path_dependence_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PATH_ANALYSIS_COMPLETE", "rows": len(pair), "outputs": ["path_dependence_summary.json", "path_dependence_report.md"]}, indent=2))

if __name__ == "__main__": main()
