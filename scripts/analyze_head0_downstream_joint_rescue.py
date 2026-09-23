#!/usr/bin/env python3
"""Trajectory-clustered offline analysis for frozen Head0+downstream rescue."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260910
CONDITIONS = ("baseline", "head0_only", "l32_only", "l36_only", "head0_l32", "head0_l36")
DONORS = ("strong_same_state", "different_event")
SUBSETS = ("overall", "t_ge2", "t_ge3", "weak", "baseline_incorrect")

def subset(d, name):
    if name == "overall": return d
    if name == "t_ge2": return d[d.t >= 2]
    if name == "t_ge3": return d[d.t >= 3]
    if name == "weak": return d[d.strength_group == "weak"]
    if name == "baseline_incorrect": return d[d.baseline_gt_margin < 0]
    raise KeyError(name)

def traj_values(d, metric):
    p = d.groupby(["target_prefix", "target_trajectory"], as_index=False)[metric].mean()
    return p.groupby("target_trajectory")[metric].mean()

def stat(v, seed=SEED):
    v = pd.Series(v, dtype=float).dropna().to_numpy()
    if not len(v): return {"mean": None, "ci95": [None, None], "p_sign_permutation": None, "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (20000, len(v)), replace=True).mean(1)
    null = (v * rng.choice([-1., 1.], (20000, len(v)))).mean(1)
    m = float(v.mean())
    return {"mean": m, "ci95": [float(np.quantile(b,.025)), float(np.quantile(b,.975))],
            "p_sign_permutation": float((np.abs(null) >= abs(m)).mean()), "n_trajectories": int(len(v))}

def rate_values(d, kind):
    x = d[~d.baseline_correct] if kind == "wrong_to_correct" else d[d.baseline_correct]
    return traj_values(x.assign(_v=x[kind].astype(float)), "_v")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/vetbench/head0_downstream_joint_rescue_v1"))
    ap.add_argument("--num-shards", type=int, default=8)
    a = ap.parse_args(); root = a.out
    manifest = json.loads((root / "joint_rescue_manifest.json").read_text())
    files = [root / "shards" / f"joint_rescue_shard_{i}.csv" for i in range(a.num_shards)]
    missing = [str(x) for x in files if not x.exists()]
    if missing: raise FileNotFoundError(f"missing shards: {missing}")
    pair = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    expected = {x["target_prefix"] for x in manifest["targets"]}
    if set(pair.target_prefix) != expected: raise AssertionError("target coverage mismatch")
    if len(pair) != len(expected) * len(DONORS) * len(CONDITIONS): raise AssertionError("row count mismatch")
    if pair.duplicated(["target_prefix", "donor_type", "condition"]).any(): raise AssertionError("duplicate rows")
    if set(pair.condition) != set(CONDITIONS) or set(pair.donor_type) != set(DONORS): raise AssertionError("condition mismatch")
    self_rows = pair[pair.condition == "baseline"]
    # baseline is emitted once per donor type; baseline logits must agree.
    if self_rows.groupby("target_prefix")["baseline_gt_margin"].nunique().max() > 1: raise AssertionError("baseline mismatch")
    if pair[pair.joint_logits_reused_from_block_only.astype(bool)]["condition"].isin(["head0_l32", "head0_l36"]).all():
        for layer in (32, 36):
            j = pair[pair.condition == f"head0_l{layer}"]
            b = pair[pair.condition == f"l{layer}_only"]
            key = ["target_prefix", "donor_type"]
            x = j.merge(b, on=key, suffixes=("_j", "_b"))
            if len(x) and not np.allclose(x.patched_gt_margin_j, x.patched_gt_margin_b, atol=1e-6):
                raise AssertionError(f"joint/block equality failed L{layer}")
    numeric = ["baseline_gt_margin", "patched_gt_margin", "delta_gt_margin", "delta_counterfactual_margin",
               "wrong_to_correct", "correct_to_wrong", "gt_state_specificity"]
    target = pair.groupby(["target_prefix", "target_trajectory", "t", "donor_type", "condition",
                           "strength_group", "baseline_correct", "gt_state", "baseline_gt_margin"],
                          as_index=False, dropna=False)[numeric[1:]].mean()
    pair.to_csv(root / "joint_rescue_pair_results.csv", index=False)
    target.to_csv(root / "joint_rescue_target_results.csv", index=False)
    rows=[]; contrasts=[]
    for donor in DONORS:
      dd = target[target.donor_type == donor]
      for ss in SUBSETS:
        sd = subset(dd, ss)
        vals = {c: traj_values(sd[sd.condition == c], "delta_gt_margin") for c in CONDITIONS}
        for c in CONDITIONS:
          rows.append({"donor_type": donor, "subset": ss, "condition": c, "metric": "delta_gt_margin",
                       "n_target_prefixes": int(len(sd[sd.condition == c])), **stat(vals[c])})
          for metric in ("delta_counterfactual_margin", "gt_state_specificity"):
            rows.append({"donor_type": donor, "subset": ss, "condition": c, "metric": metric,
                         "n_target_prefixes": int(len(sd[sd.condition == c])), **stat(traj_values(sd[sd.condition == c], metric))})
          for metric in ("wrong_to_correct", "correct_to_wrong"):
            rows.append({"donor_type": donor, "subset": ss, "condition": c, "metric": metric,
                         "n_target_prefixes": int(len(sd[sd.condition == c])), **stat(rate_values(sd[sd.condition == c], metric))})
        main = vals["head0_only"]
        for c in ("l32_only", "l36_only", "head0_l32", "head0_l36"):
          for metric in ("delta_gt_margin", "delta_counterfactual_margin", "gt_state_specificity"):
            left = traj_values(sd[sd.condition == "head0_only"], metric)
            right = traj_values(sd[sd.condition == c], metric)
            left, right = left.align(right, join="inner")
            contrasts.append({"donor_type": donor, "subset": ss, "contrast": f"{c}-head0_only",
                              "metric": metric, **stat(right-left)})
        # Interaction: joint - Head0 - block, calculated after prefix aggregation.
        for layer in (32,36):
          j=traj_values(sd[sd.condition == f"head0_l{layer}"], "delta_gt_margin")
          h=traj_values(sd[sd.condition == "head0_only"], "delta_gt_margin")
          b=traj_values(sd[sd.condition == f"l{layer}_only"], "delta_gt_margin")
          j,h = j.align(h, join="inner"); j,b = j.align(b, join="inner")
          contrasts.append({"donor_type": donor, "subset": ss, "contrast": f"interaction_L{layer}",
                            "metric": "delta_joint_minus_head0_minus_block", **stat(j-h-b)})
    summary={"protocol": "target-prefix aggregation then target-trajectory cluster bootstrap/sign permutation; pairs are not independent",
             "counts": manifest["counts"], "conditions": list(CONDITIONS), "donors": list(DONORS),
             "effects": rows, "contrasts": contrasts,
             "structural_occlusion": manifest["structural_occlusion"],
             "self_baseline_max_abs_duplicate_difference": float(self_rows.groupby("target_prefix").baseline_gt_margin.nunique().max()-1)}
    (root / "joint_rescue_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines=["# Frozen Head 0 + downstream joint rescue", "", "## Decision scope", "",
           "This is a frozen intervention test. Pair rows are descriptive; inference aggregates target prefixes and clusters by target trajectory.", "",
           "## Structural caveat", "",
           manifest["structural_occlusion"]["reason"], "",
           "Consequently, under the requested full final-token block-output replacement, Head0+L32/L36 is expected to equal L32/L36-only. A positive interaction is not identifiable with this intervention definition.", "",
           "## Results", "", "See `joint_rescue_summary.json` for all subsets, CIs, and sign permutations.", ""]
    for donor in DONORS:
      lines += [f"### {donor}", "", "| subset | condition | delta GT margin | wrong-to-correct | correct-to-wrong |", "|---|---|---:|---:|---:|"]
      frame=pd.DataFrame(rows); frame=frame[(frame.donor_type==donor)&(frame.metric.isin(["delta_gt_margin","wrong_to_correct","correct_to_wrong"]))]
      for ss in SUBSETS:
       for c in CONDITIONS:
        def fmt(metric):
          q=frame[(frame.subset==ss)&(frame.condition==c)&(frame.metric==metric)].iloc[0]
          return f"{q['mean']:+.3f} [{q['ci95'][0]:+.3f},{q['ci95'][1]:+.3f}]" if pd.notna(q['mean']) else "NA"
        lines.append(f"| {ss} | {c} | {fmt('delta_gt_margin')} | {fmt('wrong_to_correct')} | {fmt('correct_to_wrong')} |")
    lines += ["", "## Final interpretation", "",
              "A joint condition cannot demonstrate positive synergy here because the downstream full-token replacement overwrites the earlier Head 0 contribution. If joint equals block-only as expected, this is evidence of downstream occlusion, not evidence that Head 0 is irrelevant.", "",
              "If L32/L36 improves margin but categorical crossings remain sparse, stop the single-channel rescue line and report a distributed commitment bottleneck. If a true compositional test is required, it must be separately preregistered with an additive downstream delta intervention; that is outside this frozen experiment."]
    (root / "joint_rescue_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"JOINT_RESCUE_ANALYSIS_COMPLETE","rows":len(pair),"targets":len(expected),"outputs":["joint_rescue_pair_results.csv","joint_rescue_target_results.csv","joint_rescue_summary.json","joint_rescue_report.md"]},indent=2))

if __name__ == "__main__": main()
