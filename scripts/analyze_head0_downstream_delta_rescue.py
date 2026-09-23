#!/usr/bin/env python3
"""Offline trajectory-cluster analysis for composable Head0/L32 delta rescue."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20260911
CONDS=("baseline","head0_only","l32_delta_only","head0_l32_delta")
DONORS=("strong_same_state","different_event")
BETAS=(0.25,0.5,1.0)
SUBSETS=("overall","t_ge2","t_ge3","weak","baseline_incorrect")

def sub(d,n):
    if n=="overall": return d
    if n=="t_ge2": return d[d.t>=2]
    if n=="t_ge3": return d[d.t>=3]
    if n=="weak": return d[d.strength_group=="weak"]
    if n=="baseline_incorrect": return d[d.baseline_gt_margin<0]
    raise KeyError(n)
def traj(d,m):
    x=d.groupby(["target_prefix","target_trajectory"],as_index=False)[m].mean()
    return x.groupby("target_trajectory")[m].mean()
def stat(v):
    v=pd.Series(v,dtype=float).dropna().to_numpy()
    if not len(v): return {"mean":None,"ci95":[None,None],"p_sign_permutation":None,"n_trajectories":0}
    rng=np.random.default_rng(SEED); b=rng.choice(v,(20000,len(v)),replace=True).mean(1); null=(v*rng.choice([-1.,1.],(20000,len(v)))).mean(1); m=float(v.mean())
    return {"mean":m,"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],"p_sign_permutation":float((np.abs(null)>=abs(m)).mean()),"n_trajectories":int(len(v))}
def rate(d,m):
    x=d[~d.baseline_correct] if m=="wrong_to_correct" else d[d.baseline_correct]
    return traj(x.assign(_rate=x[m].astype(float)),"_rate")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("outputs/vetbench/head0_downstream_delta_rescue_v1")); ap.add_argument("--num-shards",type=int,default=8); a=ap.parse_args(); root=a.out
    m=json.loads((root/"delta_rescue_manifest.json").read_text()); files=[root/"shards"/f"delta_rescue_shard_{i}.csv" for i in range(a.num_shards)]; missing=[str(x) for x in files if not x.exists()]
    if missing: raise FileNotFoundError(missing)
    pair=pd.concat([pd.read_csv(x) for x in files],ignore_index=True); expected={x["target_prefix"] for x in m["targets"]}
    if set(pair.target_prefix)!=expected or len(pair)!=len(expected)*len(DONORS)*len(BETAS)*len(CONDS): raise AssertionError("coverage/row count mismatch")
    if pair.duplicated(["target_prefix","donor_type","beta","condition"]).any(): raise AssertionError("duplicate result key")
    if set(pair.condition)!=set(CONDS) or set(pair.donor_type)!=set(DONORS) or set(pair.beta)!=set(BETAS): raise AssertionError("condition mismatch")
    if pair.groupby(["target_prefix","donor_type","beta"]).baseline_gt_margin.nunique().max()>1: raise AssertionError("baseline mismatch")
    # beta=0 is algebraically tested offline; beta>0 must show at least one nonidentity joint result.
    if float(pair.joint_minus_l32_only_max_abs.max())<=1e-7: raise AssertionError("joint and L32 delta are identical")
    target=pair.groupby(["target_prefix","target_trajectory","t","donor_type","beta","condition","strength_group","baseline_correct","gt_state","baseline_gt_margin"],as_index=False)[["patched_gt_margin","delta_gt_margin","delta_counterfactual_margin","wrong_to_correct","correct_to_wrong","gt_state_specificity"]].mean()
    pair.to_csv(root/"delta_rescue_pair_results.csv",index=False); target.to_csv(root/"delta_rescue_target_results.csv",index=False)
    effects=[]; contrasts=[]
    for donor in DONORS:
      for beta in BETAS:
       dd=target[(target.donor_type==donor)&(target.beta==beta)]
       for ss in SUBSETS:
        sd=sub(dd,ss)
        for c in CONDS:
         cd=sd[sd.condition==c]
         for metric in ("delta_gt_margin","delta_counterfactual_margin","gt_state_specificity"):
          effects.append({"donor_type":donor,"beta":beta,"subset":ss,"condition":c,"metric":metric,"n_target_prefixes":len(cd),**stat(traj(cd,metric))})
         for metric in ("wrong_to_correct","correct_to_wrong"):
          effects.append({"donor_type":donor,"beta":beta,"subset":ss,"condition":c,"metric":metric,"n_target_prefixes":len(cd),**stat(rate(cd,metric))})
        h=traj(sd[sd.condition=="head0_only"],"delta_gt_margin"); l=traj(sd[sd.condition=="l32_delta_only"],"delta_gt_margin"); j=traj(sd[sd.condition=="head0_l32_delta"],"delta_gt_margin")
        aligned=pd.concat([j.rename("joint"),h.rename("head0"),l.rename("l32")],axis=1,join="inner").dropna()
        contrasts.append({"donor_type":donor,"beta":beta,"subset":ss,"metric":"interaction_delta_joint_minus_head0_minus_l32","contrast":"joint-head0-l32",**stat(aligned["joint"]-aligned["head0"]-aligned["l32"])})
        for c in ("head0_l32_delta","l32_delta_only"):
         left=traj(sd[sd.condition=="head0_l32_delta"],"delta_gt_margin"); right=traj(sd[sd.condition==c],"delta_gt_margin"); left,right=left.align(right,join="inner")
         contrasts.append({"donor_type":donor,"beta":beta,"subset":ss,"metric":"condition_difference","contrast":f"joint-{c}",**stat(left-right)})
    summary={"protocol":"target-prefix aggregation then target-trajectory cluster bootstrap/sign permutation; pairs are not independent","counts":m["counts"],"conditions":list(CONDS),"betas":list(BETAS),"effects":effects,"contrasts":contrasts,"sanity":{"max_joint_minus_l32_only":float(pair.joint_minus_l32_only_max_abs.max()),"min_joint_minus_l32_only":float(pair.joint_minus_l32_only_max_abs.min()),"beta_zero_rule":"algebraic/unit-tested, not run as a formal beta row","delta_source":"unmodified baseline target/donor h32"}}
    (root/"delta_rescue_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    def fmt(x):
        mean = x["mean"]
        ci = x["ci95"]
        if mean is None or pd.isna(mean) or ci is None or len(ci) != 2:
            return "NA"
        if ci[0] is None or ci[1] is None or pd.isna(ci[0]) or pd.isna(ci[1]):
            return "NA"
        return f"{mean:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
    lines=["# Composable Head 0 + L32 residual-delta rescue","","## Protocol","","Frozen pairs and L24 Head 0 intervention are unchanged. L32 uses `h32_current + beta * (h32_donor - h32_target)`, with `h32_current` taken after the Head 0 hook for joint rows. Pair rows are descriptive; inference is target-prefix then target-trajectory clustered.","","## Sanity checks","",f"Maximum joint-vs-L32-only logit difference: `{pair.joint_minus_l32_only_max_abs.max():.6g}`; minimum: `{pair.joint_minus_l32_only_max_abs.min():.6g}`. Joint is therefore not identically the block-only intervention. Beta=0 equality with Head0-only is covered algebraically by the unit test and is not an additional model run.",""]
    frame=pd.DataFrame(effects)
    for donor in DONORS:
      lines += [f"## {donor}","","| subset | beta | Head0 | L32 delta | joint | joint wrong-to-correct | joint correct-to-wrong |","|---|---:|---:|---:|---:|---:|---:|"]
      for ss in SUBSETS:
       for beta in BETAS:
        def get(c,m): return frame[(frame.donor_type==donor)&(frame.subset==ss)&(frame.beta==beta)&(frame.condition==c)&(frame.metric==m)].iloc[0]
        def show(c,m): return fmt(get(c,m))
        lines.append(f"| {ss} | {beta:g} | {show('head0_only','delta_gt_margin')} | {show('l32_delta_only','delta_gt_margin')} | {show('head0_l32_delta','delta_gt_margin')} | {show('head0_l32_delta','wrong_to_correct')} | {show('head0_l32_delta','correct_to_wrong')} |")
    lines += ["","## Final decision","","For this additive intervention, a positive interaction is identifiable. Judge it from `interaction_delta_joint_minus_head0_minus_l32` in `delta_rescue_summary.json`, using `t>=2` and `t>=3` as the primary subsets and the trajectory-cluster CI/permutation result.","","If joint margin and categorical correction exceed both single interventions with a positive interaction and no comparable harm, this supports composable Head0-to-downstream cooperation. If joint remains primarily continuous margin movement or the interaction CI includes zero, stop the single-channel rescue line and retain the distributed commitment-bottleneck interpretation.","","Different-event results must be read as a semantic harm control: donor-implied counterfactual movement should be directional, while excessive beta may cause wrong-state flips."]
    (root/"delta_rescue_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"DELTA_RESCUE_ANALYSIS_COMPLETE","rows":len(pair),"targets":len(expected),"max_joint_minus_l32_only":float(pair.joint_minus_l32_only_max_abs.max()),"outputs":["delta_rescue_pair_results.csv","delta_rescue_target_results.csv","delta_rescue_summary.json","delta_rescue_report.md"]},indent=2))
if __name__=="__main__": main()
