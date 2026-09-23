#!/usr/bin/env python3
"""Offline clustered analysis for history-to-native-readout mediation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

LAYERS=(24,32,36); CONDITIONS=("history","current_window","donor_real")
SEED=20260914

def stat(v):
    v=pd.Series(v,dtype=float).dropna().to_numpy()
    if len(v)==0: return {"mean":None,"ci95":[None,None],"p_sign_permutation":None,"n_trajectories":0}
    rng=np.random.default_rng(SEED); boot=rng.choice(v,(20000,len(v)),True).mean(1); null=(v[None,:]*rng.choice([-1.,1.],(20000,len(v)))).mean(1); m=float(v.mean())
    return {"mean":m,"ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"p_sign_permutation":float((np.abs(null)>=abs(m)).mean()),"n_trajectories":int(len(v))}

def traj(d,metric):
    p=d.groupby(["pair_id","target_prefix","target_trajectory"],as_index=False)[metric].mean()
    return p.groupby("target_trajectory")[metric].mean()

def fmt(x):
    if x["mean"] is None:return "NA"
    return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("outputs/vetbench/history_readout_mediation_v1")); ap.add_argument("--num-shards",type=int,default=8); a=ap.parse_args(); root=a.out
    m=json.loads((root/"history_readout_manifest.json").read_text()); files=[root/"shards"/f"mediation_shard_{i}.csv" for i in range(a.num_shards)]; miss=[str(x) for x in files if not x.exists()]
    if miss: raise FileNotFoundError("missing shards: "+", ".join(miss))
    d=pd.concat([pd.read_csv(x) for x in files],ignore_index=True); expected={p["pair_id"] for p in m["pairs"]}
    if set(d.pair_id)!=expected or d.duplicated(["pair_id","condition","direction","layer"]).any(): raise AssertionError("coverage or duplicate key mismatch")
    if len(d)!=len(expected)*len(CONDITIONS)*2*len(LAYERS): raise AssertionError("unexpected row count")
    if set(d.condition)!=set(CONDITIONS) or set(d.layer)!=set(LAYERS) or set(d.direction)!={"sufficiency","necessity"}: raise AssertionError("condition/layer/direction mismatch")
    for pid,g in d.groupby("pair_id"):
        assert g.input_ids_sha256.nunique()==1 and g.video_grid_thw.nunique()==1 and g.prompt_hash.nunique()==1
    d.to_csv(root/"history_readout_mediation_results.csv",index=False)
    # Signed causal effects: sufficiency injects source into target; necessity
    # restores target into source. Both are positive when history/source effect
    # reaches the endpoint in the expected direction.
    d["native_suff_effect"]=d.patched_gt_logit-d.clean_target_gt_logit
    d["native_necessity_effect"]=d.clean_source_gt_logit-d.patched_gt_logit
    d["state_suff_effect"]=d.state_probe_gt-d.target_state_probe_gt
    d["state_necessity_effect"]=d.source_state_probe_gt-d.state_probe_gt
    d["event_suff_effect"]=d.event_probe_current-d.target_event_probe_current
    d["event_necessity_effect"]=d.source_event_probe_current-d.event_probe_current
    summaries=[]; contrasts=[]
    metrics=["native_suff_effect","native_necessity_effect","state_suff_effect","state_necessity_effect","event_suff_effect","event_necessity_effect","native_cf_margin_change","self_patch_max_logit_error"]
    for subset,mask in [("overall",np.ones(len(d),bool)),("t=3",d.t==3),("t=4",d.t==4),("t=5",d.t==5),("t_ge3",d.t>=3)]:
      x=d[mask]
      for c in CONDITIONS:
       for direction in ("sufficiency","necessity"):
        for layer in LAYERS:
         q=x[(x.condition==c)&(x.direction==direction)&(x.layer==layer)]
         for metric in metrics:
          if metric in q: summaries.append({"subset":subset,"condition":c,"direction":direction,"layer":layer,"metric":metric,"n_pairs":int(q.pair_id.nunique()),**stat(traj(q,metric))})
      for layer in LAYERS:
       for metric in ("native_suff_effect","native_necessity_effect","state_suff_effect","state_necessity_effect","event_suff_effect","event_necessity_effect"):
        for c in ("history","current_window","donor_real"):
         q=x[(x.condition==c)&(x.layer==layer)]; val=traj(q,metric); contrasts.append({"subset":subset,"layer":layer,"condition":c,"metric":metric,**stat(val)})
    pd.DataFrame(summaries).to_csv(root/"history_readout_mediation_summary.csv",index=False); pd.DataFrame(contrasts).to_csv(root/"history_readout_mediation_contrasts.csv",index=False)
    summary={"protocol":"target-prefix aggregation then target-trajectory cluster bootstrap/sign permutation; pair rows descriptive","counts":m["counts"],"layers":list(LAYERS),"conditions":list(CONDITIONS),"summaries":summaries,"sanity":{"max_self_patch_error":float(d.self_patch_max_logit_error.max()),"self_patch_error_ci":stat(traj(d,"self_patch_max_logit_error"))}}
    (root/"history_readout_mediation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# History-to-readout causal mediation","","All results use fixed held-out path pairs. Inference is target-prefix aggregation followed by target-trajectory clustered bootstrap and sign permutation.","","## Primary t>=3 results","","| layer | condition | endpoint | sufficiency | necessity |","|---:|---|---|---:|---:|"]
    tab=pd.DataFrame(summaries)
    for layer in LAYERS:
      for c in CONDITIONS:
       def get(metric,direction):
        z=tab[(tab.subset=="t_ge3")&(tab.layer==layer)&(tab.condition==c)&(tab.direction==direction)&(tab.metric==metric)]
        return fmt(z.iloc[0]) if len(z) else "NA"
       lines.append(f"| {layer} | {c} | native GT logit | {get('native_suff_effect','sufficiency')} | {get('native_necessity_effect','necessity')} |")
       lines.append(f"| {layer} | {c} | state probe GT probability | {get('state_suff_effect','sufficiency')} | {get('state_necessity_effect','necessity')} |")
       lines.append(f"| {layer} | {c} | event probe current probability | {get('event_suff_effect','sufficiency')} | {get('event_necessity_effect','necessity')} |")
    lines += ["","## Interpretation","","A positive state-probe effect with a positive native effect would indicate that history-sensitive representation reaches the native readout. A positive state-probe effect with a null native effect supports a representation-to-readout bottleneck. The `current_window` condition is the main visual control; `donor_real` is a combined history+current reference and is not a matched-history negative control.","","## Sanity","",f"Maximum self-patch native logit error: `{d.self_patch_max_logit_error.max():.6g}`."]
    (root/"history_readout_mediation_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"HISTORY_MEDIATION_ANALYSIS_COMPLETE","rows":len(d),"outputs":["history_readout_mediation_summary.json","history_readout_mediation_report.md"]},indent=2))
if __name__=="__main__":main()
