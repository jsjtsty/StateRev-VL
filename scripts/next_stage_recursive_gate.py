#!/usr/bin/env python3
"""Offline, leakage-free recursive-state and correction-gate analysis.

Consumes the existing recursive event-probability cache.  All threshold
selection is confined to discovery trajectories; the validation report only
loads the frozen JSON configuration.  This intentionally does not build a
second model-input path or rerun a forward pass.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

STATES = ("Left", "Middle", "Right")
SEED = 20260916

def cstat(values, groups, n=8000):
    x = pd.DataFrame({"v": np.asarray(values, float), "g": groups}).dropna()
    if not len(x): return {"mean":None,"ci95":[None,None],"p_sign_permutation":None,"n_trajectories":0}
    z=x.groupby("g").v.mean().to_numpy(); rng=np.random.default_rng(SEED)
    b=z[rng.integers(0,len(z),(n,len(z)))].mean(1)
    null=(z*rng.choice((-1.,1.),(n,len(z)))).mean(1)
    return {"mean":float(z.mean()),"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],
            "p_sign_permutation":float((np.abs(null)>=abs(z.mean())).mean()),"n_trajectories":int(len(z))}

def native_margin(r):
    z=sorted((float(r[f"logprob_{s}"]) for s in STATES),reverse=True)
    return z[0]-z[1]

def augment(d):
    d=d.copy()
    d["native_top2_margin"]=d.apply(native_margin,axis=1)
    d["event_confidence"]=d[[c for c in d if c.startswith("event_prob_")]].max(axis=1)
    d["disagreement"]=d.recursive_prob_pred.ne(d.native_state_pred)
    return d

def choose(d, split, old):
    disc=augment(d[d.trajectory_id.isin(split["discovery_trajectories"])])
    # Select a small predeclared grid by trajectory-level net gain, while
    # requiring C->W no greater than 2 percentage points.  Outcomes are used
    # only here, never in any feature or validation decision.
    grid=[]
    for conf in (.50,.60,.70,.80,.90):
      for margin in (.05,.15,.30,.50,.80,1.20):
        trig=disc.disagreement & (disc.recursive_prob_conf>=conf) & (disc.native_top2_margin<=margin)
        out=np.where(trig,disc.recursive_prob_pred,disc.native_state_pred)
        w2c=(disc.native_state_pred.ne(disc.gt_state)&(out==disc.gt_state)).astype(float)
        c2w=(disc.native_state_pred.eq(disc.gt_state)&(out!=disc.gt_state)).astype(float)
        # independent unit is trajectory
        gain=pd.DataFrame({"v":w2c-c2w,"g":disc.trajectory_id}).groupby("g").v.mean().mean()
        harm=pd.DataFrame({"v":c2w,"g":disc.trajectory_id}).groupby("g").v.mean().mean()
        grid.append({"recursive_confidence_threshold":conf,"native_top2_margin_threshold":margin,
                     "discovery_net_gain":float(gain),"discovery_correct_to_wrong":float(harm),
                     "discovery_trigger_rate":float(trig.mean())})
    eligible=[x for x in grid if x["discovery_correct_to_wrong"]<=.02]
    best=max(eligible or grid,key=lambda x:(x["discovery_net_gain"],-x["discovery_correct_to_wrong"],x["discovery_trigger_rate"]))
    # Existing frozen file had confidence=.5/margin=.25 but used GT margin.
    # We retain numbers only and apply them to observable top-2 margin.
    legacy=old.get("config",{})
    return {"selection":"discovery trajectories only; net gain subject to C->W <= 0.02",
            "feature_definition":"recursive belief confidence + observable native top-2 logprob margin; no GT labels at inference",
            "selected":best,"legacy_frozen_proxy":{"recursive_confidence_threshold":legacy.get("confidence_threshold",.5),
              "native_top2_margin_threshold":legacy.get("native_margin_threshold",.25),"lambda":legacy.get("lambda",2)},"grid":grid}

def apply_mode(d, mode, cfg):
    d=augment(d); sel=cfg["selected"]; old=cfg["legacy_frozen_proxy"]
    if mode=="no_gate": trig=np.zeros(len(d),bool)
    elif mode=="current_frozen_gate_proxy": trig=d.disagreement&(d.recursive_prob_conf>=old["recursive_confidence_threshold"])&(d.native_top2_margin<=old["native_top2_margin_threshold"])
    elif mode=="disagreement_only": trig=d.disagreement
    elif mode=="disagreement_recursive_confidence": trig=d.disagreement&(d.recursive_prob_conf>=sel["recursive_confidence_threshold"])
    elif mode=="disagreement_recursive_confidence_native_margin": trig=d.disagreement&(d.recursive_prob_conf>=sel["recursive_confidence_threshold"])&(d.native_top2_margin<=sel["native_top2_margin_threshold"])
    else: raise KeyError(mode)
    out=np.where(trig,d.recursive_prob_pred,d.native_state_pred)
    d["mode"]=mode; d["gate_trigger"]=trig; d["output_pred"]=out
    d["correct"]=out==d.gt_state
    d["wrong_to_correct"]=(d.native_state_pred!=d.gt_state)&(out==d.gt_state)
    d["correct_to_wrong"]=(d.native_state_pred==d.gt_state)&(out!=d.gt_state)
    d["net_gain"]=d.wrong_to_correct.astype(float)-d.correct_to_wrong.astype(float)
    return d

def metrics(d):
    result={}
    for tag,mask in [("overall",np.ones(len(d),bool)),("t_ge2",d.t>=2),("t_ge3",d.t>=3)]+[(f"t{i}",d.t==i) for i in range(1,6)]:
      x=d[mask]; result[tag]={k:cstat(x[k].astype(float),x.trajectory_id) for k in ("correct","wrong_to_correct","correct_to_wrong","net_gain")}
      result[tag]["trigger_rate"]=cstat(x.gate_trigger.astype(float),x.trajectory_id)
    return result

def recursive_metrics(d, symbolic):
    rows=[]; s={"methods":{},"error_accumulation":{}}
    methods={"native":"state_correct","symbolic":"symbolic_acc","recursive_hard":"hard_correct","recursive_probability":"prob_correct"}
    x=d.merge(symbolic[["key","symbolic_acc"]],left_on="target_prefix",right_on="key",how="left")
    for name,col in methods.items():
      q={}
      for tag,mask in [("overall",np.ones(len(x),bool)),("t_ge2",x.t>=2),("t_ge3",x.t>=3)]+[(f"t{i}",x.t==i) for i in range(1,6)]: q[tag]=cstat(x.loc[mask,col],x.loc[mask,"trajectory_id"])
      s["methods"][name]=q
      y=x.sort_values(["trajectory_id","t"])[["trajectory_id","t",col]].copy(); y["previous_correct"]=y.groupby("trajectory_id")[col].shift(1)
      follow=y[y.t>=2]
      s["error_accumulation"][name]={
        "error_given_previous_error":cstat(1-follow.loc[follow.previous_correct==False,col],follow.loc[follow.previous_correct==False,"trajectory_id"]),
        "error_given_previous_correct":cstat(1-follow.loc[follow.previous_correct==True,col],follow.loc[follow.previous_correct==True,"trajectory_id"]),
        "error_rate_by_t":{str(i):float(1-y.loc[y.t==i,col].mean()) for i in range(1,6)}}
    return s

def plot_curve(d, cfg, out):
    import matplotlib.pyplot as plt
    val=augment(d)
    pts=[]
    for c in (.50,.60,.70,.80,.90):
      for m in (.05,.15,.30,.50,.80,1.20):
       z=apply_mode(val,"disagreement_recursive_confidence_native_margin",{"selected":{"recursive_confidence_threshold":c,"native_top2_margin_threshold":m},"legacy_frozen_proxy":cfg["legacy_frozen_proxy"]})
       mm=metrics(z)["overall"]; pts.append((mm["trigger_rate"]["mean"],mm["net_gain"]["mean"],mm["correct_to_wrong"]["mean"],c,m))
    p=pd.DataFrame(pts,columns=["trigger_rate","net_gain","correct_to_wrong","confidence","margin"]); p.to_csv(out/"validation_gate_curve_points.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(10,4)); sc=ax[0].scatter(p.trigger_rate,p.correct_to_wrong,c=p.net_gain,cmap="coolwarm"); ax[0].set(xlabel="trigger rate",ylabel="risk: correct→wrong");fig.colorbar(sc,ax=ax[0],label="net gain")
    ax[1].scatter(p.trigger_rate,p.net_gain,c=p.correct_to_wrong,cmap="viridis_r");ax[1].set(xlabel="trigger rate",ylabel="net gain (W→C − C→W)");fig.colorbar(ax[1].collections[0],ax=ax[1],label="correct→wrong")
    fig.tight_layout();fig.savefig(out/"validation_coverage_risk_and_gain.png",dpi=180);plt.close(fig)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,default=Path("outputs/vetbench/recursive_state_recovery_v1")); ap.add_argument("--split",type=Path,default=Path("outputs/vetbench/circuit_localization_v1/discovery_validation_split.json")); ap.add_argument("--out",type=Path,default=Path("outputs/vetbench/next_stage_recursive_gate_v1")); a=ap.parse_args()
 out=a.out; out.mkdir(parents=True,exist_ok=False)
 d=pd.read_csv(a.source/"recursive_state_results.csv"); sym=pd.read_csv("outputs/vetbench/composition_analysis_v1/symbolic_composition.csv"); split=json.loads(a.split.read_text()); old=json.loads((a.source/"gate_config.json").read_text())
 cfg=choose(d,split,old); (out/"frozen_gate_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 rec=recursive_metrics(d,sym); (out/"recursive_summary.json").write_text(json.dumps(rec,indent=2)+"\n")
 val=d[d.trajectory_id.isin(split["validation_trajectories"])].copy(); modes=("no_gate","current_frozen_gate_proxy","disagreement_only","disagreement_recursive_confidence","disagreement_recursive_confidence_native_margin")
 allx=pd.concat([apply_mode(val,m,cfg) for m in modes],ignore_index=True); allx.to_csv(out/"validation_gate_rows.csv",index=False)
 gate={"protocol":"thresholds selected on discovery only; frozen validation; trajectory-cluster bootstrap/sign permutation","modes":{m:metrics(allx[allx["mode"]==m]) for m in modes}}
 (out/"validation_gate_summary.json").write_text(json.dumps(gate,indent=2)+"\n"); plot_curve(val,cfg,out)
 def fmt(q): return f"{q['mean']:.3f} [{q['ci95'][0]:.3f},{q['ci95'][1]:.3f}]"
 lines=["# Next-stage recursive state and frozen gate report","","## Recursive state recovery (all 50 trajectories)","","| method | overall | t>=2 | t>=3 | t1 | t2 | t3 | t4 | t5 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for m,q in rec["methods"].items(): lines.append("| "+m+" | "+" | ".join(fmt(q[k]) for k in ("overall","t_ge2","t_ge3","t1","t2","t3","t4","t5"))+" |")
 lines += ["","## Frozen validation gate", "", "`current_frozen_gate_proxy` applies historical numeric thresholds to observable top-2 margin; it does not reproduce the old GT-margin leakage.","", "| mode | accuracy | trigger rate | wrong→correct | correct→wrong | net gain |","|---|---:|---:|---:|---:|---:|"]
 for m in modes:
  q=gate["modes"][m]["overall"];lines.append(f"| {m} | {fmt(q['correct'])} | {fmt(q['trigger_rate'])} | {fmt(q['wrong_to_correct'])} | {fmt(q['correct_to_wrong'])} | {fmt(q['net_gain'])} |")
 lines += ["","Error accumulation diagnostics, all t-specific gate metrics, curve points, and the PNG curve are in the adjacent JSON/CSV/PNG artifacts."]
 (out/"next_stage_recursive_gate_report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"status":"NEXT_STAGE_OFFLINE_COMPLETE","out":str(out),"validation_rows":len(val),"modes":list(modes)},indent=2))
if __name__=="__main__": main()
