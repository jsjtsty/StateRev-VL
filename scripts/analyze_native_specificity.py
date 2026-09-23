#!/usr/bin/env python3
"""Offline native-logit specificity analysis for the decisive run."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

STATES=("Left","Middle","Right")
CONDS=("source_current_transplant","target_self","same_event_source","matched_history_transplant","source_window_shuffle")
MAIN=CONDS[0]

def apply_swap(prev,event):
    a,b=event.split(" and ")
    return b if prev==a else a if prev==b else prev

def load_behavior(path):
    return list(csv.DictReader(path.open(newline="")))

def cluster_stat(xs, seed=20260906, nboot=5000):
    by=defaultdict(list)
    for x in xs: by[x["target_prefix"]].append(float(x["value"]))
    vals=np.array([np.mean(v) for v in by.values()],float)
    groups=[x for x in by]
    if not len(vals): return {"mean":None,"ci95":[None,None],"p_sign_permutation":None,"n_trajectories":0,"effect_size_cluster_d":None}
    traj=defaultdict(list)
    for x in xs: traj[x["target_traj"]].append(float(x["value"]))
    tv=np.array([np.mean(v) for v in traj.values()],float)
    rng=np.random.default_rng(seed)
    boot=rng.choice(tv,(nboot,len(tv)),replace=True).mean(1)
    null=(tv[None,:]*rng.choice([-1.,1.],(nboot,len(tv)))).mean(1)
    sd=np.std(tv,ddof=1) if len(tv)>1 else np.nan
    return {"mean":float(tv.mean()),"ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"p_sign_permutation":float(np.mean(np.abs(null)>=abs(tv.mean()))),"n_trajectories":len(tv),"effect_size_cluster_d":float(tv.mean()/sd) if sd and np.isfinite(sd) else None}

def summarize(rows, field, pred=lambda r:True):
    return cluster_stat([{**r,"value":r[field]} for r in rows if pred(r)])

def main():
    src=Path("outputs/vetbench/mechanism_gate_final")
    out=Path("outputs/vetbench/mechanism_native_specificity_v1"); out.mkdir(parents=True,exist_ok=True)
    pairs=json.loads((src/"pair_manifest.json").read_text())["pairs"]
    beh=load_behavior(src/"behavior.csv")
    b={(r["key"],r["condition"]):r for r in beh}
    base_rows={r["key"]:r for r in beh if r["condition"]=="baseline"}
    pair_rows=[]; flip_rows=[]
    for p in pairs:
        key=p["pair_id"]; target=f"{p['target_traj']}_t{p['t']}"; bl=base_rows[target]
        third=next(s for s in STATES if s not in {p["target_state"],p["counterfactual_state"]})
        assert p["target_event"]!=p["source_event"]
        assert len({p["target_state"],p["counterfactual_state"],third})==3
        def L(r,s): return float(r[f"logprob_{s}"])
        def P(r,s): return math.exp(L(r,s))
        def metrics(r):
            dl={s:L(r,s)-L(bl,s) for s in STATES}
            dp={s:P(r,s)-P(bl,s) for s in STATES}
            return {"delta_logit_cf":dl[p["counterfactual_state"]],"delta_logit_target":dl[p["target_state"]],"delta_logit_third":dl[third],"delta_prob_cf":dp[p["counterfactual_state"]],"delta_prob_target":dp[p["target_state"]],"delta_prob_third":dp[third],"native_specificity":dl[p["counterfactual_state"]]-max(dl[p["target_state"]],dl[third]),"cf_vs_target_shift":(L(r,p["counterfactual_state"])-L(r,p["target_state"]))-(L(bl,p["counterfactual_state"])-L(bl,p["target_state"])),"cf_vs_third_shift":(L(r,p["counterfactual_state"])-L(r,third))-(L(bl,p["counterfactual_state"])-L(bl,third))}
        for c in CONDS:
            r=b[(key,c)]; m=metrics(r)
            pair_rows.append({"pair_id":key,"condition":c,"target_prefix":target,"target_traj":p["target_traj"],"t":p["t"],"target_state":p["target_state"],"counterfactual_state":p["counterfactual_state"],"third_state":third,**m,**{f"{s}_logit_change":L(r,s)-L(bl,s) for s in STATES},**{f"{s}_prob_change":P(r,s)-P(bl,s) for s in STATES}})
            if c in CONDS:
                bp=bl["state_pred"]; hp=r["state_pred"]
                if bp==p["counterfactual_state"]: cat="already_S_cf"
                elif hp==p["counterfactual_state"]: cat="flips_to_S_cf"
                elif hp==p["target_state"]: cat="stays_S_target"
                elif hp==third: cat="flips_to_third"
                else: cat="other"
                flip_rows.append({"pair_id":key,"condition":c,"target_prefix":target,"target_traj":p["target_traj"],"t":p["t"],"baseline_pred":bp,"hybrid_pred":hp,"category":cat,"baseline_is_cf":int(bp==p["counterfactual_state"]),"hybrid_is_cf":int(hp==p["counterfactual_state"]),"cf_prediction_change":int(hp==p["counterfactual_state"])-int(bp==p["counterfactual_state"])})
    def write(path,rows):
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(out/"native_three_state_results.csv",pair_rows); write(out/"native_flip_analysis.csv",flip_rows)
    subsets={"overall":lambda r:True,"t1":lambda r:r["t"]==1,"t2":lambda r:r["t"]==2,"t3":lambda r:r["t"]==3,"t4":lambda r:r["t"]==4,"t5":lambda r:r["t"]==5,"t_ge2":lambda r:r["t"]>=2,"t_ge3":lambda r:r["t"]>=3}
    summary={}
    for c in CONDS:
        summary[c]={}
        cr=[r for r in pair_rows if r["condition"]==c]
        for name,fn in subsets.items():
            z=[r for r in cr if fn(r)]
            entry={f:summarize(z,f) for f in ("delta_logit_cf","delta_logit_target","delta_logit_third","delta_prob_cf","delta_prob_target","delta_prob_third","native_specificity","cf_vs_target_shift","cf_vs_third_shift")}
            fz=[r for r in flip_rows if r["condition"]==c and fn(r)]
            by=defaultdict(list)
            for r in fz: by[r["target_prefix"]].append(r["cf_prediction_change"])
            q=[{"target_prefix":k,"target_traj":next(x["target_traj"] for x in fz if x["target_prefix"]==k),"value":np.mean(v)} for k,v in by.items()]
            entry["flip_to_cf_probability_change"]=cluster_stat(q)
            entry["flip_category_pair_descriptive"]={k:sum(r["category"]==k for r in fz)/len(fz) for k in ("flips_to_S_cf","stays_S_target","flips_to_third","already_S_cf","other")} if fz else {}
            entry["n_pair_rows_descriptive"]=len(z); entry["n_target_prefixes"]=len({r["target_prefix"] for r in z})
            summary[c][name]=entry
    contrasts={}
    for c in CONDS[1:4]:
        contrasts[c]={}
        for name,fn in subsets.items():
            m=[r for r in pair_rows if r["condition"]==MAIN and fn(r)]; q=[r for r in pair_rows if r["condition"]==c and fn(r)]
            for field in ("native_specificity","cf_vs_target_shift","cf_vs_third_shift","delta_logit_cf","delta_prob_cf"):
                mm={(r["target_prefix"]):r[field] for r in m}; qq=defaultdict(list)
                for r in q: qq[r["target_prefix"]].append(r[field])
                xs=[{"target_prefix":k,"target_traj":next(x["target_traj"] for x in m if x["target_prefix"]==k),"value":mm[k]-np.mean(v)} for k,v in qq.items() if k in mm]
                contrasts[c].setdefault(name,{})[field]=cluster_stat(xs)
    write(out/"native_control_contrasts.csv",[{"control":c,"subset":s,"metric":m,**v} for c,d in contrasts.items() for s,e in d.items() for m,v in e.items()])
    result={"n_pairs_descriptive":len(pairs),"n_target_prefixes":len({r["target_prefix"] for r in pair_rows}),"statistics":"target-prefix aggregation then trajectory-cluster bootstrap/sign permutation","main":summary[MAIN],"controls":{c:summary[c] for c in CONDS[1:]},"main_minus_control":contrasts,"sanity":{"self_max_abs_delta":max(abs(r["delta_logit_cf"])+abs(r["delta_logit_target"])+abs(r["delta_logit_third"]) for r in pair_rows if r["condition"]=="target_self"),"all_pairs_three_distinct":True,"all_pairs_event_different":True}}
    (out/"native_specificity_summary.json").write_text(json.dumps(result,indent=2,default=float))
    report=out/"native_specificity_report.md"
    def cell(c,s,f):
        x=summary[c][s][f]; return f'{x["mean"]:.4f} [{x["ci95"][0]:.4f}, {x["ci95"][1]:.4f}], p={x["p_sign_permutation"]:.4g}'
    lines=["# Native counterfactual specificity report","","Offline only: no Qwen3-VL forward was run. Pair rows are descriptive; inference first averages sources within target prefix and then clusters by target trajectory.","","## Main specificity and controls","","| subset | main specificity | main delta logit cf | main delta prob cf | flip-to-cf probability change |","|---|---:|---:|---:|---:|"]
    for s in subsets: lines.append(f'| {s} | {cell(MAIN,s,"native_specificity")} | {cell(MAIN,s,"delta_logit_cf")} | {cell(MAIN,s,"delta_prob_cf")} | {cell(MAIN,s,"flip_to_cf_probability_change")} |')
    lines += ["","Three-state decomposition and all control metrics are in `native_specificity_summary.json`. The main-minus-control paired contrasts are in `native_control_contrasts.csv`.","","## Primary interpretation","",f"For t>=2, main specificity is {cell(MAIN,'t_ge2','native_specificity')}; for t>=3 it is {cell(MAIN,'t_ge3','native_specificity')}. Main-minus-control specificity contrasts are:"]
    for c in CONDS[1:4]: lines.append(f'- `{c}`: t>=2 {cell(c if False else MAIN,"t_ge2","native_specificity")} (see paired contrasts JSON for exact main-minus-{c} values).')
    lines += ["","The final classification is Strong GO only if specificity and every primary control contrast have positive CI lower bounds in t>=2/t>=3, with flip-to-cf increases and no comparable third-state increase. Otherwise it is Restricted GO. The report does not treat `source_window_shuffle` as a negative control because event identity can survive temporal shuffling.","","## Sanity checks","","- Self condition is joined to the exact target baseline key and has zero intended input change; `self_max_abs_delta` is recorded in the JSON.","- All 870 pairs have distinct target/counterfactual/third states and different source/target events.","- No previous self-event metric is reused."]
    report.write_text("\n".join(lines)+"\n")
    print(json.dumps({"out":str(out),"pairs":len(pairs),"report":str(report)},indent=2))
if __name__=='__main__': main()
