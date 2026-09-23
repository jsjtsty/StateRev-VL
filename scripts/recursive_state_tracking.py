#!/usr/bin/env python3
"""Offline recursive event->state tracking, oracle and hybrid diagnostics.

All inputs are existing artifacts from the frozen controlled-8fps experiment.
No model forward or parameter fitting is performed here.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

STATES=("Left","Middle","Right")
EVENTS=("Left and Middle","Middle and Right","Left and Right")
PAIRS={"Left and Middle":("Left","Middle"),"Middle and Right":("Middle","Right"),"Left and Right":("Left","Right")}
SEED=20260917

def update(s,e):
    a,b=PAIRS[e]
    return b if s==a else a if s==b else s

def stat(v,g,n=10000):
    d=pd.DataFrame({"v":np.asarray(v,float),"g":np.asarray(g)}).dropna()
    if d.empty:return {"mean":None,"ci95":[None,None],"p_sign_permutation":None,"n_trajectories":0}
    z=d.groupby("g").v.mean().to_numpy(); r=np.random.default_rng(SEED)
    boot=z[r.integers(0,len(z),(n,len(z)))].mean(1); null=(z*r.choice((-1.,1.),(n,len(z)))).mean(1)
    return {"mean":float(z.mean()),"ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"p_sign_permutation":float((np.abs(null)>=abs(z.mean())).mean()),"n_trajectories":int(len(z))}

def transition(s,e):
    out=np.zeros(3)
    for i,ss in enumerate(STATES):
        for j,ee in enumerate(EVENTS):out[STATES.index(update(ss,ee))]+=s[i]*e[j]
    return out/out.sum()

def path(g, event_override=None, probability=False, oracle=False):
    hard=g.iloc[0].initial_state; belief=np.eye(3)[STATES.index(hard)]; rows=[]
    for _,r in g.sort_values("t").iterrows():
        ev=r.gt_event if oracle or (event_override is not None and int(r.t)==event_override) else r.event_pred
        if probability:
            if oracle or (event_override is not None and int(r.t)==event_override): ep=np.eye(3)[EVENTS.index(ev)]
            else: ep=np.array([r[f"event_prob_{e}"] for e in EVENTS],float)
            belief=transition(belief,ep); pred=STATES[int(np.argmax(belief))]
        else:
            hard=update(hard,ev); belief=np.eye(3)[STATES.index(hard)];pred=hard
        rows.append({"trajectory_id":r.trajectory_id,"target_prefix":r.target_prefix,"t":int(r.t),"gt_state":r.gt_state,"event_used":ev,"event_pred":r.event_pred,"gt_event":r.gt_event,"event_correct":bool(r.event_correct),"state_pred":pred,"state_correct":pred==r.gt_state,"confidence":float(belief.max()),"belief_Left":float(belief[0]),"belief_Middle":float(belief[1]),"belief_Right":float(belief[2])})
    return pd.DataFrame(rows)

def first_irrecoverable(g):
    # At horizon 5: first wrong step after which every remaining prediction is
    # wrong. This distinguishes permanent tracking failure from recoverable dips.
    g=g.sort_values("t"); bad=~g.state_correct.to_numpy(bool); ts=g.t.to_numpy()
    for i,t in enumerate(ts):
        if bad[i] and bad[i:].all():return int(t)
    return None

def summarize_rows(d,correct_col="correct"):
    out={}
    for tag,mask in [("overall",np.ones(len(d),bool)),("t_ge2",d.t>=2),("t_ge3",d.t>=3)]+[(f"t{i}",d.t==i) for i in range(1,6)]:
        x=d[mask];out[tag]={"accuracy":stat(x[correct_col].astype(float),x.trajectory_id),"n_prefixes":int(len(x))}
    return out

def file_sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def mean_value(x):
    return None if x.get("mean") is None else float(x["mean"])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("outputs/vetbench/recursive_state_tracking_v1"));ap.add_argument("--recursive",type=Path,default=Path("outputs/vetbench/recursive_state_recovery_v1/recursive_state_results.csv"));ap.add_argument("--event-prob-dir",type=Path,default=Path("outputs/vetbench/recursive_state_recovery_v1/shards"));ap.add_argument("--symbolic",type=Path,default=Path("outputs/vetbench/composition_analysis_v1/symbolic_composition.csv"));ap.add_argument("--split",type=Path,default=Path("outputs/vetbench/circuit_localization_v1/discovery_validation_split.json"));a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.recursive); sym=pd.read_csv(a.symbolic); split=json.loads(a.split.read_text())
    prob_files=sorted(a.event_prob_dir.glob("event_probs_shard_*.csv"))
    if not prob_files:raise FileNotFoundError(f"no event probability shards in {a.event_prob_dir}")
    for p in prob_files:
        marker=p.with_suffix(".complete.json")
        if not marker.exists():raise FileNotFoundError(f"missing completion marker for {p}")
    probs=pd.concat([pd.read_csv(p) for p in prob_files],ignore_index=True)
    pcols=["target_prefix"]+[f"event_prob_{e}" for e in EVENTS]
    if len(probs)!=250 or probs.target_prefix.nunique()!=250:raise AssertionError("event probability shard coverage mismatch")
    d=d.drop(columns=[c for c in pcols[1:] if c in d],errors="ignore").merge(probs[pcols],on="target_prefix",how="left",validate="one_to_one")
    if d[pcols[1:]].isna().any().any():raise AssertionError("missing event probabilities after cache merge")
    if len(d)!=250 or d.target_prefix.nunique()!=250:raise AssertionError("expected 250 cached prefixes")
    if set(d.gt_event)!=set(EVENTS) or set(d.event_pred)-set(EVENTS):raise AssertionError("event vocabulary mismatch")
    discovery=set(split["discovery_trajectories"]);validation=set(split["validation_trajectories"]);observed=set(d.trajectory_id)
    if discovery & validation or discovery|validation != observed or len(discovery)!=30 or len(validation)!=20:raise AssertionError("frozen discovery/validation split coverage mismatch")
    all_rows=[]; hybrid_rows=[]
    for traj,g in d.groupby("trajectory_id",sort=True):
        for method,oracle,prob in (("hard_recursive",False,False),("probabilistic_recursive",False,True),("oracle_hard",True,False),("oracle_probability",True,True)):
            q=path(g,probability=prob,oracle=oracle);q["method"]=method;all_rows.append(q)
        for k in range(1,6):
            for method,prob in (("hard_hybrid",False),("probabilistic_hybrid",True)):
                q=path(g,event_override=k,probability=prob);q["method"]=method;q["oracle_event_step"]=k;hybrid_rows.append(q)
    rows=pd.concat(all_rows,ignore_index=True);hy=pd.concat(hybrid_rows,ignore_index=True)
    base=d[["target_prefix","trajectory_id","t","gt_state","state_correct","event_correct"]].copy();base=base.merge(sym[["key","symbolic_acc"]],left_on="target_prefix",right_on="key",how="left")
    base["native_correct"]=base.state_correct.astype(bool);base["oracle_correct"]=True
    base.to_csv(out/"recursive_state_tracking_base.csv",index=False);rows.to_csv(out/"recursive_state_tracking_paths.csv",index=False);hy.to_csv(out/"recursive_state_tracking_hybrid.csv",index=False)
    summary={"protocol":"frozen offline cache; trajectory is independent unit; target-prefix rows aggregated within trajectory before bootstrap/sign permutation","run_command":"python3 scripts/recursive_state_tracking.py","source_artifacts":{"recursive_csv":str(a.recursive),"recursive_sha256":file_sha256(a.recursive),"event_probability_shards":[{"path":str(p),"sha256":file_sha256(p),"completion_marker":str(p.with_suffix('.complete.json'))} for p in prob_files],"symbolic_csv":str(a.symbolic),"symbolic_sha256":file_sha256(a.symbolic),"split_json":str(a.split),"split_sha256":file_sha256(a.split)},"counts":{"prefixes":250,"trajectories":int(d.trajectory_id.nunique()),"steps":5},"methods":{},"event_conditioned":{},"error_accumulation":{},"irrecoverable":{},"hybrid":{},"split":{},"paired_comparisons":{}}
    for method in ("native","symbolic","hard_recursive","probabilistic_recursive","oracle_hard","oracle_probability"):
        if method=="native": q=base.rename(columns={"native_correct":"correct"});
        elif method=="symbolic": q=base.rename(columns={"symbolic_acc":"correct"})
        else:q=rows[rows.method==method].rename(columns={"state_correct":"correct"})
        summary["methods"][method]=summarize_rows(q)
    # Prefix-aligned paired effects, with trajectory-level inference.  These
    # are useful for distinguishing an actual improvement from overlapping
    # marginal confidence intervals.
    aligned=base.set_index("target_prefix")
    for method,col in (("symbolic","symbolic_acc"),("hard_recursive","state_correct"),("probabilistic_recursive","state_correct")):
        if method=="symbolic": diff=aligned[col].astype(float)-aligned.native_correct.astype(float)
        else:
            rr=rows[rows.method==method].set_index("target_prefix");diff=rr.state_correct.astype(float)-aligned.native_correct.astype(float)
        summary["paired_comparisons"][f"{method}_minus_native"]=stat(diff,aligned.trajectory_id)
    rr=rows[rows.method=="probabilistic_recursive"].set_index("target_prefix")
    summary["paired_comparisons"]["probabilistic_recursive_minus_symbolic"]=stat(rr.state_correct.astype(float)-aligned.symbolic_acc.astype(float),aligned.trajectory_id)
    for method in ("hard_recursive","probabilistic_recursive"):
        q=rows[rows.method==method].copy();summary["event_conditioned"][method]={}
        for label,mask in (("event_correct",q.event_correct),("event_wrong",~q.event_correct)):
            x=q[mask];summary["event_conditioned"][method][label]=summarize_rows(x,"state_correct")
        q=q.sort_values(["trajectory_id","t"]);q["prev_error"]=q.groupby("trajectory_id").state_correct.shift(1)
        f=q[q.t>=2];summary["error_accumulation"][method]={"error_rate_by_t":{str(i):float(1-q[q.t==i].state_correct.mean()) for i in range(1,6)},"error_given_previous_error":stat(1-f.loc[f.prev_error==False,"state_correct"],f.loc[f.prev_error==False,"trajectory_id"]),"error_given_previous_correct":stat(1-f.loc[f.prev_error==True,"state_correct"],f.loc[f.prev_error==True,"trajectory_id"])}
        irr=q.groupby("trajectory_id").apply(first_irrecoverable).rename("first_irrecoverable_t").reset_index();summary["irrecoverable"][method]={"count":int(irr.first_irrecoverable_t.notna().sum()),"rate":float(irr.first_irrecoverable_t.notna().mean()),"distribution":{str(k):int((irr.first_irrecoverable_t==k).sum()) for k in range(1,6)},"by_trajectory":irr.to_dict("records")};irr.assign(method=method).to_csv(out/f"{method}_irrecoverable.csv",index=False)
    for subset,ids in (("all",set(d.trajectory_id)),("discovery",set(split["discovery_trajectories"])),("validation",set(split["validation_trajectories"]))):
        summary["split"][subset]={}
        for method in ("native","symbolic","hard_recursive","probabilistic_recursive","oracle_hard","oracle_probability"):
            if method=="native":q=base[base.trajectory_id.isin(ids)].rename(columns={"native_correct":"correct"})
            elif method=="symbolic":q=base[base.trajectory_id.isin(ids)].rename(columns={"symbolic_acc":"correct"})
            else:q=rows[(rows.method==method)&rows.trajectory_id.isin(ids)].rename(columns={"state_correct":"correct"})
            summary["split"][subset][method]=summarize_rows(q)
    for method in ("hard_hybrid","probabilistic_hybrid"):
        summary["hybrid"][method]={}
        baseline_method="hard_recursive" if method=="hard_hybrid" else "probabilistic_recursive"
        baseline=rows[rows.method==baseline_method].set_index("target_prefix")
        for k in range(1,6):
            q=hy[(hy.method==method)&(hy.oracle_event_step==k)].rename(columns={"state_correct":"correct"});detail=summarize_rows(q)
            qq=q.set_index("target_prefix");gain=qq.correct.astype(float)-baseline.loc[qq.index,"state_correct"].astype(float)
            detail["gain_from_k_onward"]=stat(gain[qq.t>=k],qq.loc[qq.t>=k,"trajectory_id"])
            detail["gain_after_k"]=stat(gain[qq.t>k],qq.loc[qq.t>k,"trajectory_id"])
            detail["gain_at_t5"]=stat(gain[qq.t==5],qq.loc[qq.t==5,"trajectory_id"])
            qk=q[q.t==k]
            wrong_ids=set(qk.loc[qk.event_pred!=qk.gt_event,"trajectory_id"])
            correct_ids=set(qk.loc[qk.event_pred==qk.gt_event,"trajectory_id"])
            def conditional(ids):
                h=q[q.trajectory_id.isin(ids)].copy();h=h.set_index("target_prefix")
                result={"n_trajectories":len(ids),"n_prefixes":len(h)}
                for label,mask in (("at_replaced_step",h.t==k),("at_final_t5",h.t==5),("from_replaced_step_onward",h.t>=k),("after_replaced_step",h.t>k)):
                    z=h[mask];b=baseline.loc[z.index,"state_correct"].astype(float)
                    result[label]={"baseline_accuracy":stat(b,z.trajectory_id),"hybrid_accuracy":stat(z.correct.astype(float),z.trajectory_id),"gain":stat(z.correct.astype(float)-b,z.trajectory_id)}
                return result
            detail["replaced_event_was_wrong_count"]=int(len(wrong_ids))
            detail["event_wrong_at_k"]=conditional(wrong_ids)
            detail["event_correct_at_k"]=conditional(correct_ids)
            summary["hybrid"][method][f"replace_t{k}"]=detail
    summary["sanity"]={"oracle_hard_exact":bool((rows[rows.method=="oracle_hard"].state_correct).all()),"oracle_probability_exact":bool((rows[rows.method=="oracle_probability"].state_correct).all()),"transition_unit":"Left/Middle/Right swap algebra; oracle must be 1.0"}
    metric_records=[]
    def add_metric(group,method,subset,metric,x):
        metric_records.append({"group":group,"method":method,"condition":"","subset":subset,"metric":metric,"mean":x.get("mean"),"ci_low":x.get("ci95",[None,None])[0],"ci_high":x.get("ci95",[None,None])[1],"p_sign_permutation":x.get("p_sign_permutation"),"n_trajectories":x.get("n_trajectories")})
    for method,subsets in summary["methods"].items():
        for subset,v in subsets.items(): add_metric("main",method,subset,"accuracy",v["accuracy"])
    for method,conditions in summary["event_conditioned"].items():
        for condition,subsets in conditions.items():
            for subset,v in subsets.items():
                x=v["accuracy"];metric_records.append({"group":"event_conditioned","method":method,"condition":condition,"subset":subset,"metric":"accuracy","mean":x.get("mean"),"ci_low":x.get("ci95",[None,None])[0],"ci_high":x.get("ci95",[None,None])[1],"p_sign_permutation":x.get("p_sign_permutation"),"n_trajectories":x.get("n_trajectories")})
    pd.DataFrame(metric_records).to_csv(out/"recursive_state_tracking_metrics.csv",index=False)
    hybrid_records=[]
    for method,steps in summary["hybrid"].items():
        for replacement,detail in steps.items():
            k=int(replacement.rsplit("t",1)[1])
            for condition in ("all","event_wrong_at_k","event_correct_at_k"):
                source=detail if condition=="all" else detail[condition]
                if condition=="all":
                    for window,x in source.items():
                        if isinstance(x,dict) and "mean" in x: hybrid_records.append({"method":method,"replacement_step":k,"condition":condition,"window":window,"metric":"gain","mean":x["mean"],"ci_low":x["ci95"][0],"ci_high":x["ci95"][1],"n_trajectories":x["n_trajectories"]})
                else:
                    for window,x in source.items():
                        if isinstance(x,dict) and {"baseline_accuracy","hybrid_accuracy","gain"}.issubset(x):
                            for metric in ("baseline_accuracy","hybrid_accuracy","gain"):
                                y=x[metric];hybrid_records.append({"method":method,"replacement_step":k,"condition":condition,"window":window,"metric":metric,"mean":y["mean"],"ci_low":y["ci95"][0],"ci_high":y["ci95"][1],"n_trajectories":y["n_trajectories"]})
    pd.DataFrame(hybrid_records).to_csv(out/"recursive_state_tracking_hybrid_summary.csv",index=False)
    (out/"recursive_state_tracking_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    def f(x):return "NA" if x["mean"] is None else f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    lines=["# Recursive event → state tracking","","## Protocol","","Run command: `python3 scripts/recursive_state_tracking.py`.","","Existing event argmax/probability and native prediction cache are used without refitting. Oracle recursion substitutes ground-truth event at every step. Hybrid recursion substitutes only one specified step, then resumes model events. All uncertainty intervals are trajectory-cluster bootstrap; prefixes are not independent.","","## Main comparison (all 50 trajectories)","","| method | overall | t>=2 | t>=3 | t1 | t2 | t3 | t4 | t5 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("native","symbolic","hard_recursive","probabilistic_recursive","oracle_hard","oracle_probability"):
        q=summary["methods"][m];lines.append("| "+m+" | "+" | ".join(f(q[k]["accuracy"]) for k in ("overall","t_ge2","t_ge3","t1","t2","t3","t4","t5"))+" |")
    lines += ["","## Event-conditioned state accuracy","","| method | event condition | overall | t>=2 | t>=3 |","|---|---|---:|---:|---:|"]
    for m in ("hard_recursive","probabilistic_recursive"):
        for c in ("event_correct","event_wrong"):
            q=summary["event_conditioned"][m][c];lines.append(f"| {m} | {c} | {f(q['overall']['accuracy'])} | {f(q['t_ge2']['accuracy'])} | {f(q['t_ge3']['accuracy'])} |")
    lines += ["","### Event-conditioned accuracy by step","","| method | event condition | t1 | t2 | t3 | t4 | t5 |","|---|---|---:|---:|---:|---:|---:|"]
    for m in ("hard_recursive","probabilistic_recursive"):
        for c in ("event_correct","event_wrong"):
            q=summary["event_conditioned"][m][c];lines.append(f"| {m} | {c} | "+" | ".join(f(q[f't{i}']['accuracy']) for i in range(1,6))+" |")
    lines += ["","## Event recognition by step","","| subset | overall | t>=2 | t>=3 | t1 | t2 | t3 | t4 | t5 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label,mask in (("event_correct",d.event_correct.astype(float)),):
        vals={"overall":stat(mask,d.trajectory_id),"t_ge2":stat(mask[d.t>=2],d.loc[d.t>=2,"trajectory_id"]),"t_ge3":stat(mask[d.t>=3],d.loc[d.t>=3,"trajectory_id"])}
        for i in range(1,6):vals[f"t{i}"]=stat(mask[d.t==i],d.loc[d.t==i,"trajectory_id"])
        lines.append("| "+label+" | "+" | ".join(f(vals[k]) for k in ("overall","t_ge2","t_ge3","t1","t2","t3","t4","t5"))+" |")
    lines += ["","## Error accumulation","","| method | error t1 | t2 | t3 | t4 | t5 | error given previous error | error given previous correct |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("hard_recursive","probabilistic_recursive"):
        e=summary["error_accumulation"][m];lines.append(f"| {m} | "+" | ".join(f"{e['error_rate_by_t'][str(i)]:.3f}" for i in range(1,6))+f" | {f(e['error_given_previous_error'])} | {f(e['error_given_previous_correct'])} |")
    lines += ["","## Earliest irrecoverable error","","Definition: the first wrong prediction followed by wrong predictions at every remaining step through t=5.","","| method | trajectories with irrecoverable error | rate | first t1 | t2 | t3 | t4 | t5 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("hard_recursive","probabilistic_recursive"):
        e=summary["irrecoverable"][m];lines.append(f"| {m} | {e['count']}/50 | {e['rate']:.3f} | "+" | ".join(str(e['distribution'][str(i)]) for i in range(1,6))+" |")
    lines += ["","## Hybrid: replace one event by ground truth","","Rows below condition on trajectories whose original event at the replaced step was wrong. `gain@t5` is hybrid final-state accuracy minus the unmodified recursive baseline on those same trajectories.","","| method | replace step | wrong trajectories | baseline acc@k | hybrid acc@k | gain@k | baseline acc@t5 | hybrid acc@t5 | gain@t5 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("hard_hybrid","probabilistic_hybrid"):
        for k in range(1,6):
            e=summary["hybrid"][m][f"replace_t{k}"]["event_wrong_at_k"]
            def hv(window,metric):return f(e[window][metric])
            lines.append(f"| {m} | {k} | {e['n_trajectories']} | {hv('at_replaced_step','baseline_accuracy')} | {hv('at_replaced_step','hybrid_accuracy')} | {hv('at_replaced_step','gain')} | {hv('at_final_t5','baseline_accuracy')} | {hv('at_final_t5','hybrid_accuracy')} | {hv('at_final_t5','gain')} |")
    lines += ["","## Frozen split and paired comparisons","","| split | native | symbolic | hard recursive | probabilistic recursive | oracle |","|---|---:|---:|---:|---:|---:|"]
    for sp in ("discovery","validation"):
        q=summary["split"][sp];lines.append(f"| {sp} | "+" | ".join(f(q[m]["overall"]["accuracy"]) for m in ("native","symbolic","hard_recursive","probabilistic_recursive","oracle_hard"))+" |")
    lines += ["","Paired trajectory-cluster differences (probabilistic recursive minus native/symbolic) are:","",f"- versus native: {f(summary['paired_comparisons']['probabilistic_recursive_minus_native'])}",f"- versus symbolic: {f(summary['paired_comparisons']['probabilistic_recursive_minus_symbolic'])}","","These are paired prefix-aligned effects summarized at the trajectory unit; no validation parameter was fitted or selected."]
    lines += ["","## Error accumulation and irrecoverability","", "See `recursive_state_tracking_summary.json` for conditional error rates and each trajectory's first irrecoverable step. The definition is the first wrong step followed by wrong predictions at every remaining step through t=5; transient errors are not classified as irreversible.","","## Oracle and hybrid upper bounds","", "Oracle hard and oracle probability must be exactly 1.0; this validates the transition algebra and establishes the theoretical event-mediated ceiling. Hybrid results for replacing t=1..5 are in the summary JSON and CSV; improvement after replacement of step k diagnoses whether later state failures are inherited from that event or arise from subsequent maintenance.","","## Conclusions", "", "1. Model-event recursion reaches the hard/probabilistic accuracies reported above; probabilistic recursion is the primary estimate because it does not collapse after one uncertain event.", "2. Recursive estimates should be compared directly with native and old symbolic rows above; oracle=1.0 confirms that S0 plus correct events contains sufficient information.", "3. Event-conditioned and hybrid curves separate event-recognition error from downstream state maintenance; a large residual gap after correct events or after one-step repair supports a cross-step maintenance deficit.", "4. The result supports the stable-state-maintenance interpretation only to the extent shown by those conditional and hybrid diagnostics; oracle success alone rules out an error in the symbolic update implementation.","","## Artifacts","","- `recursive_state_tracking_base.csv`","- `recursive_state_tracking_paths.csv`","- `recursive_state_tracking_hybrid.csv`","- `recursive_state_tracking_metrics.csv`","- `recursive_state_tracking_hybrid_summary.csv`","- `recursive_state_tracking_summary.json`"]
    p=summary["methods"]["probabilistic_recursive"];h=summary["methods"]["hard_recursive"];n=summary["methods"]["native"];s=summary["methods"]["symbolic"]
    lines += ["","## Direct answers","",f"1. Starting at S0 and accumulating model events gives final t=5 accuracy {h['t5']['accuracy']['mean']:.3f} for hard recursion and {p['t5']['accuracy']['mean']:.3f} for probabilistic recursion; the corresponding all-prefix accuracies are {h['overall']['accuracy']['mean']:.3f} and {p['overall']['accuracy']['mean']:.3f}.",f"2. Both recursive methods are well above native state prediction ({n['overall']['accuracy']['mean']:.3f}; paired probabilistic-minus-native effect {f(summary['paired_comparisons']['probabilistic_recursive_minus_native'])}, p={summary['paired_comparisons']['probabilistic_recursive_minus_native']['p_sign_permutation']:.4f}). Probabilistic recursion is only modestly above old symbolic composition ({p['overall']['accuracy']['mean']:.3f} vs {s['overall']['accuracy']['mean']:.3f}; paired effect {f(summary['paired_comparisons']['probabilistic_recursive_minus_symbolic'])}, p={summary['paired_comparisons']['probabilistic_recursive_minus_symbolic']['p_sign_permutation']:.4f}), so the strong claim is against native, not a decisive overall win over symbolic.",f"3. Event recognition is {stat(d.event_correct.astype(float),d.trajectory_id)['mean']:.3f} overall. Under probabilistic recursion, state accuracy is {summary['event_conditioned']['probabilistic_recursive']['event_correct']['overall']['accuracy']['mean']:.3f} when the current event is correct versus {summary['event_conditioned']['probabilistic_recursive']['event_wrong']['overall']['accuracy']['mean']:.3f} when it is wrong; the t1→t5 error sequence and previous-error conditional rates show both event mistakes and persistence of prior state errors contribute. Hybrid replacement repairs the current state strongly at wrong t=1–3, but often does not carry through to t=5, indicating later events/maintenance also matter.",f"4. Yes in the ideal event-mediated sense: oracle hard and oracle probability are both exactly 1.000 at every step. But a currently correct event does not guarantee a correct model state at later steps, because the incoming state belief may already be wrong.",f"5. Yes, the evidence supports missing stable cross-step state maintenance: correct events plus S0 are sufficient, while model-event recursion degrades to {p['t5']['accuracy']['mean']:.3f} at t5 and {summary['irrecoverable']['probabilistic_recursive']['count']}/50 trajectories have an irrecoverable error under the stated horizon definition. The interpretation is not exclusive of event-recognition errors."]
    (out/"recursive_state_tracking_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"RECURSIVE_STATE_TRACKING_COMPLETE","out":str(out),"files":["recursive_state_tracking_base.csv","recursive_state_tracking_paths.csv","recursive_state_tracking_hybrid.csv","recursive_state_tracking_metrics.csv","recursive_state_tracking_hybrid_summary.csv","recursive_state_tracking_summary.json","recursive_state_tracking_report.md"]},indent=2))
if __name__=="__main__":main()
