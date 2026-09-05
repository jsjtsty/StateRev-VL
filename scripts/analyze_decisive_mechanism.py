#!/usr/bin/env python3
"""Offline analysis for decisive_mechanism_experiment.py.

The event efficacy gate is evaluated before state movement is interpreted.
Probe fitting is leave-one-trajectory-out, with scaler/PCA fit on training
trajectories only. Pair bootstrap and a paired sign permutation are reported.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vg_same_input_probe import _scaler_fit

EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
STATES = ("Left", "Middle", "Right")

def fit_predict(Xtr, ytr, Xte, classes):
    from sklearn.linear_model import LogisticRegression
    sc = _scaler_fit(Xtr); clf = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(Xtr), ytr)
    P = np.zeros((len(Xte), len(classes)))
    for j, c in enumerate(clf.classes_): P[:, int(c)] = clf.predict_proba(sc.transform(Xte))[:, j]
    return P

def boot(values, groups, seed=7, n=2000):
    rng=np.random.default_rng(seed); gs=sorted(set(groups)); a=[]
    for _ in range(n):
        take=rng.choice(gs,len(gs),replace=True); a.append(np.mean([values[i] for g in take for i,x in enumerate(groups) if x==g]))
    return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("outputs/vetbench/mechanism_gate_final")); ap.add_argument("--manifest",type=Path,default=None); args=ap.parse_args()
    out=args.out; man=json.loads((args.manifest or out/"pair_manifest.json").read_text()); pairs=man["pairs"]
    rows=list(csv.DictReader((Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv")).open()))
    keys=[f"{r['trajectory_id']}_t{r['t']}" for r in rows]; traj=[r["trajectory_id"] for r in rows]
    z={c:np.load(out/f"hidden_{c}.npz") for c in ("baseline","no_current_event")}
    event={e:i for i,e in enumerate(EVENTS)}; state={s:i for i,s in enumerate(STATES)}
    def hs(archive, ks):
        # Efficacy is a gate, so use the preregistered final decoder layer
        # rather than selecting a layer after seeing the intervention result.
        return np.stack([archive[k] for k in ks])[:, -1, :]
    # no-event: paired per-row GT event probability under a fixed LOO probe
    noevent_delta=[]; noevent_prev_delta=[]; noevent_tr=[]
    for i,r in enumerate(rows):
        train=[j for j,g in enumerate(traj) if g!=traj[i]]; y=np.array([event[x["gt_event"]] for x in rows]);
        P0=fit_predict(hs(z["baseline"],[keys[j] for j in train]),y[train],z["baseline"][keys[i]][None,-1,:],EVENTS)[0]
        P1=fit_predict(hs(z["baseline"],[keys[j] for j in train]),y[train],z["no_current_event"][keys[i]][None,-1,:],EVENTS)[0]
        noevent_delta.append(P1[y[i]]-P0[y[i]]); noevent_tr.append(traj[i])
        yp=np.array([state[x["gt_prev_state"]] for x in rows])
        Q0=fit_predict(hs(z["baseline"],[keys[j] for j in train]),yp[train],z["baseline"][keys[i]][None,-1,:],STATES)[0]
        Q1=fit_predict(hs(z["baseline"],[keys[j] for j in train]),yp[train],z["no_current_event"][keys[i]][None,-1,:],STATES)[0]
        noevent_prev_delta.append(Q1[yp[i]]-Q0[yp[i]])
    # Pair efficacy and state movement, only interpret state if source-event
    # probability/margin increases in the intended direction.
    pair_rows=[]
    pz={c:np.load(out/f"hidden_{c}.npz") for c in ("source_current_transplant","target_self","same_event_source","matched_history_transplant","source_window_shuffle")}
    beh=list(csv.DictReader((out/"behavior.csv").open())); bmap={(r["key"],r["condition"]):r for r in beh}
    for p in pairs:
        key=p["pair_id"]; target=p["target_traj"]; i=keys.index(f"{target}_t{p['t']}"); train=[j for j,g in enumerate(traj) if g!=target]; ye=np.array([event[x["gt_event"]] for x in rows]); ys=np.array([state[x["gt_state"]] for x in rows])
        train_keys=[keys[j] for j in train]
        base_e=fit_predict(hs(z["baseline"],train_keys),ye[train],z["baseline"][f"{target}_t{p['t']}"][None,-1,:],EVENTS)[0]
        hy_e=fit_predict(hs(z["baseline"],train_keys),ye[train],pz["source_current_transplant"][key][None,-1,:],EVENTS)[0]
        base_s=fit_predict(hs(z["baseline"],train_keys),ys[train],z["baseline"][f"{target}_t{p['t']}"][None,-1,:],STATES)[0]
        hy_s=fit_predict(hs(z["baseline"],train_keys),ys[train],pz["source_current_transplant"][key][None,-1,:],STATES)[0]
        st,cf=p["target_state"],p["counterfactual_state"]; se,te=p["source_event"],p["target_event"]
        hyb=bmap[(key,"source_current_transplant")]; clean=bmap[(key,"target_self")]
        native_contrast=(float(hyb[f"logprob_{cf}"])-float(hyb[f"logprob_{st}"]))-(float(clean[f"logprob_{cf}"])-float(clean[f"logprob_{st}"]))
        base_margin=float(base_e[event[se]]-base_e[event[te]])
        event_shifts={}
        for cond in ("target_self","same_event_source","matched_history_transplant","source_window_shuffle","source_current_transplant"):
            pe=fit_predict(hs(z["baseline"],train_keys),ye[train],pz[cond][key][None,-1,:],EVENTS)[0]
            event_shifts[cond]=float((pe[event[se]]-pe[event[te]])-base_margin)
        pair_rows.append({"pair_id":key,"target_traj":target,"target_prefix":f"{target}_t{p['t']}","event_p_source_delta":float(hy_e[event[se]]-base_e[event[se]]),"event_margin_source_minus_target":float(event_shifts["source_current_transplant"]),"event_shift":float(event_shifts["source_current_transplant"]),"event_shift_controls":event_shifts,"state_p_cf_delta":float(hy_s[state[cf]]-base_s[state[cf]]),"state_p_target_delta":float(hy_s[state[st]]-base_s[state[st]]),"event_success":bool(hy_e[event[se]]>hy_e[event[te]] and hy_e[event[se]]>base_e[event[se]]),"native_logit_cf_minus_target_delta":native_contrast,"native_cf_logprob_delta":float(hyb[f"logprob_{cf}"])-float(clean[f"logprob_{cf}"])})

    # A target may have many eligible sources.  Collapse those first, then
    # perform inference on target-prefix effects and trajectory clusters.
    by_target=defaultdict(list)
    for x in pair_rows: by_target[x["target_prefix"]].append(x)
    target_rows=[]
    for target_prefix,xs in by_target.items():
        target_rows.append({"target_prefix":target_prefix,"target_traj":xs[0]["target_traj"],
            "event_shift":float(np.mean([x["event_shift"] for x in xs])),
            "event_success_rate":float(np.mean([x["event_success"] for x in xs])),
            "state_p_cf_delta":float(np.mean([x["state_p_cf_delta"] for x in xs])),
            "state_p_target_delta":float(np.mean([x["state_p_target_delta"] for x in xs])),
            "event_shift_controls":{c:float(np.mean([x["event_shift_controls"][c] for x in xs])) for c in ("target_self","same_event_source","matched_history_transplant","source_window_shuffle")},
            "n_source_pairs":len(xs)})
    def clustered_stat(field):
        tr=defaultdict(list)
        for x in target_rows: tr[x["target_traj"]].append(x[field])
        means=np.array([np.mean(v) for v in tr.values()])
        return float(means.mean()),boot([float(v) for v in means],list(tr),n=2000),means
    event_effect,event_ci,event_cluster=clustered_stat("event_shift")
    state_effect,state_ci,state_cluster=clustered_stat("state_p_cf_delta")
    def paired_control(control):
        vals=[x["event_shift"]-x["event_shift_controls"][control] for x in target_rows]
        tr=defaultdict(list)
        for x,v in zip(target_rows,vals): tr[x["target_traj"]].append(v)
        means=np.array([np.mean(v) for v in tr.values()]); groups=list(tr)
        rng=np.random.default_rng(7); null=[float(np.mean(means*rng.choice([-1,1],len(means)))) for _ in range(5000)]
        return {"mean":float(means.mean()),"ci95":boot(means.tolist(),groups,n=2000),"sign_p":float(np.mean(np.abs(null)>=abs(means.mean())))}
    event_control_comparisons={c:paired_control(c) for c in ("target_self","same_event_source","matched_history_transplant","source_window_shuffle")}
    efficacy=float(np.mean([x["event_success_rate"] for x in target_rows])) if target_rows else 0
    efficacy_pair_descriptive=float(np.mean([x["event_success"] for x in pair_rows])) if pair_rows else 0
    rng=np.random.default_rng(7); null=[float(np.mean(state_cluster*rng.choice([-1,1],len(state_cluster)))) for _ in range(5000)]
    state_p=float(np.mean(np.abs(null)>=abs(state_effect))) if len(state_cluster) else float("nan")
    control_effects={}
    for cond in ("same_event_source","matched_history_transplant","source_window_shuffle"):
        ds=[]
        for p in pairs:
            key=p["pair_id"]; cf=p["counterfactual_state"]; st=p["target_state"]
            a=bmap[(key,cond)]; clean=bmap[(key,"target_self")]
            ds.append((float(a[f"logprob_{cf}"])-float(a[f"logprob_{st}"]))-(float(clean[f"logprob_{cf}"])-float(clean[f"logprob_{st}"])))
        control_effects[cond]={"native_logit_cf_minus_target_delta_mean":float(np.mean(ds)),"n":len(ds)}
    if efficacy < .7: verdict="event_invariant_intervention_invalid"
    elif not (state_ci[0] > 0 and state_p < .05): verdict="event_changed_state_not_to_counterfactual"
    else: verdict="event_and_state_counterfactual_shift"
    result={"verdict":verdict,"n_pair_rows_descriptive":len(pair_rows),"n_target_prefixes":len(target_rows),"n_target_trajectories":len(set(x["target_traj"] for x in target_rows)),"event_efficacy_rate_target_cluster":efficacy,"event_efficacy_rate_pair_descriptive":efficacy_pair_descriptive,"event_shift_mean_target_cluster":event_effect,"event_shift_ci95_trajectory_cluster":event_ci,"event_shift_paired_controls":event_control_comparisons,"pair_state_cf_delta_mean_target_cluster":state_effect,"pair_state_cf_delta_ci95_trajectory_cluster":state_ci,"pair_state_cf_sign_p_trajectory_cluster":state_p,"no_current_event_event_gt_prob_delta_mean":float(np.mean(noevent_delta)),"no_current_event_event_gt_prob_delta_ci95":boot(noevent_delta,noevent_tr),"no_current_event_prev_state_gt_prob_delta_mean":float(np.mean(noevent_prev_delta)),"no_current_event_prev_state_gt_prob_delta_ci95":boot(noevent_prev_delta,noevent_tr),"control_effects":control_effects,"target_prefix_effects":target_rows,"pairs":pair_rows,"interpretation":"Pair rows are descriptive only. All primary pair inference first averages sources within target prefix, then clusters bootstrap/permutation by trajectory. State movement is interpreted only when event efficacy passes."}
    (out/"decisive_analysis.json").write_text(json.dumps(result,indent=1)); print(json.dumps({k:v for k,v in result.items() if k!="pairs"},indent=2))
if __name__=="__main__": main()
