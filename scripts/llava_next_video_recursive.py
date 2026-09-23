#!/usr/bin/env python3
"""Frozen hidden-event recursive tracking on LLaVA validation rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/vetbench/llava_next_video_7b_replication_v1"
STATES=("Left","Middle","Right"); EVENTS=("Left and Middle","Middle and Right","Left and Right"); SI={x:i for i,x in enumerate(STATES)}; EI={x:i for i,x in enumerate(EVENTS)}
def upd(s,e):
    a,b=e.split(" and "); return b if s==a else a if s==b else s
def trans(sp,ep):
    z=np.zeros(3)
    for i,s in enumerate(STATES):
        for j,e in enumerate(EVENTS): z[SI[upd(s,e)]]+=sp[i]*ep[j]
    return z/z.sum()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--split",type=Path,default=ROOT/"outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"); a=ap.parse_args()
    out=a.out; b=pd.read_csv(out/"behavior.csv"); split=json.loads(a.split.read_text()); val=set(split["validation_trajectories"]); b=b[b.trajectory_id.isin(val)].sort_values(["trajectory_id","t"])
    dec=joblib.load(out/"frozen_hidden_event_decoder.joblib");
    # Materialize arrays before leaving the NpzFile context; retaining the
    # closed lazy archive causes ``NoneType.open`` on the first lookup.
    with np.load(out/"hidden_states.npz") as z: h={k: z[k] for k in z.files}
    x=dec["scaler"].transform(np.stack([h[k][dec["layer"]] for k in b.target_prefix])); raw=dec["clf"].predict_proba(x); p=np.zeros((len(b),3))
    for j,c in enumerate(dec["clf"].classes_): p[:,EI[c]]=raw[:,j]
    pos={k:i for i,k in enumerate(b.target_prefix)}; rows=[]
    for tid,g in b.groupby("trajectory_id",sort=True):
        initial=g.iloc[0].initial_state; hard={m:initial for m in ("explicit","hidden","oracle")}; belief={m:np.eye(3)[SI[initial]] for m in ("explicit","hidden","oracle")}; hist=[]
        for _,r in g.iterrows():
            ep=np.array([json.loads(r.event_scores_json)[f"{e}__prob"] for e in EVENTS]); ep/=ep.sum(); hp=p[pos[r.target_prefix]]; he=EVENTS[int(hp.argmax())]; hist.append(he==r.gt_event)
            for m,e,prob in (("explicit",r.event_pred,ep),("hidden",he,hp),("oracle",r.gt_event,np.eye(3)[EI[r.gt_event]])):
                hard[m]=upd(hard[m],e); belief[m]=trans(belief[m],prob)
            q=dict(r); q.update(native_state_correct=r.state_pred==r.gt_state,hidden_event_correct=he==r.gt_event,hidden_all_events_correct_so_far=all(hist),hidden_event_state_pred=hard["hidden"],hidden_event_state_correct=hard["hidden"]==r.gt_state,explicit_event_state_pred=hard["explicit"],explicit_event_state_correct=hard["explicit"]==r.gt_state,oracle_event_state_pred=hard["oracle"],oracle_event_state_correct=hard["oracle"]==r.gt_state,hidden_prob_state_pred=STATES[int(belief["hidden"].argmax())],explicit_prob_state_pred=STATES[int(belief["explicit"].argmax())])
            q["hidden_prob_state_correct"]=q["hidden_prob_state_pred"]==r.gt_state; q["explicit_prob_state_correct"]=q["explicit_prob_state_pred"]==r.gt_state; rows.append(q)
    f=pd.DataFrame(rows); methods={"native_state":"native_state_correct","explicit_event_hard":"explicit_event_state_correct","explicit_event_prob":"explicit_prob_state_correct","hidden_event_hard":"hidden_event_state_correct","hidden_event_prob":"hidden_prob_state_correct","oracle_event_hard":"oracle_event_state_correct"}; metrics=[]
    for m,c in methods.items():
        for name,mask in [("overall",np.ones(len(f),bool)),("t_ge2",f.t>=2),("t_ge3",f.t>=3)]+[(f"t{t}",f.t==t) for t in range(1,6)]: metrics.append({"method":m,"subset":name,"accuracy":float(f.loc[mask,c].mean())})
    f.to_csv(out/"recursive_validation_prefix.csv",index=False); pd.DataFrame(metrics).to_csv(out/"recursive_metrics.csv",index=False); f[f.hidden_all_events_correct_so_far].to_csv(out/"recursive_hidden_events_all_correct.csv",index=False)
    (out/"recursive_summary.json").write_text(json.dumps({"validation_trajectories":sorted(val),"metrics":metrics,"hidden_probe_layer":dec["layer"],"paired_improvement_t_ge2_hidden_minus_native":float(f.loc[f.t>=2,"hidden_event_state_correct"].mean()-f.loc[f.t>=2,"native_state_correct"].mean())},indent=2)+"\n"); print("RECURSIVE_PASS")
if __name__=="__main__": main()
