#!/usr/bin/env python3
"""Recursive event-to-state recovery plus discovery/validation gating.

CPU modes consume existing artifacts. event-probs is the only model mode and
stores a complete three-class event distribution per prefix in shard CSVs.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

STATES=("Left","Middle","Right")
EVENTS=("Left and Middle","Middle and Right","Left and Right")
EVENT_PAIRS={"Left and Middle":("Left","Middle"),"Middle and Right":("Middle","Right"),"Left and Right":("Left","Right")}
SEED=20260916

def swap(s,e):
    a,b=EVENT_PAIRS[e]
    return b if s==a else a if s==b else s

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def clustered(x, group, seed=SEED, n=4000):
    d=pd.DataFrame({"v":np.asarray(x,float),"g":np.asarray(group)})
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    if d.empty:return {"mean":None,"ci95":[None,None],"p_sign":None,"n_trajectories":0}
    z=d.groupby("g").v.mean().to_numpy(); rng=np.random.default_rng(seed)
    boot=np.array([z[rng.integers(0,len(z),len(z))].mean() for _ in range(n)])
    null=np.array([np.mean(z*rng.choice([-1,1],len(z))) for _ in range(n)])
    return {"mean":float(z.mean()),"ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "p_sign":float(np.mean(np.abs(null)>=abs(z.mean()))),"n_trajectories":int(len(z))}

def prepare(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    b=pd.read_csv(a.behavior); assert len(b)==250
    b["target_prefix"]=b.trajectory_id+"_t"+b.t.astype(str)
    split=json.loads(Path(a.split).read_text())
    if split.get("fingerprint_source"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from content_disjoint_split import assert_content_disjoint
        assert_content_disjoint(split, Path(split["fingerprint_source"]))
    n=pd.read_csv(a.native_behavior); n=n[n.condition=="baseline"].copy()
    n=n.rename(columns={"key":"target_prefix","state_pred":"native_state_pred"})
    n=n[["target_prefix","logprob_Left","logprob_Middle","logprob_Right","native_state_pred"]]
    assert len(n)==250
    rows=b.merge(n,on="target_prefix")
    m={"version":1,"seed":SEED,"behavior":str(a.behavior),"native_behavior":str(a.native_behavior),
       "split":str(a.split),"split_sha256":sha256(Path(a.split)),
       "discovery_trajectories":split["discovery_trajectories"],
       "validation_trajectories":split["validation_trajectories"],
       "event_classes":list(EVENTS),"states":list(STATES),
       "event_probs_source":"three event class teacher-forced log-probs; never inferred from GT/max-other summaries",
       "rows":rows.to_dict("records")}
    (out/"recursive_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps({"status":"RECURSIVE_MANIFEST_READY","rows":len(rows),
                      "discovery":len(split["discovery_trajectories"]),
                      "validation":len(split["validation_trajectories"])},indent=2))

def event_probs(a):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from run_state_rev_audit import event_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import EVENT_TOKEN_IDS,load_model_and_processor,render_inputs,teacher_force_logprob
    m=json.loads((Path(a.out)/"recursive_manifest.json").read_text())
    rows=m["rows"][a.shard_index::a.num_shards]
    model,proc=load_model_and_processor(Path(a.model_dir)); model.eval()
    from tqdm import tqdm
    rec=[]
    for r in tqdm(rows,desc=f"event-probs shard {a.shard_index}",unit="prefix"):
        clip=sample_clip(Path(a.dataset)/f"{r['trajectory_id']}.mp4",0,int(r["frame_end"]))
        inp,fp=render_inputs(proc,event_messages(clip,int(r["t"])),clip,"controlled_8fps")
        lp={e:float(teacher_force_logprob(model,inp,EVENT_TOKEN_IDS[e])) for e in EVENTS}
        z=np.array([lp[e] for e in EVENTS]); q=np.exp(z-z.max()); q=q/q.sum()
        rec.append({"target_prefix":r["target_prefix"],"trajectory_id":r["trajectory_id"],"t":r["t"],
                    **{f"event_logprob_{e}":lp[e] for e in EVENTS},
                    **{f"event_prob_{e}":float(q[i]) for i,e in enumerate(EVENTS)},
                    "input_ids_sha256":fp["input_ids_sha256"],
                    "pixel_sha256":fp["pixel_values_videos_sha256"],
                    "video_grid_thw":json.dumps(fp["video_grid_thw"])})
    p=Path(a.out)/"shards"; p.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rec).to_csv(p/f"event_probs_shard_{a.shard_index}.csv",index=False)
    (p/f"event_probs_shard_{a.shard_index}.complete.json").write_text(
        json.dumps({"rows":len(rec),"shard":a.shard_index},indent=2)+"\n")

def load_probs(a):
    p=Path(a.out)/"shards"
    for i in range(a.num_shards):
        assert (p/f"event_probs_shard_{i}.complete.json").exists(),f"missing shard {i}"
    fs=sorted(p.glob("event_probs_shard_*.csv"))
    d=pd.concat([pd.read_csv(x) for x in fs],ignore_index=True)
    assert len(d)==250 and d.target_prefix.nunique()==250
    return d

def recurse(rows, probs=None):
    pm={} if probs is None else probs.set_index("target_prefix").to_dict("index")
    out=[]
    for _,g in rows.sort_values(["trajectory_id","t"]).groupby("trajectory_id"):
        hard=g.iloc[0]["initial_state"]; belief=np.zeros(3); belief[STATES.index(hard)]=1.
        for _,r in g.iterrows():
            e=r["event_pred"]
            hard=swap(hard,e) if e in EVENTS else hard
            if probs is not None:
                ep=np.array([pm[r.target_prefix][f"event_prob_{x}"] for x in EVENTS])
                nxt=np.zeros(3)
                for i,s in enumerate(STATES):
                    for j,x in enumerate(EVENTS): nxt[STATES.index(swap(s,x))]+=belief[i]*ep[j]
                belief=nxt/nxt.sum()
            else:
                belief[:]=0; belief[STATES.index(hard)]=1.
            pred=STATES[int(np.argmax(belief))]
            out.append({**r.to_dict(),"recursive_hard_pred":hard,
                        "recursive_prob_pred":pred,"recursive_prob_conf":float(belief.max()),
                        "recursive_prob_Left":float(belief[0]),
                        "recursive_prob_Middle":float(belief[1]),
                        "recursive_prob_Right":float(belief[2]),
                        "hard_correct":hard==r.gt_state,"prob_correct":pred==r.gt_state})
    return pd.DataFrame(out)

def analyze(a):
    root=Path(a.out); m=json.loads((root/"recursive_manifest.json").read_text())
    b=pd.DataFrame(m["rows"]); p=load_probs(a) if a.event_probs else None
    d=recurse(b,p); d.to_csv(root/"recursive_state_results.csv",index=False)
    s={"protocol":"recursive hard/probabilistic event update; target trajectory cluster",
       "event_probs_available":p is not None,"comparisons":{}}
    for name,col in [("native","state_correct"),("hard","hard_correct"),("probabilistic","prob_correct")]:
        s["comparisons"][name]={}
        for tag,mask in [("overall",np.ones(len(d),bool)),("t_ge2",d.t>=2),("t_ge3",d.t>=3)]+[(f"t{t}",d.t==t) for t in range(1,6)]:
            x=d[mask]; s["comparisons"][name][tag]=clustered(x[col].astype(float),x.trajectory_id)
        s["comparisons"][name]["per_step_accuracy"]={str(t):float(d[d.t==t][col].mean()) for t in range(1,6)}
    sym=Path(a.symbolic)
    if sym.exists(): s["symbolic_accuracy_mean"]=float(pd.read_csv(sym).symbolic_acc.mean())
    (root/"recursive_state_summary.json").write_text(json.dumps(s,indent=2)+"\n")
    lines=["# Recursive state recovery","",
           "Hard recursion uses the aligned model event argmax at each step. "
           "Probabilistic recursion uses the three-class teacher-forced event "
           "distribution and propagates a Left/Middle/Right belief. Pair rows "
           "are not treated as independent; intervals cluster trajectories.","",
           "| method | overall | t>=2 | t>=3 |","|---|---:|---:|---:|"]
    for name in ("native","hard","probabilistic"):
        q=s["comparisons"][name]
        def f(x):
            return "NA" if x["mean"] is None else f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
        lines.append(f"| {name} | {f(q['overall'])} | {f(q['t_ge2'])} | {f(q['t_ge3'])} |")
    lines += ["","Per-step results and accumulated-error diagnostics are in "
              "recursive_state_results.csv and recursive_state_summary.json."]
    (root/"recursive_state_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"RECURSIVE_ANALYSIS_COMPLETE","rows":len(d),"event_probs":p is not None},indent=2))

def fit_gate(a):
    root=Path(a.out); d=pd.read_csv(root/"recursive_state_results.csv")
    split=json.loads(Path(a.split).read_text()); x=d[d.trajectory_id.isin(split["discovery_trajectories"])].copy()
    best=None
    for c in [.5,.6,.7,.8,.9]:
      for nm in [0,.25,.5,1,1.5,2]:
       for lam in [.25,.5,1,2]:
        margin=[]
        for _,r in x.iterrows():
            z=[r[f"logprob_{s}"] for s in STATES]; k=STATES.index(r.gt_state)
            margin.append(z[k]-max(z[:k]+z[k+1:]))
        gate=(x.recursive_prob_conf>=c)&(np.asarray(margin)<nm)&(x.recursive_prob_pred!=x.native_state_pred)
        pred=[]
        for use,(_,r) in zip(gate,x.iterrows()):
            z={s:float(r[f"logprob_{s}"]) for s in STATES}
            if use:z[x.loc[r.name,"recursive_prob_pred"]]+=lam
            pred.append(max(STATES,key=lambda s:z[s]))
        pred=np.asarray(pred,dtype=object)
        score=float(pd.DataFrame({"v":(pred==x.gt_state).astype(float),
                                  "g":x.trajectory_id.to_numpy()}).groupby("g").v.mean().mean())
        if best is None or score>best["score"]:
            best={"confidence_threshold":c,"native_margin_threshold":nm,"lambda":lam,"score":score}
    (root/"gate_config.json").write_text(json.dumps({"selection":"discovery only","config":best},indent=2)+"\n")
    print(json.dumps({"status":"GATE_FIT_COMPLETE","config":best},indent=2))

def unit(_a=None):
    assert swap("Left","Left and Middle")=="Middle"
    assert swap("Right","Left and Middle")=="Right"
    b=np.array([1.,0.,0.]); e=np.array([1.,0.,0.])
    n=np.zeros(3)
    for i,s in enumerate(STATES):
        for j,x in enumerate(EVENTS): n[STATES.index(swap(s,x))]+=b[i]*e[j]
    assert np.allclose(n,[0.,1.,0.])
    print("RECURSIVE_UNIT_PASS: deterministic and probabilistic transition algebra")

def evaluate_gate(a):
    root=Path(a.out); d=pd.read_csv(root/"recursive_state_results.csv")
    cfg=json.loads((root/"gate_config.json").read_text())["config"]
    split=json.loads(Path(a.split).read_text())
    # This is the frozen validation report. Discovery rows are never included.
    d=d[d.trajectory_id.isin(split["validation_trajectories"])].copy()
    def apply(x, mode):
        out=[]
        for _,r in x.iterrows():
            native=r.native_state_pred
            if mode=="no_gate": pred=native; trig=False
            elif mode=="always_correct": pred=r.gt_state; trig=True
            else:
                z={s:float(r[f"logprob_{s}"]) for s in STATES}
                margin=z[r.gt_state]-max(v for s,v in z.items() if s!=r.gt_state)
                trig=(r.recursive_prob_conf>=cfg["confidence_threshold"] and
                      margin<cfg["native_margin_threshold"] and
                      r.recursive_prob_pred!=native)
                if trig:z[r.recursive_prob_pred]+=cfg["lambda"]
                pred=max(STATES,key=z.get)
            out.append({**r.to_dict(),"mode":mode,"gate_trigger":trig,
                        "correct":pred==r.gt_state,"wrong_to_correct":pred!=native and pred==r.gt_state,
                        "correct_to_wrong":native==r.gt_state and pred!=r.gt_state,
                        "output_pred":pred})
        return pd.DataFrame(out)
    allx=pd.concat([apply(d,m) for m in ("no_gate","confidence_gate","always_correct")],ignore_index=True)
    allx.to_csv(root/"gate_results.csv",index=False)
    sm={"protocol":"gate fit on discovery only; validation frozen; trajectory-cluster bootstrap",
        "config":cfg,"modes":{}}
    for mode in ("no_gate","confidence_gate","always_correct"):
      q=allx[allx["mode"]==mode]; sm["modes"][mode]={}
      for tag,mask in [("overall",np.ones(len(q),bool)),("t_ge2",q["t"]>=2),("t_ge3",q["t"]>=3)]+[(f"t{t}",q["t"]==t) for t in range(1,6)]:
        z=q[mask]; sm["modes"][mode][tag]={"accuracy":clustered(z["correct"],z["trajectory_id"]),
          "wrong_to_correct":clustered(z["wrong_to_correct"],z["trajectory_id"]),
          "correct_to_wrong":clustered(z["correct_to_wrong"],z["trajectory_id"]),
          "gate_rate":float(z["gate_trigger"].mean())}
    (root/"gate_summary.json").write_text(json.dumps(sm,indent=2)+"\n")
    lines=["# Confidence-gated state correction","",
           "Thresholds were selected on discovery trajectories only and then "
           "frozen. Validation rows are reported here; always-correct is an "
           "oracle upper-bound control, not a deployable method.","",
           "| mode | subset | accuracy | wrong->correct | correct->wrong |",
           "|---|---|---:|---:|---:|"]
    for mode in ("no_gate","confidence_gate","always_correct"):
      for tag in ("overall","t_ge2","t_ge3"):
        q=sm["modes"][mode][tag]
        def f(x):
            return "NA" if x["mean"] is None else f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
        lines.append(f"| {mode} | {tag} | {f(q['accuracy'])} | {f(q['wrong_to_correct'])} | {f(q['correct_to_wrong'])} |")
    (root/"gate_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"GATE_VALIDATION_COMPLETE","rows":len(allx),"config":cfg},indent=2))

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    def common(p): p.add_argument("--out",type=Path,required=True)
    p=sp.add_parser("prepare"); common(p); p.add_argument("--behavior",type=Path,required=True); p.add_argument("--native-behavior",type=Path,required=True); p.add_argument("--split",type=Path,required=True); p.set_defaults(f=prepare)
    p=sp.add_parser("event-probs"); common(p); p.add_argument("--model-dir",required=True); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--shard-index",type=int,required=True); p.add_argument("--num-shards",type=int,required=True); p.set_defaults(f=event_probs)
    p=sp.add_parser("analyze"); common(p); p.add_argument("--event-probs",action="store_true"); p.add_argument("--num-shards",type=int,default=1); p.add_argument("--symbolic",type=Path,default=Path("outputs/vetbench/composition_analysis_v1/symbolic_composition.csv")); p.set_defaults(f=analyze)
    p=sp.add_parser("fit-gate"); common(p); p.add_argument("--split",type=Path,required=True); p.set_defaults(f=fit_gate)
    p=sp.add_parser("evaluate-gate"); common(p); p.add_argument("--split",type=Path,required=True); p.set_defaults(f=evaluate_gate)
    p=sp.add_parser("unit"); p.set_defaults(f=unit)
    a=ap.parse_args(); a.f(a)
if __name__=="__main__": main()
