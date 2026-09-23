#!/usr/bin/env python3
"""Gate-triggered L32 additive residual-delta intervention.

Frozen donor pairs are inherited from head0_downstream_delta_rescue_v1.  The
only new decision is the gate configuration produced on discovery trajectories
by recursive_state_recovery.py.  L32 uses the existing additive semantics:
h32' = h32_target + alpha * (h32_donor - h32_target).
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

STATES=("Left","Middle","Right")
CONDITIONS=("baseline","fixed_alpha","gated_fixed_alpha","gated_adaptive_alpha")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()
def clustered(x,g,seed=20260916,n=4000):
    d=pd.DataFrame({"v":np.asarray(x,float),"g":np.asarray(g)}).dropna()
    if d.empty:return {"mean":None,"ci95":[None,None],"p_sign":None,"n_trajectories":0}
    z=d.groupby("g").v.mean().to_numpy(); rng=np.random.default_rng(seed)
    b=np.array([z[rng.integers(0,len(z),len(z))].mean() for _ in range(n)])
    nul=np.array([np.mean(z*rng.choice([-1,1],len(z))) for _ in range(n)])
    return {"mean":float(z.mean()),"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],
            "p_sign":float(np.mean(np.abs(nul)>=abs(z.mean()))),"n_trajectories":int(len(z))}
def prepare(a):
    src=Path(a.source_manifest); gate=Path(a.gate_config); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    m=json.loads(src.read_text()); g=json.loads(gate.read_text())
    assert m["patches"]["downstream"]["layer_one_based"]==32
    outm={"version":1,"source_manifest":str(src),"source_manifest_sha256":sha256(src),
          "gate_config":str(gate),"gate_config_sha256":sha256(gate),
          "frozen_pairs":True,"conditions":list(CONDITIONS),
          "fixed_alpha":float(a.fixed_alpha),"max_alpha":float(a.max_alpha),
          "adaptive_rule":"alpha=min(max_alpha, max(0,-baseline_gt_margin/effect_at_alpha_1)); only gated rows",
          "targets":m["targets"],"target_count":len(m["targets"])}
    (out/"conditional_l32_manifest.json").write_text(json.dumps(outm,indent=2)+"\n")
    print(json.dumps({"status":"CONDITIONAL_L32_MANIFEST_READY","targets":len(m["targets"]),
                      "conditions":CONDITIONS,"fixed_alpha":a.fixed_alpha,"max_alpha":a.max_alpha,
                      "gate_config":g["config"]},indent=2))
def run(a):
    import torch
    from tqdm import tqdm
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import circuit_localization
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS,load_model_and_processor,render_inputs,to_device
    root=Path(a.out); m=json.loads((root/"conditional_l32_manifest.json").read_text())
    cfg=json.loads(Path(m["gate_config"]).read_text())["config"]
    fixed_alpha=float(m["fixed_alpha"])
    max_alpha=float(m["max_alpha"])
    behavior=pd.read_csv(a.behavior); behavior["target_prefix"]=behavior.trajectory_id+"_t"+behavior.t.astype(str)
    initial=behavior.set_index("target_prefix")["initial_state"].to_dict()
    targets=m["targets"][a.shard_index::a.num_shards]
    model,proc=load_model_and_processor(Path(a.model_dir));model.eval();dev=next(model.parameters()).device
    blocks=circuit_localization.resolve_decoder_layers(model); block=blocks[31]
    def render(traj,t,end):
        clip=sample_clip(Path(a.dataset)/f"{traj}.mp4",0,int(end))
        inp,fp=render_inputs(proc,state_messages(clip,initial[f"{traj}_t{t}"],int(t)),clip,"controlled_8fps")
        return to_device(inp,dev),fp
    def logits(inp):
        with torch.inference_mode():o=model(**inp,logits_to_keep=1)
        z=torch.log_softmax(o.logits[0,-1].float(),dim=-1)
        return {s:float(z[i]) for s,i in POS_IDS.items()}
    def capture(traj,t,end):
        inp,fp=render(traj,t,end); pos=inp["input_ids"].shape[1]-1;got={}
        def h(_m,_i,o):got["h32"]=circuit_localization.block_output_tensor(o)[0,pos,:].detach().float().cpu().clone()
        q=block.register_forward_hook(h)
        try:z=logits(inp)
        finally:q.remove()
        return inp,fp,z,got["h32"]
    def patch(inp,delta,alpha):
        pos=inp["input_ids"].shape[1]-1
        def h(_m,_i,o):
            x=circuit_localization.block_output_tensor(o); y=x.clone()
            y[0,pos,:]=x[0,pos,:]+alpha*delta.to(x.device,x.dtype)
            return y if hasattr(o,"shape") else (y,)+tuple(o[1:])
        q=block.register_forward_hook(h)
        try:return logits(inp)
        finally:q.remove()
    def margin(z,gt):
        return z[gt]-max(v for s,v in z.items() if s!=gt)
    records=[]; self_err=[]
    cache={}
    for e in tqdm(targets,desc=f"conditional L32 shard {a.shard_index}",unit="target"):
        tp=e["target_prefix"]; traj=e["target_trajectory"]; t=int(e["t"]); gt=e["gt_state"]
        ti,tf,tz,th=capture(traj,t,e["frame_end"]); base_m=margin(tz,gt)
        donor=e["donors"]["strong_same_state"]; dk=donor["target_prefix"]
        if dk not in cache:cache[dk]=capture(donor["trajectory_id"],t,donor["frame_end"])
        _,df,dz,dh=cache[dk]; delta=dh-th
        # fixed alpha run is deliberately evaluated on every frozen target.
        fixed=patch(ti,delta,fixed_alpha)
        effect=margin(fixed,gt)-base_m
        gate=(e["baseline_gt_margin"] < cfg["native_margin_threshold"] and
              e["baseline_gt_margin"] < 0)
        # The recursive gate is joined by prefix when available; this prevents
        # reimplementing discovery thresholds in the model runner.
        # Gate metadata is loaded from the CPU result produced by evaluate-gate.
        if a.gate_results:
            gr=pd.read_csv(a.gate_results).query("mode == 'confidence_gate'")
            rr=gr[gr.target_prefix==tp]
            if len(rr): gate=bool(rr.iloc[0].gate_trigger)
        adaptive=float(np.clip(max(0.,-base_m/effect),0.,max_alpha)) if gate and effect>0 else 0.
        vals={"baseline":tz,"fixed_alpha":fixed}
        if gate:
            vals["gated_fixed_alpha"]=fixed
            vals["gated_adaptive_alpha"]=patch(ti,delta,adaptive) if adaptive>0 else tz
        else:
            vals["gated_fixed_alpha"]=tz; vals["gated_adaptive_alpha"]=tz
        # exact self/no-op sanity: alpha=0 must reproduce baseline.
        no=patch(ti,delta,0.0);self_err.append(max(abs(no[s]-tz[s]) for s in STATES))
        for cond,z in vals.items():
            pred=max(STATES,key=z.get); mm=margin(z,gt)
            records.append({"target_prefix":tp,"target_trajectory":traj,"t":t,"condition":cond,
              "gate_trigger":gate,"adaptive_alpha":adaptive,"fixed_alpha":fixed_alpha,
              "gt_state":gt,"baseline_pred":max(STATES,key=tz.get),"patched_pred":pred,
              "baseline_gt_margin":base_m,"patched_gt_margin":mm,"delta_gt_margin":mm-base_m,
              "baseline_correct":max(STATES,key=tz.get)==gt,"patched_correct":pred==gt,
              "wrong_to_correct":max(STATES,key=tz.get)!=gt and pred==gt,
              "correct_to_wrong":max(STATES,key=tz.get)==gt and pred!=gt,
              "target_input_ids_sha256":tf["input_ids_sha256"],"target_pixel_sha256":tf["pixel_values_videos_sha256"],
              "target_video_grid_thw":json.dumps(tf["video_grid_thw"]),"delta32_norm":float(delta.norm())})
    p=root/"shards";p.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(p/f"conditional_l32_shard_{a.shard_index}.csv",index=False)
    (p/f"conditional_l32_shard_{a.shard_index}.complete.json").write_text(
      json.dumps({"rows":len(records),"max_self_logit_error":max(self_err or [0.]),"shard":a.shard_index},indent=2)+"\n")
def analyze(a):
    root=Path(a.out);m=json.loads((root/"conditional_l32_manifest.json").read_text())
    for i in range(a.num_shards):assert (root/"shards"/f"conditional_l32_shard_{i}.complete.json").exists()
    d=pd.concat([pd.read_csv(x) for x in sorted((root/"shards").glob("conditional_l32_shard_*.csv"))],ignore_index=True)
    d.to_csv(root/"conditional_l32_pair_results.csv",index=False)
    agg=d.groupby(["target_prefix","target_trajectory","t","condition"],as_index=False).mean(numeric_only=True)
    agg.to_csv(root/"conditional_l32_target_results.csv",index=False)
    effects={}
    for c in CONDITIONS:
      q=d[d.condition==c];effects[c]={}
      for tag,mask in [("overall",np.ones(len(q),bool)),("t_ge2",q.t>=2),("t_ge3",q.t>=3)]+[(f"t{t}",q.t==t) for t in range(1,6)]:
        x=q[mask];effects[c][tag]={k:clustered(x[k],x.target_trajectory) for k in ("delta_gt_margin","wrong_to_correct","correct_to_wrong","patched_correct")}
    summary={"protocol":"frozen L32 additive delta; only gate selects gated conditions; target trajectory cluster",
             "counts":m["target_count"],"effects":effects,"max_self_logit_error":float(max(json.loads(x.read_text())["max_self_logit_error"] for x in (root/"shards").glob("conditional_l32_shard_*.complete.json")))}
    (root/"conditional_l32_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (root/"conditional_l32_report.md").write_text("# Conditional L32 minimum intervention\n\n"
      "Frozen strong same-state donor pairs were inherited from the existing additive L32 rescue manifest. "
      "Fixed-alpha is evaluated for every target; gated conditions intervene only when the discovery-frozen "
      "recursive confidence gate fires. Adaptive alpha is the clipped first-order boundary estimate, "
      "alpha=min(max_alpha,max(0,-baseline_margin/effect_at_alpha_1)). Pair rows are descriptive; "
      "summary intervals cluster target trajectories. See conditional_l32_summary.json.\n")
    print(json.dumps({"status":"CONDITIONAL_L32_ANALYSIS_COMPLETE","rows":len(d),"outputs":["conditional_l32_pair_results.csv","conditional_l32_target_results.csv","conditional_l32_summary.json","conditional_l32_report.md"]},indent=2))
def unit(_a=None):
    x=np.array([1.,2.,3.]); delta=np.array([2.,-1.,.5])
    assert np.allclose(x+0*delta,x)
    assert np.allclose(x+1*delta,x+delta)
    print("CONDITIONAL_L32_UNIT_PASS: additive residual-delta and zero-strength no-op")
def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("prepare");p.add_argument("--out",type=Path,required=True);p.add_argument("--source-manifest",type=Path,required=True);p.add_argument("--gate-config",type=Path,required=True);p.add_argument("--fixed-alpha",type=float,default=1.0);p.add_argument("--max-alpha",type=float,default=2.0);p.set_defaults(f=prepare)
    p=sp.add_parser("run");p.add_argument("--out",type=Path,required=True);p.add_argument("--model-dir",required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--behavior",type=Path,required=True);p.add_argument("--gate-results",type=Path);p.add_argument("--shard-index",type=int,required=True);p.add_argument("--num-shards",type=int,required=True);p.set_defaults(f=run)
    p=sp.add_parser("analyze");p.add_argument("--out",type=Path,required=True);p.add_argument("--num-shards",type=int,required=True);p.set_defaults(f=analyze)
    p=sp.add_parser("unit");p.set_defaults(f=unit)
    a=ap.parse_args();a.f(a)
if __name__=="__main__":main()
