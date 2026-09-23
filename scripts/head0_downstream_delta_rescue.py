#!/usr/bin/env python3
"""Composable L24 Head0 + L32 residual-delta rescue.

The L32 intervention is additive, not an overwrite:
  h32' = h32_current + beta * (h32_donor - h32_target)
For the joint condition, h32_current is captured after the L24 Head0 patch.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HEAD_LAYER, HEAD, HEAD_DIM = 24, 0, 128
DOWNSTREAM_LAYER = 32
ALPHAS = (1.0,)
BETAS = (0.25, 0.5, 1.0)
SEED = 20260911
REGIME = "controlled_8fps"
DONOR_TYPES = ("strong_same_state", "different_event")
SOURCE_DONORS = {"strong_same_state": "strong_same_state", "different_event": "different_event"}
BASE_CONDITIONS = ("baseline", "head0_only", "l32_delta_only", "head0_l32_delta")
STATES = ("Left", "Middle", "Right")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def build_manifest(args):
    source_path = Path(args.joint_manifest)
    source = json.loads(source_path.read_text())
    if source["patches"]["head0"]["layer_one_based"] != HEAD_LAYER or source["patches"]["head0"]["head"] != HEAD:
        raise AssertionError("joint manifest does not contain frozen L24 Head 0")
    targets=[]
    for x in source["targets"]:
        targets.append({k:x[k] for k in ("target_prefix","target_trajectory","t","frame_end","initial_state","prev_state","current_event","gt_state","strength_group","baseline_gt_margin")}|{
            "baseline_incorrect": bool(x["baseline_gt_margin"] < 0),
        "donors": {k:x["donors"][v] for k,v in SOURCE_DONORS.items()}})
    return {
        "version":1, "seed":SEED, "source_joint_manifest":str(source_path),
        "source_joint_manifest_sha256":sha256(source_path), "frozen_pairs":True,
        "donor_types":list(DONOR_TYPES), "conditions":list(BASE_CONDITIONS),
        "betas":list(BETAS), "head_alpha":1.0,
        "patches": {"head0": source["patches"]["head0"], "downstream": {
            "layer_one_based":DOWNSTREAM_LAYER, "token":"final prompt token",
            "location":"decoder block output", "operation":"add beta*(donor baseline - target baseline)"}},
        "intervention_order": "capture target/donor baseline h32; Head0 hook runs at L24 pre-o_proj; L32 additive hook runs on current post-block output",
        "delta_definition":"delta32 = h32_donor_baseline - h32_target_baseline",
        "sanity_rules": {"beta_zero":"joint beta=0 equals Head0-only", "nonidentity":"beta>0 joint is not required to equal L32-delta-only and must be checked", "delta_source":"delta32 comes only from unmodified baseline captures"},
        "statistics":"target-prefix aggregation then target-trajectory cluster bootstrap/sign permutation; pairs are not independent",
        "regime":source["regime"], "dataset":source["dataset"], "target_split":source["target_split"], "donor_pool":source["donor_pool"],
        "independence_caveat":source["independence_caveat"], "weak_rule":source["weak_rule"], "weak_threshold":source["weak_threshold"],
        "baseline_incorrect_rule":"baseline GT margin < 0",
        "counts": {"target_prefixes":len(targets), "target_trajectories":len({x["target_trajectory"] for x in targets}), "donor_types":len(DONOR_TYPES), "betas":len(BETAS), "rows_expected":len(targets)*len(DONOR_TYPES)*len(BETAS)*len(BASE_CONDITIONS), "weak":sum(x["strength_group"]=="weak" for x in targets), "baseline_margin_lt_zero":sum(x["baseline_gt_margin"]<0 for x in targets)},
        "targets":targets}

def audit_manifest(args):
    root=Path(args.out); m=json.loads((root/"delta_rescue_manifest.json").read_text())
    source=Path(m["source_joint_manifest"])
    assert sha256(source)==m["source_joint_manifest_sha256"]
    assert tuple(m["conditions"])==BASE_CONDITIONS and tuple(m["betas"])==BETAS
    original=json.loads(source.read_text()); by={x["target_prefix"]:x for x in original["targets"]}
    for x in m["targets"]:
        o=by[x["target_prefix"]]
        assert x["donors"]["strong_same_state"]==o["donors"][SOURCE_DONORS["strong_same_state"]]
        assert x["donors"]["different_event"]==o["donors"][SOURCE_DONORS["different_event"]]
        for d in x["donors"].values(): assert d["trajectory_id"]!=x["target_trajectory"] and d["t"]==x["t"] and d["prev_state"]==x["prev_state"]
        assert x["donors"]["strong_same_state"]["event"]==x["current_event"] and x["donors"]["strong_same_state"]["state"]==x["gt_state"]
        assert x["donors"]["different_event"]["event"]!=x["current_event"] and x["donors"]["different_event"]["state"]!=x["gt_state"]
    print(json.dumps({"status":"DELTA_RESCUE_AUDIT_PASS","counts":m["counts"],"betas":m["betas"],"intervention_order":m["intervention_order"]},indent=2))

def prepare(args):
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True); m=build_manifest(args)
    (root/"delta_rescue_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps({"status":"DELTA_RESCUE_MANIFEST_READY","counts":m["counts"],"betas":m["betas"],"frozen_pairs":True,"matching":"inherited frozen joint manifest"},indent=2))

def unit():
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from circuit_localization import block_output_tensor, replace_positions_hook
    from head_localization_l24 import pre_oproj_hook
    t=torch.randn(1,128); d=torch.randn(1,128); x=torch.randn(1,4,4096)
    assert torch.allclose(t+0*(d-t),t) and torch.allclose(t+1*(d-t),d)
    h=pre_oproj_hook(d,[0],3)(None,(x,))[0]; assert torch.equal(h[0,3,:128],d[0])
    delta=d-t; add=replace_positions_hook([3],x[:,3,:]+0.5*torch.randn(1,4096))
    y=add(None,None,x); assert y.shape==x.shape
    assert torch.allclose(t+0*(d-t),t)
    print("DELTA_RESCUE_UNIT_PASS: additive delta algebra, Head0 slice, and block hook shape")

def run(args):
    import torch
    from tqdm import tqdm
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import circuit_localization
    from head_localization_l24 import pre_oproj_hook
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS, load_model_and_processor, render_inputs, to_device
    root=Path(args.out); m=json.loads((root/"delta_rescue_manifest.json").read_text()); targets=m["targets"][args.shard_index::args.num_shards]
    print(json.dumps({"shard":args.shard_index,"targets":len(targets),"rows_expected":len(targets)*len(DONOR_TYPES)*len(BETAS)*len(BASE_CONDITIONS)},indent=2))
    model,processor=load_model_and_processor(Path(args.model_dir)); model.eval(); device=next(model.parameters()).device
    blocks=circuit_localization.resolve_decoder_layers(model); oproj=blocks[HEAD_LAYER-1].self_attn.o_proj; down=blocks[DOWNSTREAM_LAYER-1]
    attn=blocks[HEAD_LAYER-1].self_attn
    if int(getattr(attn,"num_heads",32))!=32 or int(getattr(attn,"head_dim",HEAD_DIM))!=HEAD_DIM: raise RuntimeError("unexpected L24 attention structure")
    def render(p):
        clip=sample_clip(Path(args.dataset)/f"{p['trajectory_id']}.mp4",0,int(p["frame_end"]))
        inp,fp=render_inputs(processor,state_messages(clip,p["initial_state"],int(p["t"])),clip,REGIME)
        return to_device(inp,device),fp
    def logits(inp):
        with torch.inference_mode(): out=model(**inp,logits_to_keep=1)
        lp=torch.log_softmax(out.logits[0,-1].float(),dim=-1)
        return {s:float(lp[i]) for s,i in POS_IDS.items()}
    def capture(p):
        inp,fp=render(p); pos=inp["input_ids"].shape[1]-1; got={}
        def hh(_m,v): got["head0"]=v[0][0,pos,:HEAD_DIM].detach().float().cpu().clone(); return v
        def bh(_m,_i,o): got["h32"]=circuit_localization.block_output_tensor(o)[0,pos,:].detach().float().cpu().clone()
        hs=[oproj.register_forward_pre_hook(hh),down.register_forward_hook(bh)]
        try: native=logits(inp)
        finally:
            for h in hs:h.remove()
        return inp,fp,native,got
    def patched(inp, head=None, delta=None, beta=0.):
        pos=inp["input_ids"].shape[1]-1; hs=[]
        if head is not None: hs.append(oproj.register_forward_pre_hook(pre_oproj_hook(head.reshape(1,HEAD_DIM),[HEAD],pos)))
        if delta is not None:
            def add(_m,_i,o):
                h=circuit_localization.block_output_tensor(o); q=h.clone(); q[0,pos,:]=h[0,pos,:]+beta*delta.to(h.device,h.dtype)
                return q if hasattr(o,"shape") else (q,)+tuple(o[1:])
            hs.append(down.register_forward_hook(add))
        try: return logits(inp)
        finally:
            for h in hs:h.remove()
    cache={}; records=[]; nonidentity=[]; beta0_errors=[]
    for entry in tqdm(targets,desc=f"delta shard {args.shard_index}",unit="target",dynamic_ncols=True):
        target={"trajectory_id":entry["target_trajectory"],"target_prefix":entry["target_prefix"],"t":entry["t"],"frame_end":entry["frame_end"],"initial_state":entry["initial_state"]}
        ti,tf,tbase,tact=capture(target)
        for donor_type in DONOR_TYPES:
            d=entry["donors"][donor_type]; key=d["target_prefix"]
            if key not in cache: cache[key]=capture({"trajectory_id":d["trajectory_id"],"target_prefix":key,"t":d["t"],"frame_end":d["frame_end"],"initial_state":d.get("initial_state",entry["initial_state"])})
            _,df,dbase,dact=cache[key]; delta32=dact["h32"]-tact["h32"]; gt=entry["gt_state"]; desired=gt if donor_type=="strong_same_state" else d["state"]
            basepred=max(STATES,key=tbase.get); basegm=tbase[gt]-max(tbase[s] for s in STATES if s!=gt)
            computed={"baseline":tbase}
            # The donor Head0 activation is required here. Using tact["head0"]
            # would be a target self-patch and would collapse the joint
            # condition to L32-delta-only, falsely triggering the nonidentity
            # sanity check.
            donor_head0 = dact["head0"]
            head0_logits = patched(ti, donor_head0, None)
            beta0_joint = patched(ti, donor_head0, delta32, 0.0)
            beta0_error = max(abs(beta0_joint[s] - head0_logits[s]) for s in STATES)
            beta0_errors.append(beta0_error)
            if beta0_error > 1e-5:
                raise AssertionError(f"beta=0 joint differs from Head0-only: {beta0_error}")
            for beta in BETAS:
                computed[("head0_only",beta)]=head0_logits
                computed[("l32_delta_only",beta)]=patched(ti,None,delta32,beta)
                computed[("head0_l32_delta",beta)]=patched(ti,donor_head0,delta32,beta)
                diff=max(abs(computed[("head0_l32_delta",beta)][s]-computed[("l32_delta_only",beta)][s]) for s in STATES)
                nonidentity.append(diff)
                for condition in BASE_CONDITIONS:
                    z=tbase if condition=="baseline" else computed[(condition,beta)]
                    pred=max(STATES,key=z.get); gm=z[gt]-max(z[s] for s in STATES if s!=gt)
                    cm=(z[desired]-z[gt]) if desired!=gt else np.nan
                    records.append({"target_prefix":entry["target_prefix"],"target_trajectory":entry["target_trajectory"],"t":entry["t"],"donor_type":donor_type,"condition":condition,"beta":beta,"donor_prefix":key,"donor_trajectory":d["trajectory_id"],"prev_state":entry["prev_state"],"target_event":entry["current_event"],"donor_event":d["event"],"gt_state":gt,"desired_state":desired,"strength_group":entry["strength_group"],"baseline_gt_margin":basegm,"patched_gt_margin":gm,"delta_gt_margin":gm-basegm,"delta_counterfactual_margin":cm-(tbase[desired]-tbase[gt] if desired!=gt else np.nan),"baseline_pred":basepred,"patched_pred":pred,"baseline_correct":basepred==gt,"patched_correct":pred==gt,"wrong_to_correct":basepred!=gt and pred==gt,"correct_to_wrong":basepred==gt and pred!=gt,"flips_to_desired":basepred!=desired and pred==desired,"gt_state_specificity":(z[gt]-tbase[gt])-max(z[s]-tbase[s] for s in STATES if s!=gt),"delta_logit_gt":z[gt]-tbase[gt],"delta_logit_desired":z[desired]-tbase[desired],"target_input_ids_sha256":tf["input_ids_sha256"],"target_pixel_sha256":tf["pixel_values_videos_sha256"],"target_video_grid_thw":json.dumps(tf["video_grid_thw"]),"donor_input_ids_sha256":df["input_ids_sha256"],"donor_pixel_sha256":df["pixel_values_videos_sha256"],"donor_video_grid_thw":json.dumps(df["video_grid_thw"]),"delta32_norm":float(delta32.norm()),"joint_minus_l32_only_max_abs":diff,"delta_source":"baseline target/donor h32"})
    if not nonidentity or max(nonidentity)<=1e-7: raise AssertionError("joint logits are identically equal to L32-delta-only; check donor Head0 patch")
    out=root/"shards"/f"delta_rescue_shard_{args.shard_index}.csv"; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(records).to_csv(out,index=False)
    print(json.dumps({"status":"DELTA_RESCUE_SHARD_COMPLETE","output":str(out),"targets":len(targets),"rows":len(records),"max_joint_minus_l32_only":max(nonidentity),"max_beta0_joint_vs_head0":max(beta0_errors)},indent=2))

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="command",required=True); common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--out",default="outputs/vetbench/head0_downstream_delta_rescue_v1"); common.add_argument("--joint-manifest",default="outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_manifest.json"); common.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct"); common.add_argument("--dataset",default="dataset/vetbench/cup")
    sub.add_parser("prepare",parents=[common]); sub.add_parser("audit",parents=[common]); sub.add_parser("unit",parents=[common]); r=sub.add_parser("run",parents=[common]); r.add_argument("--shard-index",type=int,required=True); r.add_argument("--num-shards",type=int,default=8)
    a=ap.parse_args()
    if a.command=="prepare": prepare(a)
    elif a.command=="audit": audit_manifest(a)
    elif a.command=="unit": unit()
    else: run(a)
if __name__=="__main__": main()
