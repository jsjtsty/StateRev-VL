#!/usr/bin/env python3
"""Causal mediation of history-sensitive residuals into native readout.

Fixed held-out path-dependence pairs are reused.  Only final-token decoder
block-output vectors at L24/L32/L36 are retained for one pair at a time.
The model is loaded only by ``run``.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

LAYERS = (24, 32, 36)
REGIME = "controlled_8fps"
CONDS = ("history", "current_window", "donor_real")
STATES = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
SEED = 20260914

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def block_tensor(out): return out if hasattr(out,"shape") else out[0]

def decoder_layers(model):
    layers=model.model.language_model.layers
    if len(layers)!=36: raise RuntimeError(f"expected 36 decoder layers, got {len(layers)}")
    return layers

def fit_decoders(hidden_path, behavior, split):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z=np.load(hidden_path,allow_pickle=False)
    behavior=behavior.copy(); behavior["target_prefix"]=behavior.trajectory_id+"_t"+behavior.t.astype(str)
    rows=behavior[behavior.trajectory_id.isin(split["discovery_trajectories"])].sort_values("target_prefix")
    missing=[k for k in rows.target_prefix if k not in z.files]
    if missing: raise KeyError(f"baseline hidden cache missing discovery keys: {missing[:3]}")
    out={}
    for l in LAYERS:
        X=np.vstack([z[k][l] for k in rows.target_prefix])
        sc=StandardScaler().fit(X); Xs=sc.transform(X)
        pc=PCA(n_components=min(80,len(Xs)-1),random_state=0).fit(Xs)
        Z=pc.transform(Xs)
        st=LogisticRegression(C=1.0,max_iter=1000,random_state=0).fit(Z,rows.gt_state)
        ev=LogisticRegression(C=1.0,max_iter=1000,random_state=0).fit(Z,rows.gt_event)
        out[l]=(sc,pc,st,ev)
    return out

def decode(decoders, layer, vector):
    sc,pc,st,ev=decoders[layer]
    z=pc.transform(sc.transform(np.asarray(vector)[None,:]))
    sp=st.predict_proba(z)[0]; ep=ev.predict_proba(z)[0]
    return ({str(c):float(v) for c,v in zip(st.classes_,sp)},
            {str(c):float(v) for c,v in zip(ev.classes_,ep)})

def logits(model, inp, pos_ids):
    import torch
    with torch.inference_mode(): out=model(**inp,logits_to_keep=1)
    lp=torch.log_softmax(out.logits[0,-1].float(),dim=-1)
    return {s:float(lp[i]) for s,i in pos_ids.items()}

def capture(model, inp, layers):
    import torch
    blocks=decoder_layers(model); pos=inp["input_ids"].shape[1]-1; got={}; endpoint={}
    handles=[]
    for l in layers:
        def hook(_m,_i,o,l=l):
            got[l]=block_tensor(o)[0,pos,:].detach().float().cpu().numpy()
        handles.append(blocks[l-1].register_forward_hook(hook))
    # Existing hidden_*.npz uses model hidden_states indexing.  Index 36 is
    # after the language-model final norm, unlike the L36 block hook above.
    # Capture that post-norm endpoint so the decoder feature space matches.
    norm_handle=None
    if 36 in layers:
        def norm_hook(_m,_i,o): endpoint[36]=block_tensor(o)[0,pos,:].detach().float().cpu().numpy()
        norm_handle=model.model.language_model.norm.register_forward_hook(norm_hook)
    try:
        with torch.inference_mode(): out=model(**inp,logits_to_keep=1)
        lp=torch.log_softmax(out.logits[0,-1].float(),dim=-1)
        lp={s:float(lp[i]) for s,i in {"Left":5415,"Middle":43935,"Right":5979}.items()}
    finally:
        for h in handles:h.remove()
        if norm_handle is not None: norm_handle.remove()
    for l in layers:
        endpoint.setdefault(l, got[l])
    return lp,got,endpoint

def patched(model, inp, layer, replacement):
    import torch
    blocks=decoder_layers(model); pos=inp["input_ids"].shape[1]-1
    endpoint={}
    def hook(_m,_i,o):
        h=block_tensor(o); q=h.clone()
        if tuple(replacement.shape)!=(h.shape[-1],): raise RuntimeError("patch shape mismatch")
        q[0,pos,:]=torch.as_tensor(replacement,device=h.device,dtype=h.dtype)
        endpoint["block"]=q[0,pos,:].detach().float().cpu().numpy()
        return q if hasattr(o,"shape") else (q,)+tuple(o[1:])
    norm_handle=None
    if layer==36:
        def norm_hook(_m,_i,o): endpoint["endpoint"]=block_tensor(o)[0,pos,:].detach().float().cpu().numpy()
        norm_handle=model.model.language_model.norm.register_forward_hook(norm_hook)
    handle=blocks[layer-1].register_forward_hook(hook)
    try:
        lp=logits(model,inp,{"Left":5415,"Middle":43935,"Right":5979})
    finally:
        handle.remove()
        if norm_handle is not None: norm_handle.remove()
    return lp, endpoint.get("endpoint", endpoint["block"])
def build_clips(tc,dc,t):
    import mg_common as mg
    history=tc.copy(); n=mg.expected_frames(REGIME,len(tc)); idx=mg.sampled_indices(len(tc),n); mask=mg.window_sample_mask(len(tc),t,n)
    history[idx[~mask]]=dc[idx[~mask]]
    current=mg.transplant_sampled_window(tc,dc,t,REGIME)[0]
    return {"target_real":tc,"history":history,"current_window":current,"donor_real":dc}

def run(a):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import mg_common as mg
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS,load_model_and_processor,render_inputs,to_device
    root=Path(a.out); m=json.loads((root/"history_readout_manifest.json").read_text())
    behavior=pd.read_csv(a.behavior); behavior["target_prefix"]=behavior.trajectory_id+"_t"+behavior.t.astype(str)
    split=json.loads(Path(a.split).read_text()); dec=fit_decoders(a.baseline_hidden,behavior,split)
    model,processor=load_model_and_processor(Path(a.model_dir)); model.eval(); dev=next(model.parameters()).device
    pairs=m["pairs"][a.shard_index::a.num_shards]; rows=[]
    for p in __import__('tqdm').tqdm(pairs,desc=f"history mediation shard {a.shard_index}",unit="pair",dynamic_ncols=True):
        tc=sample_clip(Path(a.dataset)/f"{p['target_trajectory']}.mp4",0,p['frame_end'])
        dc=sample_clip(Path(a.dataset)/f"{p['donor_trajectory']}.mp4",0,p["donor_frame_end"])
        assert len(tc)==len(dc)==p["frame_end"]==p["donor_frame_end"]
        clips=build_clips(tc,dc,p["t"])
        inputs={}; fps={}; clean={}; acts={}; endpoint_acts={}
        for c,clip in clips.items():
            inp,fp=render_inputs(processor,state_messages(clip,p["initial_state"],p["t"]),clip,REGIME)
            inputs[c]=to_device(inp,dev); fps[c]=fp; clean[c],acts[c],endpoint_acts[c]=capture(model,inputs[c],LAYERS)
        ref=fps["target_real"]
        for c in CONDS:
            assert fps[c]["input_ids_sha256"]==ref["input_ids_sha256"]
            assert fps[c]["video_grid_thw"]==ref["video_grid_thw"]
            assert fps[c]["pixel_values_videos_shape"]==ref["pixel_values_videos_shape"]
        for layer in LAYERS:
            # self patch: target activation into target must be numerical zero.
            self_lp,_=patched(model,inputs["target_real"],layer,acts["target_real"][layer])
            self_err=max(abs(self_lp[s]-clean["target_real"][s]) for s in STATES)
            for c in CONDS:
                # Sufficiency: target input receives source activation.
                suff_lp,suff_endpoint=patched(model,inputs["target_real"],layer,acts[c][layer])
                # Necessity/reversal: source input receives target activation.
                nec_lp,nec_endpoint=patched(model,inputs[c],layer,acts["target_real"][layer])
                for direction,source_cond,lp in (("sufficiency",c,suff_lp),("necessity",c,nec_lp)):
                    vec=(suff_endpoint if direction=="sufficiency" else nec_endpoint)
                    sp,ep=decode(dec,layer,vec); gt=p["gt_state"]; event=p["event"]
                    native={s:lp[s]-clean["target_real"][s] for s in STATES}
                    target_sp,target_ep=decode(dec,layer,endpoint_acts["target_real"][layer]); source_sp,source_ep=decode(dec,layer,endpoint_acts[c][layer])
                    rows.append({"pair_id":p["pair_id"],"target_prefix":p["target_prefix"],"target_trajectory":p["target_trajectory"],"donor_trajectory":p["donor_trajectory"],"t":p["t"],"condition":c,"direction":direction,"layer":layer,"gt_state":gt,"event":event,"clean_target_gt_logit":clean["target_real"][gt],"clean_source_gt_logit":clean[c][gt],"patched_gt_logit":lp[gt],"delta_logit_Left":native["Left"],"delta_logit_Middle":native["Middle"],"delta_logit_Right":native["Right"],"native_cf_margin_change":(lp[gt]-lp[next(s for s in STATES if s!=gt)])-(clean["target_real"][gt]-clean["target_real"][next(s for s in STATES if s!=gt)]),"state_probe_gt":sp.get(gt),"target_state_probe_gt":target_sp.get(gt),"source_state_probe_gt":source_sp.get(gt),"event_probe_current":ep.get(event),"target_event_probe_current":target_ep.get(event),"source_event_probe_current":source_ep.get(event),"self_patch_max_logit_error":self_err,"input_ids_sha256":fps[c]["input_ids_sha256"],"pixel_sha256":fps[c]["pixel_values_videos_sha256"],"video_grid_thw":json.dumps(fps[c]["video_grid_thw"]),"prompt_hash":fps[c]["prompt_text_sha256"]})
    out=root/"shards"; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/f"mediation_shard_{a.shard_index}.csv",index=False)
    (out/f"mediation_shard_{a.shard_index}.meta.json").write_text(json.dumps({"pairs":len(pairs),"rows":len(rows),"layers":LAYERS,"conditions":CONDS},indent=2)+"\n")
    print(json.dumps({"status":"HISTORY_MEDIATION_SHARD_COMPLETE","shard":a.shard_index,"pairs":len(pairs),"rows":len(rows)},indent=2))

def prepare(a):
    src=Path(a.path_manifest); m=json.loads(src.read_text()); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    forwards_per_pair=(len(CONDS)+1)+2*len(CONDS)*len(LAYERS)+len(LAYERS)
    outm={"version":2,"seed":SEED,"source_manifest":str(src),"source_manifest_sha256":sha256(src),"split":m["split"],"behavior":m["behavior"],"baseline_hidden":m["baseline_hidden"],"pairs":m["pairs"],"layers":list(LAYERS),"conditions":list(CONDS),"patch_location":"final input token; patch decoder block output before downstream blocks; endpoint decoding uses block output at L24/L32 and post-final-norm output at L36 to match hidden cache","statistics":"target-prefix aggregation then target-trajectory cluster bootstrap/sign permutation","counts":{"pairs":len(m["pairs"]),"target_trajectories":len({p["target_trajectory"] for p in m["pairs"]}),"conditions":len(CONDS),"layers":len(LAYERS),"forwards_per_pair":forwards_per_pair,"total_forwards":len(m["pairs"])*forwards_per_pair}}
    (out/"history_readout_manifest.json").write_text(json.dumps(outm,indent=2)+"\n"); print(json.dumps({"status":"HISTORY_MEDIATION_MANIFEST_READY","counts":outm["counts"]},indent=2))

def audit(a):
    m=json.loads((Path(a.out)/"history_readout_manifest.json").read_text()); assert sha256(m["source_manifest"])==m["source_manifest_sha256"]; assert tuple(m["layers"])==LAYERS; assert tuple(m["conditions"])==CONDS; assert len(m["pairs"])==36; assert "post-final-norm" in m["patch_location"]; print(json.dumps({"status":"HISTORY_MEDIATION_AUDIT_PASS","counts":m["counts"],"layer36_endpoint":"post-final-norm"},indent=2))

def unit():
    import torch
    h=torch.zeros(1,4,3); r=torch.ones(3); q=h.clone(); q[0,2]=r; assert torch.equal(q[0,2],r) and torch.equal(q[0,0],h[0,0]); print("HISTORY_MEDIATION_UNIT_PASS")

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True); common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--out",default="outputs/vetbench/history_readout_mediation_v1"); common.add_argument("--behavior",default="outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"); common.add_argument("--split",default="outputs/vetbench/circuit_localization_v2/discovery_validation_split.json"); common.add_argument("--baseline-hidden",default="outputs/vetbench/mechanism_gate_final/hidden_baseline.npz"); common.add_argument("--dataset",default="dataset/vetbench/cup"); common.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct")
    p=sub.add_parser("prepare",parents=[common]); p.add_argument("--path-manifest",default="outputs/vetbench/path_dependence_v1/path_dependence_manifest.json")
    sub.add_parser("audit",parents=[common]); sub.add_parser("unit")
    p=sub.add_parser("run",parents=[common]); p.add_argument("--shard-index",type=int,required=True); p.add_argument("--num-shards",type=int,default=8)
    a=ap.parse_args(); prepare(a) if a.mode=="prepare" else audit(a) if a.mode=="audit" else unit() if a.mode=="unit" else run(a)
if __name__=="__main__": main()
