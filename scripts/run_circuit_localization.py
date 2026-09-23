#!/usr/bin/env python3
"""GPU runner for strict event and module circuit localization.

Only explicit ``--stage discovery`` or ``--stage validation`` loads the model.
It captures one final-token activation per decoder block for the current pair,
then immediately releases it after the bidirectional patch sweep. No full
sequence activation is persisted.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parent))
import circuit_localization as plan
import mg_common as mg
from state_rev_input_pipeline import load_model_and_processor,render_inputs,to_device,POS_IDS
from run_state_rev_audit import state_messages
from run_vetbench_screening import sample_clip

LAYERS=(4,8,12,16,20,24,28,32,36)
STATES=("Left","Middle","Right")
EVENTS=("Left and Middle","Middle and Right","Left and Right")

def logits(model, inp):
    with torch.inference_mode():
        x=model(**inp,logits_to_keep=1).logits[0,-1].float()
    lp=torch.log_softmax(x,dim=-1)
    return {s:float(lp[i].item()) for s,i in POS_IDS.items()}

def final_norm(model):
    return model.model.language_model.norm

def patch_module_for(block, patch_module):
    if patch_module == "block_output": return block
    if patch_module == "attention_output": return block.self_attn
    if patch_module == "mlp_output": return block.mlp
    raise ValueError(f"unknown patch module: {patch_module}")

def capture(model, inp, layers, patch_module="block_output"):
    blocks=plan.resolve_decoder_layers(model); pos=inp["input_ids"].shape[1]-1; got={}; final={}; handles=[]
    for report_layer in layers:
        idx=report_layer-1
        module=patch_module_for(blocks[idx],patch_module)
        def hook(_m,_i,o,idx=idx): got[idx]=plan.block_output_tensor(o)[0,pos,:].detach().float().cpu().clone()
        handles.append(module.register_forward_hook(hook))
    def norm_hook(_m,_i,o): final["x"]=plan.block_output_tensor(o)[0,pos,:].detach().float().cpu().clone()
    handles.append(final_norm(model).register_forward_hook(norm_hook))
    try:
        with torch.inference_mode(): out=model(**inp,logits_to_keep=1)
        lp=torch.log_softmax(out.logits[0,-1].float(),dim=-1)
    finally:
        for h in handles:h.remove()
    return ({s:float(lp[i].item()) for s,i in POS_IDS.items()},got,final["x"])

def patch_forward(model,inp,report_layer,replacement,patch_module="block_output"):
    block=plan.resolve_decoder_layers(model)[report_layer-1]; pos=inp["input_ids"].shape[1]-1
    module=patch_module_for(block,patch_module)
    final={}
    def hook(_m,_i,o):
        h=plan.block_output_tensor(o)
        if tuple(replacement.shape)!=(h.shape[-1],): raise RuntimeError("final-token replacement shape mismatch")
        q=h.clone(); q[0,pos,:]=replacement.to(q.device,q.dtype)
        return q if hasattr(o,"shape") else (q,)+tuple(o[1:])
    def norm_hook(_m,_i,o): final["x"]=plan.block_output_tensor(o)[0,pos,:].detach().float().cpu().clone()
    hd=module.register_forward_hook(hook); hn=final_norm(model).register_forward_hook(norm_hook)
    try: return logits(model,inp),final["x"]
    finally: hd.remove(); hn.remove()

def event_decoder(hidden, rows, fit_trajectories, heldout_traj=None):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    fit=set(fit_trajectories)
    if heldout_traj in fit: fit.remove(heldout_traj)
    train=[r for r in rows if r["trajectory_id"] in fit]
    keys=[r["key"] for r in train]
    y=np.array([EVENTS.index(r["gt_event"]) for r in train])
    X=np.vstack([hidden[k] for k in keys]); sc=StandardScaler().fit(X); X=sc.transform(X); pc=PCA(n_components=min(80,X.shape[0]-1),random_state=0).fit(X); X=pc.transform(X)
    clf=LogisticRegression(C=1.0,max_iter=500).fit(X,y)
    def predict(x):
        q=clf.predict_proba(pc.transform(sc.transform(np.asarray(x)[None,:])))[0]; p=np.zeros(3)
        for j,c in enumerate(clf.classes_):p[int(c)]=q[j]
        return p
    return predict

def apply_event(prev,event):
    a,b=event.split(" and "); return b if prev==a else a if prev==b else prev

def render_clip(processor,clip,initial,t):
    return render_inputs(processor,state_messages(clip,initial,t),clip,"controlled_8fps")[0]

def one_pair(model,processor,rows,p,condition,device,layers,base_hidden,dataset,
             decoder_fit_trajectories,patch_module="block_output"):
    tc=sample_clip(Path(dataset)/f"{p['target_traj']}.mp4",0,p["frame_end"])
    sc=sample_clip(Path(dataset)/f"{p['source_traj']}.mp4",0,p["frame_end"])
    same=sample_clip(Path(dataset)/f"{p['same_event_traj']}.mp4",0,p["frame_end"])
    if condition=="target_self": vc=tc
    elif condition=="same_event_source": vc,_=mg.transplant_sampled_window(tc,same,p["t"],"controlled_8fps")
    elif condition=="matched_history_transplant": vc,_=mg.transplant_sampled_window(tc,sc,p["t"],"controlled_8fps",destination="history")
    else: vc,_=mg.transplant_sampled_window(tc,sc,p["t"],"controlled_8fps")
    ti=to_device(render_clip(processor,tc,p["target_initial"],p["t"]),device)
    vi=to_device(render_clip(processor,vc,p["target_initial"],p["t"]),device)
    tl,ta,tfinal=capture(model,ti,layers,patch_module); vl,va,vfinal=capture(model,vi,layers,patch_module)
    event_pred=event_decoder(base_hidden,rows,decoder_fit_trajectories,p["target_traj"])
    es=event_pred(vfinal.numpy()); et=event_pred(tfinal.numpy())
    st,cf=p["target_state"],p["counterfactual_state"]
    src_i=EVENTS.index(p["source_event"]); tgt_i=EVENTS.index(p["target_event"])
    clean_target_event_margin=float(et[src_i]-et[tgt_i])
    clean_hybrid_event_margin=float(es[src_i]-es[tgt_i])
    out=[]
    for l in layers:
        suff,sfinal=patch_forward(model,ti,l,va[l-1],patch_module)
        nec,nfinal=patch_forward(model,vi,l,ta[l-1],patch_module)
        s_event=event_pred(sfinal.numpy()); n_event=event_pred(nfinal.numpy())
        for direction,lp in (("sufficiency",suff),("necessity",nec)):
            ep=s_event if direction=="sufficiency" else n_event
            out.append({"pair_id":p["pair_id"],"target_prefix":f"{p['target_traj']}_t{p['t']}","target_traj":p["target_traj"],"t":p["t"],"condition":condition,"direction":direction,"layer":l,"patch_module":patch_module,"target_state":st,"counterfactual_state":cf,"third_state":next(x for x in STATES if x not in {st,cf}),"logit_cf":lp[cf],"logit_target":lp[st],"logit_third":lp[next(x for x in STATES if x not in {st,cf})],"event_source_prob":float(ep[src_i]),"event_target_prob":float(ep[tgt_i]),"event_margin_source_minus_target":float(ep[src_i]-ep[tgt_i]),"clean_target_cf_margin":tl[cf]-tl[st],"clean_hybrid_cf_margin":vl[cf]-vl[st],"clean_target_event_source_prob":float(et[src_i]),"clean_target_event_target_prob":float(et[tgt_i]),"clean_hybrid_event_source_prob":float(es[src_i]),"clean_hybrid_event_target_prob":float(es[tgt_i]),"clean_target_event_margin":clean_target_event_margin,"clean_hybrid_event_margin":clean_hybrid_event_margin,"decoder_fit_stage":"discovery","decoder_fit_trajectory_count":len(decoder_fit_trajectories)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("discovery","validation"),required=True); ap.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct"); ap.add_argument("--dataset",default="dataset/vetbench/cup"); ap.add_argument("--out",default="outputs/vetbench/circuit_localization_v2"); ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--layers",default=','.join(map(str,LAYERS))); ap.add_argument("--patch-module",choices=("block_output","attention_output","mlp_output"),default="block_output"); args=ap.parse_args()
    out=Path(args.out); split=json.loads((out/"discovery_validation_split.json").read_text()); manifest=json.loads((out/"discovery_pair_manifest.json").read_text()); pairs=manifest["discovery_pairs"] if args.stage=="discovery" else manifest["validation_pairs"]
    pairs=pairs[args.shard_index::args.num_shards]; rows=plan.read_rows(Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"))
    base=np.load("outputs/vetbench/mechanism_gate_final/hidden_baseline.npz"); base_hidden={k:base[k][-1] for k in base.files}
    model,processor=load_model_and_processor(Path(args.model_dir)); dev=next(model.parameters()).device; layers=[int(x) for x in args.layers.split(',')]
    selected=[]
    for p in tqdm(pairs,desc=f"{args.stage} shard {args.shard_index}",unit="pair"):
        for c in ("target_self","same_event_source","matched_history_transplant","source_current_transplant"):
            selected.extend(one_pair(model,processor,rows,p,c,dev,layers,base_hidden,args.dataset,split["discovery_trajectories"],args.patch_module))
    shard_out=out/"shards"/args.stage/f"shard_{args.shard_index}" if args.num_shards>1 else out
    shard_out.mkdir(parents=True,exist_ok=True)
    name=f"strict_{args.patch_module}_{args.stage}.csv"; path=shard_out/name; new=not path.exists(); fields=list(selected[0])
    with path.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); 
        if new:w.writeheader()
        w.writerows(selected)
    print(json.dumps({"stage":args.stage,"shard":args.shard_index,"pairs":len(pairs),"rows_written":len(selected),"output":str(path)},indent=2))
if __name__=="__main__": main()
