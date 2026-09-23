#!/usr/bin/env python3
"""Preparation and hook primitives for staged circuit localization.

This module is intentionally safe by default: ``prepare`` and ``dry-run`` are
CPU-only and never load the model. ``smoke`` is the only mode that loads a
model, and must be invoked explicitly by the user.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path("outputs/vetbench/circuit_localization_v1")
SRC=Path("outputs/vetbench/mechanism_gate_final")
COARSE=(4,8,12,16,20,24,28,32,36)
N_LAYERS=36

def read_rows(path):
    rows=list(csv.DictReader(path.open(newline="")))
    for r in rows:
        r["t"]=int(r["t"]); r["key"]=f"{r['trajectory_id']}_t{r['t']}"
    assert len(rows)==250
    return rows

def build_split(rows, seed=20260906, discovery_n=30):
    trajectories=sorted({r["trajectory_id"] for r in rows})
    rng=np.random.default_rng(seed); perm=list(rng.permutation(trajectories))
    discovery=sorted(perm[:discovery_n]); validation=sorted(perm[discovery_n:])
    assert not set(discovery)&set(validation) and len(discovery)+len(validation)==50
    return {"seed":seed,"unit":"target trajectory","discovery_trajectories":discovery,
            "validation_trajectories":validation,"discovery_n":len(discovery),"validation_n":len(validation)}

def pair_manifest(pairs, split):
    disc=set(split["discovery_trajectories"]); val=set(split["validation_trajectories"])
    by_prefix={}
    for p in sorted(pairs,key=lambda x:x["pair_id"]):
        prefix=f"{p['target_traj']}_t{p['t']}"
        by_prefix.setdefault(prefix,p)
    discovery=[by_prefix[k] for k in sorted(by_prefix) if by_prefix[k]["target_traj"] in disc]
    validation=[p for p in pairs if p["target_traj"] in val]
    return {"selection":"one lexicographically first source pair per target prefix; no model results used",
            "discovery_pairs":discovery,"validation_pairs":validation,
            "n_discovery_pairs":len(discovery),"n_validation_pairs":len(validation),
            "n_all_pairs":len(pairs)}

def resolve_decoder_layers(model):
    """Return Qwen3-VL decoder blocks and fail loudly on a different model path."""
    layers=model.model.language_model.layers
    if len(layers)!=N_LAYERS: raise RuntimeError(f"expected 36 decoder layers, got {len(layers)}")
    return layers

def block_output_tensor(output):
    """Qwen decoder blocks return hidden tensor or tuple(hidden, auxiliaries)."""
    return output if hasattr(output,"shape") else output[0]

def replace_positions_hook(positions, replacement):
    """Build a block-output hook; positions are absolute sequence positions.

    Semantics are explicitly ``decoder block output``. The replacement must be
    [n_positions, hidden_dim], matching the target block output dtype/device.
    """
    pos=np.asarray(positions,dtype=np.int64)
    def hook(_module,_inputs,output):
        h=block_output_tensor(output)
        if h.ndim!=3 or replacement.ndim!=2 or replacement.shape != (len(pos),h.shape[-1]):
            raise RuntimeError(f"hook shape mismatch output={tuple(h.shape)} replacement={tuple(replacement.shape)} positions={len(pos)}")
        q=h.clone(); q[:,pos,:]=replacement.to(device=h.device,dtype=h.dtype)
        return q if hasattr(output,"shape") else (q,)+tuple(output[1:])
    return hook

def assert_activation_shape(target, source, positions):
    if tuple(target.shape)!=tuple(source.shape): raise AssertionError(f"activation shape mismatch {target.shape} vs {source.shape}")
    if target.ndim!=2 or target.shape[0]!=len(positions): raise AssertionError("activation/position mismatch")

def run_prepare(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(Path(args.behavior))
    if args.split:
        split=json.loads(Path(args.split).read_text())
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from content_disjoint_split import assert_content_disjoint
        fingerprint=Path(split.get("fingerprint_source", "outputs/vetbench/validity_gate_v1/input_fingerprints.csv"))
        assert_content_disjoint(split, fingerprint)
    else:
        split=build_split(rows,args.seed,args.discovery_n)
    pairs=json.loads((Path(args.source)/"pair_manifest.json").read_text())["pairs"]
    manifests=pair_manifest(pairs,split)
    (out/"discovery_validation_split.json").write_text(json.dumps(split,indent=2))
    (out/"discovery_pair_manifest.json").write_text(json.dumps({"version":1,"regime":"controlled_8fps",**manifests},indent=2))
    plan={"coarse_layers":list(COARSE),"refine_rule":"discovery-only: contiguous >=2 coarse checkpoints, direction-consistent sufficiency and necessity, CI lower bound > 0","hook_semantics":"decoder block output; final prompt token or selected video-window positions","module_order":["attention_output","mlp_output","block_output"],"statistics":"target-prefix aggregation then trajectory-cluster bootstrap/sign permutation","discovery_pairs":len(manifests["discovery_pairs"]),"validation_pairs":len(manifests["validation_pairs"])}
    (out/"experiment_plan.json").write_text(json.dumps(plan,indent=2))
    print(json.dumps({"out":str(out),"discovery_trajectories":len(split["discovery_trajectories"]),"validation_trajectories":len(split["validation_trajectories"]),"discovery_pairs":manifests["n_discovery_pairs"],"validation_pairs":manifests["n_validation_pairs"],"coarse_layers":list(COARSE)},indent=2))

def run_dry(args):
    out=Path(args.out); split=json.loads((out/"discovery_validation_split.json").read_text()); m=json.loads((out/"discovery_pair_manifest.json").read_text())
    # Two clean forwards (target/hybrid) plus one patched forward in each
    # direction for every coarse checkpoint. Captures are retained only for
    # the current pair and are not written as a full-sequence cache.
    per_pair=2+2*len(COARSE)
    print(json.dumps({"status":"DRY_RUN_PASS","model_forwards":{"discovery_coarse":per_pair*len(m["discovery_pairs"]),"validation_coarse":per_pair*len(m["validation_pairs"]),"per_pair_coarse":per_pair,"module_decomposition":"after validated region only","video_window":"after validated region only"},"split":{"discovery":len(split["discovery_trajectories"]),"validation":len(split["validation_trajectories"])},"leakage":"layer/module/region selection uses discovery only; validation uses all eligible pairs after freeze","cache":"online selected-layer activations; no full-sequence persistent cache"},indent=2))

def run_smoke(args):
    """One real-model hook smoke; deliberately never called by default."""
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from state_rev_input_pipeline import load_model_and_processor, render_inputs, to_device
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    import mg_common as mg
    model, processor=load_model_and_processor(Path(args.model_dir)); model.eval()
    rows=read_rows(Path(args.behavior)); r=next(x for x in rows if x["trajectory_id"]==args.trajectory and x["t"]==args.t)
    clip=sample_clip(Path(args.dataset)/f"{r['trajectory_id']}.mp4",0,int(r["frame_end"]))
    msgs=state_messages(clip,r["initial_state"],r["t"]); inputs,_=render_inputs(processor,msgs,clip,"controlled_8fps")
    dev=next(model.parameters()).device; inp=to_device(inputs,dev); pos=inp["input_ids"].shape[1]-1
    layers=resolve_decoder_layers(model); captured={}
    handles=[]
    for i in (0,35):
        def hook(_m,_i,o,layer=i):
            h=block_output_tensor(o); captured[layer]=h[0,pos,:].detach().float().cpu().clone()
        handles.append(layers[i].register_forward_hook(hook))
    with torch.inference_mode():
        clean=model(**inp,logits_to_keep=1).logits[0,-1].float().cpu()
    for h in handles:h.remove()
    # self patch at the final token must be numerically identical.
    for i in (0,35):
        rep=captured[i]
        def ph(_m,_i,o,rep=rep):
            h=block_output_tensor(o).clone(); h[0,pos,:]=rep.to(h.device,h.dtype)
            return h if hasattr(o,"shape") else (h,)+tuple(o[1:])
        handle=layers[i].register_forward_hook(ph)
        with torch.inference_mode(): patched=model(**inp,logits_to_keep=1).logits[0,-1].float().cpu()
        handle.remove()
        err=float((patched-clean).abs().max())
        if err>1e-4: raise AssertionError(f"self patch layer {i} max logit error {err}")
    # Full-sequence reconstruction at the final block is restricted to this
    # one smoke pair. It verifies block-output hook semantics without making a
    # persistent all-token activation cache.
    manifest=json.loads((Path(args.source)/"pair_manifest.json").read_text())
    pair=next(p for p in manifest["pairs"] if p["target_traj"]==args.trajectory and p["t"]==args.t)
    tc=sample_clip(Path(args.dataset)/f"{pair['target_traj']}.mp4",0,pair["frame_end"])
    sc=sample_clip(Path(args.dataset)/f"{pair['source_traj']}.mp4",0,pair["frame_end"])
    hc,_=mg.transplant_sampled_window(tc,sc,args.t,"controlled_8fps")
    hi,_=render_inputs(processor,state_messages(tc,pair["target_initial"],args.t),hc,"controlled_8fps")
    hi=to_device(hi,dev); captured_h={}
    def capture_h(_m,_i,o): captured_h["x"]=block_output_tensor(o).detach().clone()
    handle=layers[35].register_forward_hook(capture_h)
    with torch.inference_mode(): hout=model(**hi,logits_to_keep=1).logits[0,-1].float().cpu()
    handle.remove()
    captured_t={}
    def capture_t(_m,_i,o): captured_t["x"]=block_output_tensor(o).detach().clone()
    handle=layers[35].register_forward_hook(capture_t)
    with torch.inference_mode(): _=model(**inp,logits_to_keep=1)
    handle.remove()
    def reconstruct(_m,_i,o):
        h=block_output_tensor(o)
        if tuple(h.shape)!=tuple(captured_h["x"].shape): raise AssertionError("full sequence shape mismatch")
        return captured_h["x"] if hasattr(o,"shape") else (captured_h["x"],)+tuple(o[1:])
    handle=layers[35].register_forward_hook(reconstruct)
    with torch.inference_mode(): rlog=model(**inp,logits_to_keep=1).logits[0,-1].float().cpu()
    handle.remove(); err=float((rlog-hout).abs().max())
    if err>2e-3: raise AssertionError(f"hybrid reconstruction max logit error {err}")
    print(json.dumps({"status":"MODEL_SMOKE_PASS","trajectory":args.trajectory,"t":args.t,"layers_tested":[0,35],"last_token":pos,"max_self_patch_logit_error":"<=1e-4","hybrid_reconstruction_max_logit_error":err,"full_sequence_cache":"temporary only; not persisted"},indent=2))

def run_unit(_args):
    class Fake:
        def __init__(self): self.model=type("M",(),{"language_model":type("L",(),{"layers":[object() for _ in range(36)]})()})()
    import torch
    assert len(resolve_decoder_layers(Fake()))==36
    h=torch.zeros(1,5,4); repl=torch.arange(8,dtype=torch.float32).reshape(2,4)
    hook=replace_positions_hook([1,3],repl); out=hook(None,None,h)
    assert torch.equal(out[0,1],repl[0]) and torch.equal(out[0,3],repl[1]) and torch.equal(out[0,0],h[0,0])
    assert_activation_shape(repl,repl,[1,3])
    try: replace_positions_hook([1],repl)(None,None,h); raise AssertionError("bad shape accepted")
    except RuntimeError: pass
    print("HOOK_UNIT_PASS: decoder block output, position replacement, shape guard")

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True)
    for mode in ("prepare","dry-run","unit"):
        p=sub.add_parser(mode); p.add_argument("--out",default=str(ROOT)); p.add_argument("--source",default=str(SRC)); p.add_argument("--behavior",default="outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"); p.add_argument("--seed",type=int,default=20260906); p.add_argument("--discovery-n",type=int,default=30); p.add_argument("--split",default=None,help="frozen split manifest; content-disjoint manifests are asserted")
    p=sub.add_parser("smoke"); p.add_argument("--out",default=str(ROOT)); p.add_argument("--source",default=str(SRC)); p.add_argument("--behavior",default="outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"); p.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct"); p.add_argument("--dataset",default="dataset/vetbench/cup"); p.add_argument("--trajectory",default="cup_001"); p.add_argument("--t",type=int,default=1)
    a=ap.parse_args()
    if a.mode=="prepare": run_prepare(a)
    elif a.mode=="dry-run": run_dry(a)
    elif a.mode=="unit": run_unit(a)
    else: run_smoke(a)
if __name__=="__main__": main()
