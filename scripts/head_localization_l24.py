#!/usr/bin/env python3
"""L24 pre-o_proj attention-head localization.

The intervention point is the input of ``self_attn.o_proj``.  At this point
Qwen's per-head attention outputs have been concatenated but have not yet
been mixed by ``o_proj``.  This module is safe by default: ``audit`` and
``unit`` are CPU/offline; only ``run`` loads the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

LAYER = 24
STATES = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
CONDITIONS = ("target_self", "same_event_source", "matched_history_transplant",
              "source_current_transplant")


def structure_from_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    text = cfg.get("text_config", cfg)
    return {
        "layer": LAYER,
        "num_attention_heads": int(text.get("num_attention_heads", -1)),
        "num_key_value_heads": int(text.get("num_key_value_heads", -1)),
        "head_dim": int(text.get("head_dim", text.get("hidden_size", 0) // max(1, int(text.get("num_attention_heads", 1))))),
        "hidden_size": int(text.get("hidden_size", -1)),
        "num_hidden_layers": int(text.get("num_hidden_layers", -1)),
        "gqa_group_size": int(text.get("num_attention_heads", 0)) // max(1, int(text.get("num_key_value_heads", 1))),
        "intervention_point": "layer[23].self_attn.o_proj input pre-hook",
        "head_concat_shape": "[batch, sequence, num_attention_heads * head_dim]",
        "o_proj_mixes_heads": True,
        "head_indexing": "contiguous [head * head_dim:(head+1)*head_dim]",
    }


def pre_oproj_hook(replacement: np.ndarray | object, heads: list[int], position: int):
    """Create a pre-hook replacing selected concatenated head slices.

    ``replacement`` is [n_heads, head_dim], or a torch tensor with that shape.
    The hook explicitly checks the pre-o_proj tensor shape and never slices the
    post-o_proj hidden state.
    """
    import torch
    repl = replacement if torch.is_tensor(replacement) else torch.as_tensor(replacement)
    heads = list(map(int, heads))

    def hook(_module, inputs):
        if not inputs:
            raise RuntimeError("o_proj pre-hook received no inputs")
        x = inputs[0]
        if x.ndim != 3:
            raise RuntimeError(f"expected o_proj input [B,T,H], got {tuple(x.shape)}")
        width = int(repl.shape[-1])
        if repl.ndim != 2 or repl.shape[0] != len(heads):
            raise RuntimeError("replacement must be [n_heads, head_dim]")
        if x.shape[-1] % width != 0:
            raise RuntimeError("o_proj input width is not divisible by head_dim")
        q = x.clone()
        for i, head in enumerate(heads):
            lo, hi = head * width, (head + 1) * width
            if hi > q.shape[-1]:
                raise RuntimeError(f"head {head} exceeds concat width {q.shape[-1]}")
            q[:, position, lo:hi] = repl[i].to(device=q.device, dtype=q.dtype)
        return (q, *inputs[1:])
    return hook


def all_heads_equal(full: object, heads: int, head_dim: int) -> bool:
    import torch
    x = full if torch.is_tensor(full) else torch.as_tensor(full)
    return x.ndim == 2 and tuple(x.shape) == (heads, head_dim)


def run_unit() -> None:
    import torch
    class Fake:
        def __init__(self): self.hooks = []
        def register_forward_pre_hook(self, h): self.hooks.append(h); return h
    mod = Fake(); x = torch.arange(1 * 4 * 8, dtype=torch.float32).reshape(1, 4, 8)
    src = torch.randn(2, 4)
    h = pre_oproj_hook(src, [0, 1], 3)
    y = h(mod, (x,))[0]
    expected = x.clone(); expected[0, 3, :4] = src[0]; expected[0, 3, 4:8] = src[1]
    assert torch.equal(y, expected)
    # Patching every head with the full source concat is exactly full
    # attention-output replacement before o_proj.
    full = torch.randn(2, 4)
    y2 = pre_oproj_hook(full, [0, 1], 1)(mod, (x,))[0]
    expected2 = x.clone(); expected2[0, 1] = full.reshape(-1)
    assert torch.equal(y2, expected2)
    # Self replacement is bit-identical at the intervention tensor.
    assert torch.equal(pre_oproj_hook(x[0, 2].reshape(2, 4), [0, 1], 2)(mod, (x,))[0], x)
    assert all_heads_equal(full, 2, 4)
    print("HEAD_UNIT_PASS: pre-o_proj concat slicing, all-head equivalence, self patch")


def write_audit(args) -> None:
    cfg = Path(args.model_dir) / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    audit = structure_from_config(cfg)
    if (audit["num_attention_heads"], audit["num_key_value_heads"], audit["head_dim"], audit["num_hidden_layers"]) != (32, 8, 128, 36):
        raise RuntimeError(f"unexpected Qwen3-VL text attention config: {audit}")
    audit["model_dir"] = str(args.model_dir)
    audit["layer_index_zero_based"] = LAYER - 1
    audit["num_heads_expected"] = audit["num_attention_heads"]
    audit["gqa_mapping"] = {str(h): h // audit["gqa_group_size"]
                             for h in range(audit["num_attention_heads"])}
    (out / "head_structure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

def prepare(args) -> None:
    src=Path(args.source); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for name in ("discovery_validation_split.json","discovery_pair_manifest.json"):
        p=src/name
        if not p.exists(): raise FileNotFoundError(p)
        (out/name).write_bytes(p.read_bytes())
    plan={"layer":LAYER,"protocol":"fixed v2 discovery/validation split; no head selection on validation","source":str(src),"conditions":list(CONDITIONS),"all_heads":32,"gqa_group_size":4}
    (out/"experiment_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    print(json.dumps({"status":"PREPARE_PASS","out":str(out),"source":str(src)},indent=2))


def load_runtime():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import circuit_localization as plan
    import mg_common as mg
    from state_rev_input_pipeline import load_model_and_processor, render_inputs, to_device, POS_IDS
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    return plan, mg, load_model_and_processor, render_inputs, to_device, POS_IDS, state_messages, sample_clip


def main_run(args) -> None:
    plan, mg, load_model_and_processor, render_inputs, to_device, POS_IDS, state_messages, sample_clip = load_runtime()
    import torch
    root = Path(args.out); split = json.loads((root / "discovery_validation_split.json").read_text())
    manifest = json.loads((root / "discovery_pair_manifest.json").read_text())
    pairs = manifest["discovery_pairs"] if args.stage == "discovery" else manifest["validation_pairs"]
    pairs = pairs[args.shard_index::args.num_shards]
    model, processor = load_model_and_processor(Path(args.model_dir)); model.eval()
    device = next(model.parameters()).device
    block = plan.resolve_decoder_layers(model)[LAYER - 1]
    oproj = block.self_attn.o_proj
    heads = int(getattr(block.self_attn, "num_heads", 32))
    head_dim = int(getattr(block.self_attn, "head_dim", 128))
    if heads != 32 or head_dim != 128:
        raise RuntimeError(f"unexpected L24 attention shape heads={heads}, head_dim={head_dim}")
    rows = plan.read_rows(Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"))
    base = np.load("outputs/vetbench/mechanism_gate_final/hidden_baseline.npz")
    base_hidden = {k: base[k][-1] for k in base.files}
    from run_circuit_localization import event_decoder
    decoder = event_decoder(base_hidden, rows, split["discovery_trajectories"])

    def render(clip, initial, t):
        return to_device(render_inputs(processor, state_messages(clip, initial, t), clip, "controlled_8fps")[0], device)
    def forward_logits(inp):
        with torch.inference_mode():
            out = model(**inp, logits_to_keep=1)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        return {s: float(lp[i]) for s, i in POS_IDS.items()}
    def capture(inp):
        got = {}
        pos = inp["input_ids"].shape[1] - 1
        def ph(_m, ins): got["x"] = ins[0][0, pos].detach().float().cpu().clone(); return ins
        final={}
        def nh(_m,_i,o): final["x"]=o[0,pos].detach().float().cpu().clone()
        hd = oproj.register_forward_pre_hook(ph)
        hn = model.model.language_model.norm.register_forward_hook(nh)
        try: lp = forward_logits(inp)
        finally: hd.remove(); hn.remove()
        x = got["x"].reshape(heads, head_dim)
        return lp, x, final["x"]
    def patched(inp, replacement, selected):
        pos = inp["input_ids"].shape[1] - 1
        hook = pre_oproj_hook(replacement, selected, pos)
        final={}
        def nh(_m,_i,o): final["x"]=o[0,pos].detach().float().cpu().clone()
        hd = oproj.register_forward_pre_hook(hook)
        hn = model.model.language_model.norm.register_forward_hook(nh)
        try: return forward_logits(inp), final["x"]
        finally: hd.remove(); hn.remove()

    records=[]
    selected_heads=list(range(heads)) if args.heads == "all" else [int(x) for x in args.heads.split(',') if x.strip()]
    if any(h<0 or h>=heads for h in selected_heads): raise ValueError("head index out of range")
    is_joint=args.joint_label != ""
    for p in pairs:
        tc=sample_clip(Path(args.dataset)/f"{p['target_traj']}.mp4",0,p['frame_end'])
        sc=sample_clip(Path(args.dataset)/f"{p['source_traj']}.mp4",0,p['frame_end'])
        same=sample_clip(Path(args.dataset)/f"{p['same_event_traj']}.mp4",0,p['frame_end'])
        if args.condition == "target_self": vc=tc
        elif args.condition == "same_event_source": vc,_=mg.transplant_sampled_window(tc,same,p['t'],"controlled_8fps")
        elif args.condition == "matched_history_transplant": vc,_=mg.transplant_sampled_window(tc,sc,p['t'],"controlled_8fps",destination="history")
        else: vc,_=mg.transplant_sampled_window(tc,sc,p['t'],"controlled_8fps")
        target_inp=render(tc,p['target_initial'],p['t']); variant_inp=render(vc,p['target_initial'],p['t'])
        ti,th,te=capture(target_inp); vi,vh,ve=capture(variant_inp)
        st,cf=p['target_state'],p['counterfactual_state']; third=next(x for x in STATES if x not in {st,cf})
        src_i=EVENTS.index(p['source_event']); tgt_i=EVENTS.index(p['target_event'])
        ep_t=decoder(te.numpy()); ep_v=decoder(ve.numpy())
        patch_heads=selected_heads if is_joint else list(range(heads))
        for head in patch_heads:
            use_heads=selected_heads if is_joint else [head]
            sp,sh=patched(target_inp,vh[use_heads],[*use_heads])
            np_,nh=patched(variant_inp,th[use_heads],[*use_heads])
            for direction,lp,eh in (("sufficiency",sp,sh),("necessity",np_,nh)):
                ep=decoder(eh.numpy())
                records.append({"pair_id":p['pair_id'],"target_prefix":f"{p['target_traj']}_t{p['t']}","target_traj":p['target_traj'],"t":p['t'],"condition":args.condition,"direction":direction,"layer":LAYER,"head":("joint:"+args.joint_label if is_joint else head),"patched_heads":','.join(map(str,use_heads)),"target_state":st,"counterfactual_state":cf,"third_state":third,"logit_cf":lp[cf],"logit_target":lp[st],"logit_third":lp[third],"event_margin":float(ep[src_i]-ep[tgt_i]),"clean_target_cf_margin":float(ti[cf]-ti[st]),"clean_hybrid_cf_margin":float(vi[cf]-vi[st]),"clean_target_event_margin":float(ep_t[src_i]-ep_t[tgt_i]),"clean_hybrid_event_margin":float(ep_v[src_i]-ep_v[tgt_i])})
    out=Path(args.out)/"shards"/args.stage/f"shard_{args.shard_index}"; out.mkdir(parents=True,exist_ok=True)
    stem=f"joint_{args.joint_label}" if is_joint else "head"
    path=out/f"{stem}_{args.condition}_{args.stage}.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps({"stage":args.stage,"condition":args.condition,"shard":args.shard_index,"pairs":len(pairs),"rows":len(records),"output":str(path)},indent=2))


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("audit");p.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct");p.add_argument("--out",default="outputs/vetbench/head_localization_l24_v1")
    p=sub.add_parser("prepare");p.add_argument("--source",default="outputs/vetbench/circuit_localization_v2");p.add_argument("--out",default="outputs/vetbench/head_localization_l24_v1")
    sub.add_parser("unit")
    p=sub.add_parser("run");p.add_argument("--stage",choices=("discovery","validation"),required=True);p.add_argument("--condition",choices=CONDITIONS,default="source_current_transplant");p.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct");p.add_argument("--dataset",default="dataset/vetbench/cup");p.add_argument("--out",default="outputs/vetbench/head_localization_l24_v1");p.add_argument("--shard-index",type=int,default=0);p.add_argument("--num-shards",type=int,default=1);p.add_argument("--heads",default='all');p.add_argument("--joint-label",default='')
    a=ap.parse_args()
    if a.mode=="unit":run_unit()
    elif a.mode=="audit":write_audit(a)
    elif a.mode=="prepare":prepare(a)
    else:main_run(a)
if __name__=="__main__":main()
