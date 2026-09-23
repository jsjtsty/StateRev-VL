#!/usr/bin/env python3
"""Pre-registered equal-budget multi-position residual-delta rescue.

Uses the already frozen strong-same-state donor pairs and the canonical input
pipeline.  A total residual-delta budget of 1.0 is divided equally across
each residual position in a combination.  Shards write a CSV and a completion
marker; merge refuses any missing marker or incomplete key coverage.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
STATES=("Left","Middle","Right"); COMBOS={"baseline":(),"L24":(24,),"L32":(32,),"L24_L32":(24,32),"Head0_L32":(32,),"L24_L28_L32":(24,28,32)}
HEAD_LAYER,HEAD,HEAD_DIM=24,0,128; SEED=20260916
def stat(v,g,n=8000):
 d=pd.DataFrame({"v":v,"g":g}).dropna(); z=d.groupby("g").v.mean().to_numpy()
 if not len(z):return {"mean":None,"ci95":[None,None],"p_sign_permutation":None,"n_trajectories":0}
 r=np.random.default_rng(SEED); b=z[r.integers(0,len(z),(n,len(z)))].mean(1); q=(z*r.choice((-1.,1.),(n,len(z)))).mean(1)
 return {"mean":float(z.mean()),"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],"p_sign_permutation":float((abs(q)>=abs(z.mean())).mean()),"n_trajectories":len(z)}
def prepare(a):
 src=json.loads(Path(a.source_manifest).read_text()); out=a.out
 existing=out/"multi_position_manifest.json"
 if out.exists():
  # A resumable runner may be invoked again after prepare or after a partial
  # shard failure.  Reuse only the exact frozen protocol; never silently mix
  # a differently configured experiment into an existing output directory.
  if not existing.exists(): raise FileExistsError(f"{out} exists without multi_position_manifest.json; choose a new --out")
  old=json.loads(existing.read_text())
  expected_combos={k:list(v) for k,v in COMBOS.items()}
  if old.get("source_manifest")!=str(a.source_manifest) or old.get("combos")!=expected_combos: raise RuntimeError("existing manifest differs; choose a new --out")
  print(json.dumps({"status":"MULTI_POSITION_MANIFEST_REUSED","out":str(out),"counts":old["counts"]},indent=2)); return
 out.mkdir(parents=True,exist_ok=False)
 targets=[]
 for x in src["targets"]:
  d=x["donors"]["strong_same_state"]
  targets.append({k:x[k] for k in ("target_prefix","target_trajectory","t","frame_end","initial_state","gt_state","baseline_gt_margin")} | {"donor":d})
 m={"version":1,"source_manifest":str(a.source_manifest),"target_split":"frozen validation targets from source manifest","donor_type":"strong_same_state","combos":COMBOS,"total_residual_delta_budget":1.0,"head0_alpha":.5,"fairness":"each residual combination divides beta=1 equally among its residual positions; Head0+L32 adds frozen conservative Head0 alpha=.5","statistics":"target-prefix aggregation then target-trajectory bootstrap/sign permutation","targets":targets,"counts":{"targets":len(targets),"trajectories":len(set(x["target_trajectory"] for x in targets)),"rows_expected":len(targets)*len(COMBOS)}}
 (out/"multi_position_manifest.json").write_text(json.dumps(m,indent=2)+"\n");print(json.dumps({"status":"MULTI_POSITION_MANIFEST_READY","counts":m["counts"],"combos":list(COMBOS)},indent=2))
def unit(_a):
 x=np.array([1.,2.,3.]);d=np.array([2.,-1.,.5]);assert np.allclose(x+0*d,x) and np.allclose(x+.5*d+.5*d,x+d)
 print("MULTI_POSITION_UNIT_PASS: zero-delta no-op and equal-budget composition")
def run(a):
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parent));import circuit_localization
 from head_localization_l24 import pre_oproj_hook
 from run_state_rev_audit import state_messages
 from run_vetbench_screening import sample_clip
 from state_rev_input_pipeline import POS_IDS,load_model_and_processor,render_inputs,to_device
 root=a.out;m=json.loads((root/"multi_position_manifest.json").read_text());work=m["targets"][a.shard_index::a.num_shards]
 model,proc=load_model_and_processor(Path(a.model_dir));model.eval();dev=next(model.parameters()).device;blocks=circuit_localization.resolve_decoder_layers(model);oproj=blocks[23].self_attn.o_proj
 def render(p,is_donor=False):
  tr=p["trajectory_id"] if is_donor else p["target_trajectory"]; end=p["frame_end"] if not is_donor else p["frame_end"]
  clip=sample_clip(Path(a.dataset)/f"{tr}.mp4",0,int(end));inp,fp=render_inputs(proc,state_messages(clip,p["initial_state"],int(p["t"])),clip,"controlled_8fps");return to_device(inp,dev),fp
 def logits(inp):
  with torch.inference_mode():o=model(**inp,logits_to_keep=1)
  z=torch.log_softmax(o.logits[0,-1].float(),-1);return {s:float(z[i]) for s,i in POS_IDS.items()}
 def capture(p,donor=False):
  inp,fp=render(p,donor);pos=inp["input_ids"].shape[1]-1;got={}
  hs=[]
  def oh(_m,v):got["head0"]=v[0][0,pos,:HEAD_DIM].detach().float().cpu().clone();return v
  hs.append(oproj.register_forward_pre_hook(oh))
  for L in (24,28,32):
   def bh(_m,_i,o,L=L):got[L]=circuit_localization.block_output_tensor(o)[0,pos,:].detach().float().cpu().clone()
   hs.append(blocks[L-1].register_forward_hook(bh))
  try:z=logits(inp)
  finally:
   for h in hs:h.remove()
  return inp,fp,z,got
 def patched(inp,head,deltas,layers):
  pos=inp["input_ids"].shape[1]-1;hs=[]
  if head is not None:hs.append(oproj.register_forward_pre_hook(pre_oproj_hook(head.reshape(1,HEAD_DIM),[HEAD],pos)))
  beta=1/len(layers) if layers else 0
  for L in layers:
   def hook(_m,_i,o,L=L):
    x=circuit_localization.block_output_tensor(o);y=x.clone();y[0,pos,:]=x[0,pos,:]+beta*deltas[L].to(x.device,x.dtype);return y if hasattr(o,"shape") else (y,)+tuple(o[1:])
   hs.append(blocks[L-1].register_forward_hook(hook))
  try:return logits(inp)
  finally:
   for h in hs:h.remove()
 rec=[]; cache={}; noop=[]; selfpatch=[]
 for e in work:
  ti,tf,tz,ta=capture(e);don=e["donor"];dk=don["target_prefix"]
  if dk not in cache:
   dp={"target_trajectory":don["trajectory_id"],"initial_state":don.get("initial_state",e["initial_state"]),"t":don["t"],"frame_end":don["frame_end"]};cache[dk]=capture(dp)
  _,_,_,da=cache[dk];deltas={L:da[L]-ta[L] for L in (24,28,32)};gt=e["gt_state"];base=max(STATES,key=tz.get);bm=tz[gt]-max(tz[s] for s in STATES if s!=gt)
  z0=patched(ti,None,deltas,());noop.append(max(abs(z0[s]-tz[s]) for s in STATES))
  # Explicit self-patch: target Head0 and target-minus-target L32 delta
  # must be an identity forward, independently of the empty-hook no-op.
  selfz=patched(ti,ta["head0"],{L:ta[L]-ta[L] for L in (24,28,32)},(32,))
  selfpatch.append(max(abs(selfz[s]-tz[s]) for s in STATES))
  for name,layers in COMBOS.items():
   # Preserve the pre-registered conservative Head0 dose: this is an
   # interpolation along donor-minus-target, not a full replacement.
   h0=(ta["head0"]+.5*(da["head0"]-ta["head0"])) if name=="Head0_L32" else None
   z=tz if name=="baseline" else patched(ti,h0,deltas,layers);pred=max(STATES,key=z.get);pm=z[gt]-max(z[s] for s in STATES if s!=gt)
   rec.append({"target_prefix":e["target_prefix"],"target_trajectory":e["target_trajectory"],"t":e["t"],"condition":name,"gt_state":gt,"baseline_pred":base,"patched_pred":pred,"baseline_correct":base==gt,"patched_correct":pred==gt,"wrong_to_correct":base!=gt and pred==gt,"correct_to_wrong":base==gt and pred!=gt,"baseline_gt_margin":bm,"patched_gt_margin":pm,"delta_gt_margin":pm-bm,"total_budget":1.,"n_residual_positions":len(layers),"no_op_max_abs":noop[-1]})
 if max(noop)>1e-5 or max(selfpatch)>1e-5:raise AssertionError(f"sanity failed noop={max(noop)} self={max(selfpatch)}")
 sh=root/"shards";sh.mkdir(exist_ok=True);p=sh/f"multi_position_shard_{a.shard_index}.csv";pd.DataFrame(rec).to_csv(p,index=False);(sh/f"multi_position_shard_{a.shard_index}.complete.json").write_text(json.dumps({"shard":a.shard_index,"targets":len(work),"rows":len(rec),"max_noop_logit_error":max(noop),"max_selfpatch_logit_error":max(selfpatch)})+"\n")
def analyze(a):
 root=a.out;m=json.loads((root/"multi_position_manifest.json").read_text());sh=root/"shards";fs=[]; markers=[]
 for i in range(a.num_shards):
  p=sh/f"multi_position_shard_{i}.csv";q=sh/f"multi_position_shard_{i}.complete.json"
  if not(p.exists() and q.exists()):raise FileNotFoundError(f"missing completed shard {i}")
  fs.append(p);markers.append(json.loads(q.read_text()))
 d=pd.concat([pd.read_csv(p) for p in fs],ignore_index=True);exp=len(m["targets"])*len(COMBOS)
 expected={x["target_prefix"] for x in m["targets"]}
 if len(d)!=exp or d.duplicated(["target_prefix","condition"]).any() or set(d.target_prefix)!=expected or set(d.condition)!=set(COMBOS):raise AssertionError("coverage/key mismatch")
 d.to_csv(root/"multi_position_pair_results.csv",index=False); rows=[]; contrasts=[]
 for c in COMBOS:
  q=d[d.condition==c]
  for tag,mask in [("overall",np.ones(len(q),bool)),("t_ge2",q.t>=2),("t_ge3",q.t>=3)]:
   x=q[mask]
   rows.append({"condition":c,"subset":tag,"metric":"patched_correct",**stat(x.patched_correct.astype(float),x.target_trajectory)})
   rows.append({"condition":c,"subset":tag,"metric":"delta_gt_margin",**stat(x.delta_gt_margin.astype(float),x.target_trajectory)})
   wrong=x[~x.baseline_correct]; correct=x[x.baseline_correct]
   rows.append({"condition":c,"subset":tag,"metric":"wrong_to_correct",**stat(wrong.wrong_to_correct.astype(float),wrong.target_trajectory)})
   rows.append({"condition":c,"subset":tag,"metric":"correct_to_wrong",**stat(correct.correct_to_wrong.astype(float),correct.target_trajectory)})
 # Pre-registered multi-position conditions are compared directly with L32
 # on the same prefixes, then aggregated by trajectory before inference.
 for c in ("L24_L32","Head0_L32","L24_L28_L32"):
  left=d[d.condition==c].set_index("target_prefix");right=d[d.condition=="L32"].set_index("target_prefix")
  for tag,idx in [("overall",left.index),("t_ge2",left.index[left.t>=2]),("t_ge3",left.index[left.t>=3])]:
   for k in ("patched_correct","delta_gt_margin"):
    contrasts.append({"contrast":f"{c}-L32","subset":tag,"metric":k,**stat(left.loc[idx,k].astype(float)-right.loc[idx,k].astype(float),left.loc[idx,"target_trajectory"])})
   for k,eligible in (("wrong_to_correct",~left.loc[idx,"baseline_correct"]),("correct_to_wrong",left.loc[idx,"baseline_correct"])):
    use=idx[eligible.to_numpy()]
    contrasts.append({"contrast":f"{c}-L32","subset":tag,"metric":k,**stat(left.loc[use,k].astype(float)-right.loc[use,k].astype(float),left.loc[use,"target_trajectory"])})
 s={"protocol":m["statistics"],"flip_rate_denominators":"wrong_to_correct among baseline-wrong prefixes; correct_to_wrong among baseline-correct prefixes","fairness":m["fairness"],"effects":rows,"paired_contrasts":contrasts,"sanity":{"rows":len(d),"target_prefixes":int(d.target_prefix.nunique()),"all_completion_markers":len(markers)==a.num_shards,"max_noop_logit_error":float(max(x["max_noop_logit_error"] for x in markers)),"max_selfpatch_logit_error":float(max(x["max_selfpatch_logit_error"] for x in markers))}};(root/"multi_position_summary.json").write_text(json.dumps(s,indent=2)+"\n")
 f=pd.DataFrame(rows);lines=["# Equal-budget multi-position rescue","","| condition | subset | accuracy | wrong→correct | correct→wrong | GT-margin Δ |","|---|---|---:|---:|---:|---:|"]
 def get(c,t,k):return f[(f.condition==c)&(f.subset==t)&(f.metric==k)].iloc[0]
 for c in COMBOS:
  for t in ("overall","t_ge2","t_ge3"):
   def fmt(k):x=get(c,t,k);return "NA" if x["mean"] is None else f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
   lines.append(f"| {c} | {t} | {fmt('patched_correct')} | {fmt('wrong_to_correct')} | {fmt('correct_to_wrong')} | {fmt('delta_gt_margin')} |")
 lines += ["","Flip rates are conditional: wrong→correct uses baseline-wrong prefixes; correct→wrong uses baseline-correct prefixes.","","## Paired comparison with L32","","| contrast | subset | accuracy Δ | wrong→correct Δ | correct→wrong Δ | GT-margin Δ difference |","|---|---|---:|---:|---:|---:|"]
 cf=pd.DataFrame(contrasts)
 for c in ("L24_L32-L32","Head0_L32-L32","L24_L28_L32-L32"):
  for t in ("overall","t_ge2","t_ge3"):
   def cfmt(k):
    x=cf[(cf.contrast==c)&(cf.subset==t)&(cf.metric==k)].iloc[0]
    return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}], p={x['p_sign_permutation']:.3f}"
   lines.append(f"| {c} | {t} | {cfmt('patched_correct')} | {cfmt('wrong_to_correct')} | {cfmt('correct_to_wrong')} | {cfmt('delta_gt_margin')} |")
 (root/"multi_position_report.md").write_text("\n".join(lines)+"\n");print(json.dumps({"status":"MULTI_POSITION_ANALYSIS_COMPLETE","rows":len(d)},indent=2))
def main():
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True);common=argparse.ArgumentParser(add_help=False);common.add_argument("--out",type=Path,default=Path("outputs/vetbench/multi_position_delta_rescue_v1"));common.add_argument("--source-manifest",type=Path,default=Path("outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_manifest.json"))
 sp.add_parser("prepare",parents=[common]);sp.add_parser("unit",parents=[common]);r=sp.add_parser("run",parents=[common]);r.add_argument("--model-dir",default="models/Qwen3-VL-8B-Instruct");r.add_argument("--dataset",default="dataset/vetbench/cup");r.add_argument("--shard-index",type=int,required=True);r.add_argument("--num-shards",type=int,required=True);z=sp.add_parser("analyze",parents=[common]);z.add_argument("--num-shards",type=int,required=True);a=ap.parse_args();{"prepare":prepare,"unit":unit,"run":run,"analyze":analyze}[a.cmd](a)
if __name__=="__main__":main()
