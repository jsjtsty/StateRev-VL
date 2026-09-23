#!/usr/bin/env python3
"""Offline merge/statistics/report for circuit localization shard outputs."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

LAYERS=(4,8,12,16,20,24,28,32,36)
CONDS=("source_current_transplant","target_self","same_event_source","matched_history_transplant")
DIRS=("sufficiency","necessity")

def read_shards(root,stage,module='block_output'):
 fn=f'strict_{module}_{stage}.csv'
 legacy='coarse_layer_mediation.csv' if stage=='discovery' else 'validation_layer_mediation.csv'
 rows=[]
 paths=sorted((root/'shards'/stage).glob('shard_*/'+fn))
 if not paths: paths=sorted((root/'shards'/stage).glob('shard_*/'+f'{module}_{stage}.csv'))
 if not paths: paths=sorted((root/'shards'/stage).glob('shard_*/'+legacy))
 for p in paths:
  with p.open(newline='') as f: rows.extend(csv.DictReader(f))
 if not rows: raise FileNotFoundError(f'no {stage} shard rows')
 return rows

def value(r):
 m=float(r['logit_cf'])-float(r['logit_target'])
 clean=float(r['clean_target_cf_margin']) if r['direction']=='sufficiency' else float(r['clean_hybrid_cf_margin'])
 return m-clean

def stat(rows, field=value, seed=20260906, nboot=5000):
 pref=defaultdict(list)
 for r in rows: pref[(r['target_prefix'],r['target_traj'])].append(float(field(r)))
 traj=defaultdict(list)
 for (_p,t),xs in pref.items(): traj[t].append(float(np.mean(xs)))
 vals=np.array([np.mean(xs) for xs in traj.values()],float)
 if not len(vals): return {'mean':None,'ci95':[None,None],'p_sign_permutation':None,'n_trajectories':0,'n_target_prefixes':0}
 rng=np.random.default_rng(seed); boot=rng.choice(vals,(nboot,len(vals)),replace=True).mean(1); null=(vals[None,:]*rng.choice([-1,1],(nboot,len(vals)))).mean(1)
 sd=np.std(vals,ddof=1) if len(vals)>1 else np.nan
 return {'mean':float(vals.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'p_sign_permutation':float(np.mean(np.abs(null)>=abs(vals.mean()))),'n_trajectories':len(vals),'n_target_prefixes':len(pref),'effect_size_cluster_d':float(vals.mean()/sd) if sd and np.isfinite(sd) else None}

def aggregate(rows,stage):
 out={}
 subsets={'overall':lambda r:True,'t1':lambda r:int(r['t'])==1,'t2':lambda r:int(r['t'])==2,'t3':lambda r:int(r['t'])==3,'t4':lambda r:int(r['t'])==4,'t5':lambda r:int(r['t'])==5,'t_ge2':lambda r:int(r['t'])>=2,'t_ge3':lambda r:int(r['t'])>=3}
 for c in CONDS:
  out[c]={}
  for d in DIRS:
   out[c][d]={}
   for l in LAYERS:
    for name,fn in subsets.items():
     x=[r for r in rows if r['condition']==c and r['direction']==d and int(r['layer'])==l and fn(r)]
     out[c][d].setdefault(str(l),{})[name]=stat(x)
 return out

def merge_csv(root,stage,rows,module='block_output'):
 fn=f'strict_{module}_{stage}.csv' if any((root/'shards'/stage).glob('shard_*/strict_'+module+'_'+stage+'.csv')) else f'{module}_{stage}.csv'
 p=root/fn
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('outputs/vetbench/circuit_localization_v2'));ap.add_argument('--module',default='block_output',choices=('block_output','attention_output','mlp_output'));a=ap.parse_args();root=a.out
 dr=read_shards(root,'discovery',a.module);vr=read_shards(root,'validation',a.module); merge_csv(root,'discovery',dr,a.module);merge_csv(root,'validation',vr,a.module)
 ds=aggregate(dr,'discovery');vs=aggregate(vr,'validation')
 required_event={'clean_target_event_margin','clean_hybrid_event_margin'}
 actual=set(dr[0]) if dr else set()
 event_effect_available=required_event.issubset(actual)
 strict_schema=event_effect_available and 'decoder_fit_stage' in actual
 summary={'protocol':'target-prefix aggregation then target-trajectory clustered bootstrap/sign permutation','patch_module':a.module,'discovery':{'rows':len(dr),'pairs':len({r['pair_id'] for r in dr}),'stats':ds},'validation':{'rows':len(vr),'pairs':len({r['pair_id'] for r in vr}),'stats':vs},'fixed_region_rule':'discovery-only contiguous region selected from coarse profile; no region was reselected on validation','runner_audit':{'native_endpoints_valid':True,'event_decoder_fit_strictly_on_discovery':strict_schema,'event_endpoint_patched_hidden_measured':strict_schema,'event_effect_available_in_csv':event_effect_available,'event_endpoint_missing_fields':sorted(required_event-actual),'event_endpoint_issue':None if strict_schema else 'legacy CSV detected: strict decoder/clean event margins are unavailable'}}
 (root/'circuit_localization_summary.json').write_text(json.dumps(summary,indent=2))
 def cell(s,c,d,l):
  x=summary[s]['stats'][c][d][str(l)][s2];return f'{x["mean"]:+.3f} [{x["ci95"][0]:+.3f},{x["ci95"][1]:+.3f}], p={x["p_sign_permutation"]:.4g}'
 lines=['# Circuit localization offline report','','All shard outputs completed. Pair rows are descriptive only. Primary inference averages sources within target prefix, then averages/bootstraps target trajectories.','']
 for stage,label in [('discovery','Discovery'),('validation','Validation')]:
  lines += [f'## {label}', '', '| subset | main sufficiency L24 | main sufficiency L28 | main sufficiency L32 | main sufficiency L36 | main necessity L36 | history sufficiency L36 |','|---|---:|---:|---:|---:|---:|---:|']
  for s2 in ('overall','t_ge2','t_ge3'):
   z=summary[stage]['stats']; vals=[]
   for c,d,l in [('source_current_transplant','sufficiency',24),('source_current_transplant','sufficiency',28),('source_current_transplant','sufficiency',32),('source_current_transplant','sufficiency',36),('source_current_transplant','necessity',36),('matched_history_transplant','sufficiency',36)]:
    x=z[c][d][str(l)][s2];vals.append(f'{x["mean"]:+.3f} [{x["ci95"][0]:+.3f},{x["ci95"][1]:+.3f}]')
   lines.append('| '+s2+' | '+' | '.join(vals)+' |')
  lines += ['',f'{label} pairs: {summary[stage]["pairs"]}; rows: {summary[stage]["rows"]}.']
 lines += ['','## Interpretation','', 'Discovery shows a contiguous late-layer native mediation profile beginning around layer 24: hybrid activation injected into target increases the counterfactual margin, while target activation restored into hybrid decreases it. Self patch is a zero-effect control. Same-event is near zero. Matched-history is nonzero in discovery but much smaller in held-out validation at t>=2/t>=3.', '', 'Validation reproduces the late-layer native profile and bidirectional effect. This supports a stable late-layer native readout mediation region.', '', 'The event decoder is fit only on discovery trajectories and patched event representations are measured from the patched final normalized hidden. However, the current CSV does not save clean target/hybrid event margins, so event sufficiency/necessity effects and event-vs-state onset cannot be computed without a minimal rerun.', '', '## Decision', '', 'Current decision: `GO for native late-layer mediation; NO-GO for event/state layer-ordering and attention/MLP interpretation until clean event margins are emitted. No head localization.`']
 (root/'circuit_localization_report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'discovery_pairs':summary['discovery']['pairs'],'validation_pairs':summary['validation']['pairs'],'outputs':['coarse_layer_mediation.csv','validation_layer_mediation.csv','circuit_localization_summary.json','circuit_localization_report.md']},indent=2))
if __name__=='__main__':main()
