#!/usr/bin/env python3
"""Offline L24 head discovery/validation analysis.

Selection is discovery-only.  Effects are aggregated within target prefix,
then clustered by target trajectory.  Pair counts are never used as n.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

CONDS=("source_current_transplant","target_self","same_event_source","matched_history_transplant")
DIRS=("sufficiency","necessity")
SUBS={"overall":lambda r:True,"t1":lambda r:int(r['t'])==1,"t_ge2":lambda r:int(r['t'])>=2,"t_ge3":lambda r:int(r['t'])>=3}

def load(root,stage,condition,required=True):
 fs=sorted((root/'shards'/stage).glob(f'shard_*/head_{condition}_{stage}.csv'))
 if not fs:
  if required: raise FileNotFoundError(f'missing head_{condition}_{stage}.csv')
  return []
 rows=[]
 for f in fs: rows.extend(csv.DictReader(f.open(newline='')))
 return rows

def endpoint(r,kind):
 if kind=='event': return float(r['event_margin'])-(float(r['clean_target_event_margin']) if r['direction']=='sufficiency' else float(r['clean_hybrid_event_margin']))
 return float(r['logit_cf'])-float(r['logit_target'])-(float(r['clean_target_cf_margin']) if r['direction']=='sufficiency' else float(r['clean_hybrid_cf_margin']))

def stat(rows,kind):
 p=defaultdict(list)
 for r in rows:p[(r['target_prefix'],r['target_traj'])].append(endpoint(r,kind))
 t=defaultdict(list)
 for (_p,tr),x in p.items():t[tr].append(float(np.mean(x)))
 v=np.asarray([np.mean(x) for x in t.values()])
 if not len(v):return {'mean':None,'ci95':[None,None],'p_sign_permutation':None,'n_trajectories':0,'n_target_prefixes':0}
 rng=np.random.default_rng(20260907); b=rng.choice(v,(10000,len(v)),True).mean(1); null=(v[None,:]*rng.choice([-1.,1.],(10000,len(v)))).mean(1)
 return {'mean':float(v.mean()),'ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))],'p_sign_permutation':float(np.mean(np.abs(null)>=abs(v.mean()))),'n_trajectories':len(v),'n_target_prefixes':len(p)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('outputs/vetbench/head_localization_l24_v1'));ap.add_argument('--stage',choices=('discovery','all'),default='all');a=ap.parse_args(); a.root.mkdir(parents=True,exist_ok=True)
 disc={c:load(a.root,'discovery',c) for c in CONDS}
 val={c:load(a.root,'validation',c,required=(a.stage=='all')) for c in CONDS}
 out=[]
 stages=(('discovery',disc),) if a.stage=='discovery' else (('discovery',disc),('validation',val))
 for stage,data in stages:
  for c,rows in data.items():
   for h in range(32):
    for d in DIRS:
     for sub,keep in SUBS.items():
      rr=[r for r in rows if int(r['head'])==h and r['direction']==d and keep(r)]
      for k in ('event','native_state'):out.append({'stage':stage,'condition':c,'head':h,'direction':d,'subset':sub,'endpoint':k,**stat(rr,k)})
 with (a.root/'head_effects.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 for stage,name in (('discovery','discovery_head_results.csv'),('validation','validation_head_results.csv')):
  sub=[r for r in out if r['stage']==stage]
  if not sub: continue
  with (a.root/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(sub[0]));w.writeheader();w.writerows(sub)
 # preregistered discovery classification: positive sufficiency CI and
 # negative necessity CI at t>=2 for event/state.  Event-routing allows event
 # only; event-to-state requires both endpoints.
 cand={'event_routing':[],'event_to_state':[]}
 def get(h,k,d):return next(x for x in out if x['stage']=='discovery' and x['condition']=='source_current_transplant' and int(x['head'])==h and x['endpoint']==k and x['direction']==d and x['subset']=='t_ge2')
 for h in range(32):
  e=get(h,'event','sufficiency'); en=get(h,'event','necessity'); s=get(h,'native_state','sufficiency'); sn=get(h,'native_state','necessity')
  ep=float(e['ci95'][0])>0 and float(en['ci95'][1])<0
  sp=float(s['ci95'][0])>0 and float(sn['ci95'][1])<0
  if ep and not sp:cand['event_routing'].append(h)
  if ep and sp:cand['event_to_state'].append(h)
 (a.root/'candidate_heads.json').write_text(json.dumps({'selection':'discovery only; t>=2; sufficiency/necessity CI sign rule','event_routing_heads':cand['event_routing'],'event_to_state_heads':cand['event_to_state']},indent=2)+'\n')
 summary={'protocol':'target-prefix aggregation then target-trajectory clustered bootstrap/sign permutation','analysis_stage':a.stage,'discovery_pairs':len({r['pair_id'] for r in disc['source_current_transplant']}),'validation_pairs':len({r['pair_id'] for r in val['source_current_transplant']}),'candidate_heads':cand,'note':'joint head analysis requires a follow-up run with frozen candidate groups'}
 (a.root/'head_localization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 try:
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(10,4))
  for kind,color in (('event','tab:blue'),('native_state','tab:orange')):
   y=[]
   for h in range(32): y.append(float(get(h,kind,'sufficiency')['mean']))
   ax.plot(range(32),y,'o-',label=kind,color=color)
  ax.axhline(0,color='black',linewidth=.7); ax.set_xlabel('L24 attention head'); ax.set_ylabel('discovery t>=2 effect'); ax.legend(); fig.tight_layout(); fig.savefig(a.root/'head_effects.png',dpi=160); plt.close(fig)
 except Exception:
  pass
 report=['# L24 head localization','',f"Event-routing candidates: `{cand['event_routing']}`.",f"Event-to-state candidates: `{cand['event_to_state']}`.",'','Selection uses discovery only; validation is descriptive until candidates are frozen.','Joint candidate-group intervention is a separate follow-up run.']
 (a.root/'head_localization_report.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
