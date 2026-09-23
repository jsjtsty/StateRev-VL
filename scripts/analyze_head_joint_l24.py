#!/usr/bin/env python3
"""Offline analysis of frozen candidate-head joint interventions."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

CONDS=("target_self","same_event_source","matched_history_transplant","source_current_transplant")
SUBS={"overall":lambda r:True,"t1":lambda r:int(r['t'])==1,"t_ge2":lambda r:int(r['t'])>=2,"t_ge3":lambda r:int(r['t'])>=3}

def rows(root,stage,label,condition):
 fs=sorted((root/'shards'/stage).glob(f'shard_*/joint_{label}_{condition}_{stage}.csv'))
 if not fs: return []
 out=[]
 for f in fs: out.extend(csv.DictReader(f.open(newline='')))
 return out

def effect(r,kind):
 if kind=='event': return float(r['event_margin'])-(float(r['clean_target_event_margin']) if r['direction']=='sufficiency' else float(r['clean_hybrid_event_margin']))
 return float(r['logit_cf'])-float(r['logit_target'])-(float(r['clean_target_cf_margin']) if r['direction']=='sufficiency' else float(r['clean_hybrid_cf_margin']))

def stat(rr,kind):
 p=defaultdict(list)
 for r in rr:p[(r['target_prefix'],r['target_traj'])].append(effect(r,kind))
 t=defaultdict(list)
 for (_p,tr),x in p.items():t[tr].append(float(np.mean(x)))
 v=np.asarray([np.mean(x) for x in t.values()],float)
 if not len(v):return {'mean':None,'ci95':[None,None],'p_sign_permutation':None,'n_trajectories':0,'n_target_prefixes':0}
 rng=np.random.default_rng(20260907); b=rng.choice(v,(10000,len(v)),True).mean(1); n=(v[None,:]*rng.choice([-1.,1.],(10000,len(v)))).mean(1)
 return {'mean':float(v.mean()),'ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))],'p_sign_permutation':float(np.mean(np.abs(n)>=abs(v.mean()))),'n_trajectories':len(v),'n_target_prefixes':len(p)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('outputs/vetbench/head_localization_l24_v1'));a=ap.parse_args()
 cand=json.loads((a.root/'candidate_heads.json').read_text())
 groups={'event_routing':cand.get('event_routing_heads',[]),'event_to_state':cand.get('event_to_state_heads',[])}
 out=[]
 for label,heads in groups.items():
  if not heads: continue
  for stage in ('discovery','validation'):
   for c in CONDS:
    rr=rows(a.root,stage,label,c)
    for direction in ('sufficiency','necessity'):
     for subset,keep in SUBS.items():
      q=[r for r in rr if r['direction']==direction and keep(r)]
      for endpoint in ('event','native_state'):
       out.append({'group':label,'heads':','.join(map(str,heads)),'stage':stage,'condition':c,'direction':direction,'subset':subset,'endpoint':endpoint,**stat(q,endpoint)})
 a.root.mkdir(parents=True,exist_ok=True)
 path=a.root/'joint_head_results.csv'
 if out:
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 summary={'protocol':'frozen discovery-selected head groups; prefix aggregation then trajectory-cluster bootstrap/sign permutation','groups':groups,'rows':len(out),'output':str(path)}
 (a.root/'joint_head_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (a.root/'joint_head_report.md').write_text('# L24 joint-head analysis\n\n'+json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
