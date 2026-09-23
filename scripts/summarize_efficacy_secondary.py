#!/usr/bin/env python3
"""Summarize efficacy-passing prefixes from already-generated v2 CSVs."""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_mechanism_v2 import cluster_summary

root=Path('outputs/vetbench/mechanism_gate_final_v2')
gate=json.loads(Path('outputs/vetbench/mechanism_gate_final/decisive_analysis.json').read_text())
passing={x['target_prefix'] for x in gate['target_prefix_effects'] if x['event_success_rate'] >= .7}
def summarize(path, fields):
 rows=list(csv.DictReader(path.open()))
 out={}
 for protocol in sorted(set(r.get('protocol','native') for r in rows)):
  for subset,pred in [('overall',lambda r:True),('t_ge2',lambda r:int(r['t'])>=2),('t_ge3',lambda r:int(r['t'])>=3)]:
   xs=[r for r in rows if r.get('protocol','native')==protocol and r['target_prefix'] in passing and pred(r)]
   for field in fields:
    by=defaultdict(list)
    for r in xs: by[r['target_prefix']].append(float(r[field]))
    vals=[np.mean(v) for v in by.values()]
    out[f'{protocol}:{subset}:{field}']=cluster_summary(vals,[k.rsplit('_t',1)[0] for k in by])
 return out
result={'n_efficacy_passing_target_prefixes':len(passing),
 'native':summarize(root/'native_counterfactual_results.csv',['native_cf_shift','native_cf_probability_change']),
 'ood':summarize(root/'ood_state_decoder_results.csv',['ood_state_shift','ood_margin_shift'])}
(root/'efficacy_passing_secondary_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
