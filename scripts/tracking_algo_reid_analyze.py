#!/usr/bin/env python3
"""Addendum J analysis: native accuracy per condition with paired bootstrap
CI of Cfull - Ctele, and held-out-history last-token location code."""
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B = Path(__file__).resolve().parents[1] / 'outputs/tracking_algo_v1'
FIXED = {'qwen3vl8b': 28, 'qwen35_9b': 32, 'qwen36_27b': 49}
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
ap.add_argument('--name', default='reid', choices=['reid', 'reid2']); a = ap.parse_args()
CONDS = ('Cfull', 'Ctele', 'Ifull') if a.name == 'reid' else ('full', 'steps', 'tele', 'telefinal', 'blank')
REF, CMP = CONDS[0], CONDS[1:]
items = {json.loads(l)['id']: json.loads(l) for l in open(B / f'data_{a.name}/items.jsonl')}
rng = np.random.default_rng(0)


def paired(x, y):
    d = x - y
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)]
    return round(float(d.mean()), 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


out = {}
ev = B / f'eval_{a.name}/{a.model}/rows_shard0of1.jsonl'
if ev.exists():
    rows = {json.loads(l)['id']: json.loads(l) for l in open(ev)}
    corr = {c: np.array([rows[f'{c}_{j:03d}']['pred'] == items[f'{c}_{j:03d}']['labels']['ground_truth'] for j in range(300)], float)
            for c in CONDS}
    out['native'] = {c: round(float(v.mean()), 3) for c, v in corr.items()}
    for c in CMP:
        out['native'][f'{REF}-{c}'] = paired(corr[REF], corr[c])
    out['native_by_k'] = {c: {k: round(float(corr[c][[items[f'{c}_{j:03d}']['k'] == k for j in range(300)]].mean()), 3) for k in (2, 3, 4)} for c in corr}
tp = B / f'tokenprobe/{a.model}_{a.name}.npz'
if tp.exists():
    Z = np.load(tp); layers = json.loads((B / f'tokenprobe/{a.model}_{a.name}.json').read_text())['layers']
    code = {}
    for c in CONDS:
        ids = [f'{c}_{j:03d}' for j in range(300)]
        y = np.array([items[i]['labels']['ground_truth'] for i in ids])
        tup = [str((items[i]['init'], items[i]['swaps'])) for i in ids]; u = sorted(set(tup)); g = [u.index(t) for t in tup]
        code[c] = {}
        for li, L in enumerate(layers):
            X = np.stack([Z[i][li, -1] for i in ids]).astype(np.float32)
            p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), X, y, cv=GroupKFold(5), groups=g)
            code[c][L] = (p == y).astype(float)
    out['code'] = {c: {L: round(float(v.mean()), 3) for L, v in d.items()} for c, d in code.items()}
    Lf = min(layers, key=lambda L: abs(L - FIXED[a.model]))
    out['code_fixed_layer'] = Lf
    for c in CMP:
        out[f'code_{REF}-{c}_fixed'] = paired(code[REF][Lf], code[c][Lf])
print(json.dumps(out, indent=1))
(B / f'{a.name}_{a.model}.json').write_text(json.dumps(out, indent=1))
