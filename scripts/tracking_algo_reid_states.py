#!/usr/bin/env python3
"""Addendum K follow-up (exploratory): WHICH layout does the location code /
native answer reflect under full and steps? For each j = 0..k, S_j is the
position of the ball's cup after swap j (S_0 = init). Held-out-history
decodability of S_j at the last token; native agreement with S_j restricted
to items where S_j differs from the final state."""
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B = Path(__file__).resolve().parents[1] / 'outputs/tracking_algo_v1'
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--layer', type=int, required=True)
a = ap.parse_args()
items = {json.loads(l)['id']: json.loads(l) for l in open(B / 'data_reid2/items.jsonl')}
rows = {json.loads(l)['id']: json.loads(l) for l in open(B / f'eval_reid2/{a.model}/rows_shard0of1.jsonl')}
Z = np.load(B / f'tokenprobe/{a.model}_reid2.npz'); layers = json.loads((B / f'tokenprobe/{a.model}_reid2.json').read_text())['layers']
li = layers.index(a.layer)


def traj(it):
    s, out = it['init'], [it['init']]
    for x, y in it['swaps']:
        s = y if s == x else x if s == y else s
        out.append(s)
    return out


for cond in ('full', 'steps', 'blank'):
    for k in (2, 3, 4):
        ids = [f'{cond}_{j:03d}' for j in range(300) if items[f'{cond}_{j:03d}']['k'] == k]
        T = np.array([traj(items[i]) for i in ids])
        tup = [str((items[i]['init'], items[i]['swaps'])) for i in ids]; u = sorted(set(tup)); g = [u.index(t) for t in tup]
        X = np.stack([Z[i][li, -1] for i in ids]).astype(np.float32)
        pred = np.array([rows[i]['pred'] for i in ids])
        code, nat = [], []
        for j in range(k + 1):
            code.append(round(float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
                                                     X, T[:, j], cv=GroupKFold(5), groups=g).mean()), 2))
            m = T[:, j] != T[:, -1] if j < k else np.ones(len(ids), bool)
            nat.append(round(float((pred[m] == T[m, j]).mean()), 2) if m.sum() else None)
        print(a.model, cond, f'k={k}', 'code S0..Sk', code, '| native=S_j (S_j != final)', nat, flush=True)
