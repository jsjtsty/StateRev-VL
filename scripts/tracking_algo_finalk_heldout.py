#!/usr/bin/env python3
"""Addendum I control: is the final state decoded as a STATE (shared code that
generalizes to unseen swap histories) or as a lookup over history tuples?
Compares random 5-fold CV with GroupKFold where the groups are full
(init, pair_1..pair_k) tuples (test histories never seen in training).
Also reports a conjunctive symbolic baseline (one-hot of the full tuple)."""
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B = Path(__file__).resolve().parents[1] / 'outputs/tracking_algo_v1'
PAIRS = [(0, 1), (1, 2), (0, 2)]
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--layers', default='')
a = ap.parse_args()
Z = np.load(B / f'tokenprobe/{a.model}_finalk.npz')
layers = json.loads((B / f'tokenprobe/{a.model}_finalk.json').read_text())['layers']
use = [int(x) for x in a.layers.split(',')] if a.layers else layers
items = {json.loads(l)['id']: json.loads(l) for l in open(B / 'data_finalk/items.jsonl')}
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
out = {}
for k in (1, 2, 3, 4):
    ids = [i for i in Z.files if items[i]['k'] == k]
    y = np.array([items[i]['labels']['ground_truth'] for i in ids])
    tup = [(items[i]['init'],) + tuple(PAIRS.index(tuple(sorted(s))) for s in items[i]['swaps']) for i in ids]
    uniq = sorted(set(tup)); g = np.array([uniq.index(t) for t in tup])
    oh = np.eye(len(uniq))[g]
    rnd = StratifiedKFold(5, shuffle=True, random_state=0)
    grp = GroupKFold(5)
    r = {'n': len(ids), 'n_tuples': len(uniq),
         'sym_tuple_random': round(float(cross_val_score(LogisticRegression(max_iter=3000), oh, y, cv=rnd).mean()), 3)}
    for L in use:
        li = layers.index(L)
        X = np.stack([Z[i][li, -1] for i in ids]).astype(np.float32)
        r[f'L{L}_random'] = round(float(cross_val_score(mk(), X, y, cv=rnd).mean()), 3)
        r[f'L{L}_heldout_hist'] = round(float(cross_val_score(mk(), X, y, cv=grp, groups=g).mean()), 3)
    out[f'k{k}'] = r
    print(a.model, f'k={k}', r, flush=True)
(B / f'finalk_heldout_{a.model}.json').write_text(json.dumps(out, indent=1))
