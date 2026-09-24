#!/usr/bin/env python3
"""Addendum I analysis: final-state decodability vs k (last token and the
video-token group right after the last swap), 5-fold CV per k and layer."""
from pathlib import Path
import argparse
import json
import math
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
B = Path(__file__).resolve().parents[1] / 'outputs/tracking_algo_v1'
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); a = ap.parse_args()
Z = np.load(B / f'tokenprobe/{a.model}_finalk.npz')
layers = json.loads((B / f'tokenprobe/{a.model}_finalk.json').read_text())['layers']
items = {json.loads(l)['id']: json.loads(l) for l in open(B / 'data_finalk/items.jsonl')}
cv = lambda X, y: round(float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
                                              X.astype(np.float32), y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()), 3)
out = {}
for k in (1, 2, 3, 4):
    ids = [i for i in Z.files if items[i]['k'] == k]
    y = np.array([items[i]['labels']['ground_truth'] for i in ids])
    r = {'n': len(ids), 'majority': round(float(np.bincount(y).max() / len(y)), 3)}
    for li, L in enumerate(layers):
        g = [min(math.ceil(items[i]['swap_end_times'][-1] / 0.5), Z[i].shape[1] - 2) for i in ids]
        r[f'last_L{L}'] = cv(np.stack([Z[i][li, -1] for i in ids]), y)
        r[f'after_L{L}'] = cv(np.stack([Z[i][li, gi] for i, gi in zip(ids, g)]), y)
    out[f'k{k}'] = r
    print(a.model, f'k={k}', {kk: v for kk, v in r.items() if kk.startswith(('n', 'maj', 'last'))}, flush=True)
    print(a.model, f'k={k}', {kk: v for kk, v in r.items() if kk.startswith('after')}, flush=True)
(B / f'finalk_{a.model}.json').write_text(json.dumps(out, indent=1))
