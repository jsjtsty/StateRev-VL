#!/usr/bin/env python3
"""Hidden-state probes on the synthetic G1 set (last input token, all layers).

For opaque3 and transp3 separately: 5-fold CV linear decodability of the
ground-truth final position, the last_touch_init shortcut, the initial
position, the first and the last swapped pair, and the model's own answer.
Chance for each label is reported as its majority rate. Items are
independent videos, so plain stratified-by-nothing KFold is used; transp3
and opaque3 share swap sequences, so they are never mixed within a probe.
"""
from pathlib import Path
import argparse
import glob
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/tracking_algo_v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layers', default='')
    a = ap.parse_args()
    H = {}
    for f in glob.glob(str(BASE / 'eval' / a.model / 'hidden_shard*.npz')):
        H.update(dict(np.load(f)))
    items = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'data/items.jsonl')}
    rows = {(json.loads(l)['id']): json.loads(l) for f in glob.glob(str(BASE / 'eval' / a.model / 'rows_shard*.jsonl'))
            for l in open(f) if json.loads(l)['variant'] == 'init'}
    n_layers = next(iter(H.values())).shape[0]
    layers = [int(x) for x in a.layers.split(',')] if a.layers else sorted(set(np.linspace(0, n_layers - 1, 10).round().astype(int)))
    out = {}
    for s in ('opaque3', 'transp3'):
        ids = [i for i in H if items[i]['set'] == s]
        X = np.stack([H[i] for i in ids]).astype(np.float32)
        lab = pd.DataFrame({
            'ground_truth': [items[i]['labels']['ground_truth'] for i in ids],
            'last_touch_init': [items[i]['labels']['last_touch_init'] for i in ids],
            'init': [items[i]['init'] for i in ids],
            'first_pair': [str(sorted(items[i]['swaps'][0])) for i in ids],
            'last_pair': [str(sorted(items[i]['swaps'][-1])) for i in ids],
            'model_answer': [rows[i]['pred'] if i in rows else -1 for i in ids],
        })
        R = {'n': len(ids), 'majority': {c: round(float(lab[c].value_counts(normalize=True).max()), 3) for c in lab}}
        for L in layers:
            r = {}
            for c in lab:
                y = lab[c].values
                acc = []
                for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[tr, L], y[tr])
                    acc.append((m.predict(X[te, L]) == y[te]).mean())
                r[c] = round(float(np.mean(acc)), 3)
            R[f'L{L}'] = r
            print(a.model, s, f'L{L}', r, flush=True)
        out[s] = R
    (BASE / f'probe_{a.model}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
