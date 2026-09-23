#!/usr/bin/env python3
"""Exploratory follow-up to Addendum C (the pre-registered per-condition probe
has n=90 and is underpowered: S1 on opaque3 subsampled to n=90 gives 0.62
vs 0.92 at n=300).

1. Transfer: fit a decoder of "ball position after swap 1" on opaque3 video
   tokens (group right after swap 1, n=300), then apply it, frozen, to the
   horizon videos at the group right after their LAST swap.
2. Pooled CV: all m=1 conditions pooled vs all m>=2 conditions pooled.
"""
from pathlib import Path
import argparse
import json
import math
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'outputs/tracking_algo_v1'


def ap_(s, a, b):
    return b if s == a else a if s == b else s


def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X, y)


def cv(X, y):
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        acc.append((fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    return round(float(np.mean(acc)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    Zo = np.load(B / f'tokenprobe/{a.model}.npz')
    Zh = np.load(B / f'tokenprobe/{a.model}_horizon.npz')
    layers = json.loads((B / f'tokenprobe/{a.model}.json').read_text())['layers']
    lh = json.loads((B / f'tokenprobe/{a.model}_horizon.json').read_text())['layers']
    assert layers == lh
    io = {json.loads(l)['id']: json.loads(l) for l in open(B / 'data/items.jsonl')}
    ih = {json.loads(l)['id']: json.loads(l) for l in open(B / 'data_horizon/items.jsonl')}
    ido = [i for i in Zo.files if io[i]['set'] == 'opaque3']
    yo = np.array([ap_(io[i]['init'], *io[i]['swaps'][0]) for i in ido])
    g1 = math.ceil(3.0 / 0.5)
    conds = sorted({ih[i]['set'] for i in Zh.files})
    out = {}
    for li, L in enumerate(layers):
        dec = fit(np.stack([Zo[i][li, g1] for i in ido]).astype(np.float32), yo)
        r = {}
        for c in conds:
            ids = [i for i in Zh.files if ih[i]['set'] == c]
            g = [min(math.ceil(ih[i]['swap_end_times'][-1] / 0.5), Zh[i].shape[1] - 2) for i in ids]
            X = np.stack([Zh[i][li, gi] for i, gi in zip(ids, g)]).astype(np.float32)
            y = np.array([ih[i]['labels']['ground_truth'] for i in ids])
            r[c] = round(float((dec.predict(X) == y).mean()), 3)
        for name, sel in (('pooled_m1', lambda c: c.endswith('m1') or c.startswith('K1')),
                          ('pooled_m1_after_reveal', lambda c: c in ('K3_m1', 'K4_m1')),
                          ('pooled_m2plus', lambda c: c in ('K3_m2', 'K3_m3', 'K4_m4'))):
            ids = [i for i in Zh.files if sel(ih[i]['set'])]
            g = [min(math.ceil(ih[i]['swap_end_times'][-1] / 0.5), Zh[i].shape[1] - 2) for i in ids]
            X = np.stack([Zh[i][li, gi] for i, gi in zip(ids, g)]).astype(np.float32)
            y = np.array([ih[i]['labels']['ground_truth'] for i in ids])
            r[f'cv_{name}(n={len(ids)})'] = cv(X, y)
        out[f'L{L}'] = r
        print(a.model, f'L{L}', r, flush=True)
    (B / f'horizon_transfer_{a.model}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
