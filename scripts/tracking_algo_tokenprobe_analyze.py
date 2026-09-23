#!/usr/bin/env python3
"""Where along the video is the latent state carried?

For each layer and each step j, decode S_j (ball position after swap j) from
the pooled video tokens of the first temporal group that starts after swap j
has finished (causal attention: that group has seen swaps 1..j and nothing
later), and decode swap pair j from the group in the middle of swap j.
5-fold CV, logistic regression; opaque3 vs transp3.
"""
from pathlib import Path
import argparse
import json
import math
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/tracking_algo_v1'
T0, CYC, SWAP, GROUP_SEC = 2.0, 1.25, 1.0, 0.5


def ap_(s, a, b):
    return b if s == a else a if s == b else s


def cv(X, y):
    if len(set(y)) < 2:
        return float('nan')
    acc = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[tr], y[tr])
        acc.append((m.predict(X[te]) == y[te]).mean())
    return round(float(np.mean(acc)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    Z = np.load(BASE / f'tokenprobe/{a.model}.npz')
    meta = json.loads((BASE / f'tokenprobe/{a.model}.json').read_text())
    items = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'data/items.jsonl')}
    out = {'layers': meta['layers']}
    for s in ('opaque3', 'transp3'):
        ids = [i for i in Z.files if items[i]['set'] == s]
        R = {}
        for li, L in enumerate(meta['layers']):
            r = {}
            for j in range(1, 7):
                use = [i for i in ids if items[i]['k'] >= j]
                g_after = math.ceil((T0 + (j - 1) * CYC + SWAP) / GROUP_SEC)
                g_mid = int((T0 + (j - 1) * CYC + SWAP / 2) // GROUP_SEC)
                use = [i for i in use if Z[i].shape[1] > g_after]
                if len(use) < 50:
                    continue
                st, pr = [], []
                for i in use:
                    x = items[i]['init']
                    for a_, b_ in items[i]['swaps'][:j]:
                        x = ap_(x, a_, b_)
                    st.append(x)
                    pr.append(str(sorted(items[i]['swaps'][j - 1])))
                Xa = np.stack([Z[i][li, g_after] for i in use]).astype(np.float32)
                Xm = np.stack([Z[i][li, g_mid] for i in use]).astype(np.float32)
                Xe = np.stack([Z[i][li, -1] for i in use]).astype(np.float32)
                r[f'S{j}@after'] = cv(Xa, np.array(st))
                r[f'S{j}@end'] = cv(Xe, np.array(st))
                r[f'pair{j}@mid'] = cv(Xm, np.array(pr))
            R[f'L{L}'] = r
            print(a.model, s, f'L{L}', r, flush=True)
        out[s] = R
    (BASE / f'tokenprobe/{a.model}_analysis.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
