#!/usr/bin/env python3
"""Depth set analysis: decodability of S_j (j=1..4) at the video-token group
right after swap j and at the last token, overall and split by how many
times the ball has moved so far (0 / 1 / 2+ moves up to swap j)."""
from pathlib import Path
import argparse
import json
import math
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
B = Path(__file__).resolve().parents[1] / 'outputs/tracking_algo_v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    Z = np.load(B / f'tokenprobe/{a.model}_depth.npz')
    layers = json.loads((B / f'tokenprobe/{a.model}_depth.json').read_text())['layers']
    items = {json.loads(l)['id']: json.loads(l) for l in open(B / 'data_depth/items.jsonl')}
    ids = sorted(Z.files)
    out = {}
    for li, L in enumerate(layers):
        r = {}
        for j in range(1, 5):
            S, moves = [], []
            for i in ids:
                s, mv = items[i]['init'], 0
                for a_, b_ in items[i]['swaps'][:j]:
                    if s in (a_, b_):
                        mv += 1
                    s = b_ if s == a_ else a_ if s == b_ else s
                S.append(s); moves.append(min(mv, 2))
            S, moves = np.array(S), np.array(moves)
            g = [min(math.ceil(items[i]['swap_end_times'][j - 1] / 0.5), Z[i].shape[1] - 2) for i in ids]
            for where, X in (('after', np.stack([Z[i][li, gi] for i, gi in zip(ids, g)])),
                             ('last', np.stack([Z[i][li, -1] for i in ids]))):
                pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
                                         X.astype(np.float32), S, cv=StratifiedKFold(5, shuffle=True, random_state=0))
                ok = pred == S
                r[f'S{j}@{where}'] = round(float(ok.mean()), 3)
                r[f'S{j}@{where}_by_moves'] = {int(m): round(float(ok[moves == m].mean()), 3) for m in np.unique(moves)}
        out[f'L{L}'] = r
        print(a.model, f'L{L}', {k: v for k, v in r.items() if 'by_moves' not in k}, flush=True)
    (B / f'depth_{a.model}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
