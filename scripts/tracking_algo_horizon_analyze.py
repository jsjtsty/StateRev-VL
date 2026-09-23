#!/usr/bin/env python3
"""Addendum C analysis: final-state decodability per horizon condition."""
from pathlib import Path
import argparse
import glob
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
BASE = ROOT / 'outputs/tracking_algo_v1'
PRIMARY = {'qwen3vl8b': 24, 'qwen35_9b': 18, 'qwen36_27b': 34}


def cv(X, y, rng_state=0):
    if len(set(y)) < 2:
        return float('nan')
    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        return float('nan')
    acc = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=rng_state).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[tr], y[tr])
        acc.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--name', default='horizon', help='horizon | binding')
    a = ap.parse_args()
    Z = np.load(BASE / f'tokenprobe/{a.model}_{a.name}.npz')
    layers = json.loads((BASE / f'tokenprobe/{a.model}_{a.name}.json').read_text())['layers']
    items = {json.loads(l)['id']: json.loads(l) for l in open(BASE / f'data_{a.name}/items.jsonl')}
    rows = {}
    for f in glob.glob(str(BASE / f'eval_{a.name}' / a.model / 'rows_shard*.jsonl')):
        for l in open(f):
            r = json.loads(l); rows[r['id']] = r['pred']
    conds = sorted({items[i]['set'] for i in Z.files})
    out = {}
    lines = [f'## {a.model}', '', '| condition | majority | last-token (primary L%d) | last-token best layer | video group after last swap (primary L) | native acc |' % PRIMARY.get(a.model, -1),
             '|---|---|---|---|---|---|']
    for c in conds:
        ids = [i for i in Z.files if items[i]['set'] == c]
        y = np.array([items[i]['labels']['ground_truth'] for i in ids])
        R = {'majority': float(np.bincount(y).max() / len(y))}
        per_layer_last, per_layer_after = {}, {}
        for li, L in enumerate(layers):
            Xl = np.stack([Z[i][li, -1] for i in ids]).astype(np.float32)
            g = [min(math.ceil(items[i]['swap_end_times'][-1] / 0.5), Z[i].shape[1] - 2) for i in ids]
            Xa = np.stack([Z[i][li, gi] for i, gi in zip(ids, g)]).astype(np.float32)
            per_layer_last[L] = round(cv(Xl, y), 3)
            per_layer_after[L] = round(cv(Xa, y), 3)
        R['last_token'] = per_layer_last
        R['after_last_swap'] = per_layer_after
        nat = [rows[i] == items[i]['labels']['ground_truth'] for i in ids if i in rows]
        R['native_acc'] = float(np.mean(nat)) if nat else None
        out[c] = R
        pL = PRIMARY.get(a.model, layers[len(layers) // 2])
        best = max(per_layer_last, key=per_layer_last.get)
        lines.append(f"| {c} | {R['majority']:.2f} | {per_layer_last.get(pL, float('nan')):.3f} | {per_layer_last[best]:.3f} (L{best}) | "
                     f"{per_layer_after.get(pL, float('nan')):.3f} | {R['native_acc'] if R['native_acc'] is None else round(R['native_acc'], 3)} |")
    (BASE / f'{a.name}_{a.model}.json').write_text(json.dumps(out, indent=1))
    (BASE / f'{a.name}_{a.model}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
