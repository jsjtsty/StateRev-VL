#!/usr/bin/env python3
"""Addendum H analysis: per condition, how often each last-token probe reports
the SOURCE value (among pairs where source and target differ), and native
answer following/changing."""
from pathlib import Path
import argparse
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('ground_truth', 'S1', 'pair1', 'pair2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3vl8b')
    ap.add_argument('--probe-layer', type=int, default=24)
    a = ap.parse_args()
    P = ROOT / 'outputs/tracking_algo_v1/patch'
    df = pd.DataFrame([json.loads(l) for l in open(P / f'{a.model}_rows_2swap.jsonl')])
    base = df[df['cond'] == 'none'].set_index('target')
    L = a.probe_layer
    out = []
    for c, g in df[df['cond'] != 'none'].groupby('cond', sort=False):
        r = {'cond': c, 'n': len(g)}
        for k in LABELS:
            d = g[g[f'src_{k}'] != g[f'tgt_{k}']]
            r[f'{k}_follow_src'] = (d[f'p{L}_{k}'] == d[f'src_{k}']).mean() if len(d) else float('nan')
            # baseline: unpatched probe on the same targets reporting the source value
            b = base.loc[d['target']]
            r[f'{k}_base'] = (b[f'p{L}_{k}'].values == d[f'src_{k}'].values).mean() if len(d) else float('nan')
        b = base.loc[g['target']]
        r['native_changed'] = (g['native_pred'].values != b['native_pred'].values).mean()
        r['native_follow_src_gt'] = (g['native_pred'] == g['src_ground_truth']).mean()
        out.append(r)
    res = pd.DataFrame(out)
    pd.set_option('display.width', 250)
    print(res.round(3).to_string(index=False))
    nb = df[df['cond'] == 'none']
    print('baseline: native acc', round((nb['native_pred'] == nb['tgt_ground_truth']).mean(), 3),
          '| probe acc', {k: round((nb[f'p{L}_{k}'] == nb[f'tgt_{k}']).mean(), 3) for k in LABELS})
    res.to_csv(P / f'{a.model}_patch2_summary_L{L}.csv', index=False)


if __name__ == '__main__':
    main()
