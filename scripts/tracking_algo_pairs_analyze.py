#!/usr/bin/env python3
"""Addendum A analysis: minimal-pair flip rates (see PREREG.md)."""
from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/tracking_algo_v1'


def main():
    rng = np.random.default_rng(0)
    items = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'data_pairs/items.jsonl')}
    lines = ['| model | n pairs | acc base | flip gt_only | flip sc_only | flip neither | sc−gt [95% CI] | sc−neither [95% CI] | gt−neither [95% CI] | H3a | H3b |',
             '|' + '---|' * 11]
    res = {}
    for d in sorted(glob.glob(str(BASE / 'eval_pairs' / '*'))):
        model = Path(d).name
        rows = [json.loads(l) for f in glob.glob(d + '/rows_shard*.jsonl') for l in open(f)]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['pair'] = [items[i]['pair'] for i in df.id]
        df['kind'] = [items[i]['kind'] for i in df.id]
        df['gt'] = [items[i]['labels']['ground_truth'] for i in df.id]
        w = df.pivot_table(index='pair', columns='kind', values='pred', aggfunc='first').dropna()
        if len(w) == 0:
            continue
        F = {k: (w[k] != w['base']).astype(float).values for k in ('gt_only', 'sc_only', 'neither')}
        n = len(w)
        idx = rng.integers(0, n, (10000, n))

        def diff(a, b):
            v = F[a][idx].mean(1) - F[b][idx].mean(1)
            return float(F[a].mean() - F[b].mean()), [float(np.quantile(v, .025)), float(np.quantile(v, .975))], float((v <= 0).mean())
        sg, sg_ci, sg_p = diff('sc_only', 'gt_only')
        sn, sn_ci, _ = diff('sc_only', 'neither')
        gn, gn_ci, _ = diff('gt_only', 'neither')
        base = df[df.kind == 'base']
        acc_base = float((base['pred'] == base['gt']).mean())
        h3a = sg_p < .05
        h3b = gn_ci[0] <= 0 or gn_ci[1] < .10
        res[model] = {'n': n, 'acc_base': acc_base, **{f'flip_{k}': float(v.mean()) for k, v in F.items()},
                      'sc_minus_gt': [sg, sg_ci, sg_p], 'sc_minus_neither': [sn, sn_ci], 'gt_minus_neither': [gn, gn_ci],
                      'H3a': h3a, 'H3b': h3b}
        lines.append(f'| {model} | {n} | {acc_base:.3f} | {F["gt_only"].mean():.3f} | {F["sc_only"].mean():.3f} | {F["neither"].mean():.3f} | '
                     f'{sg:+.3f} [{sg_ci[0]:+.3f},{sg_ci[1]:+.3f}] | {sn:+.3f} [{sn_ci[0]:+.3f},{sn_ci[1]:+.3f}] | '
                     f'{gn:+.3f} [{gn_ci[0]:+.3f},{gn_ci[1]:+.3f}] | {h3a} | {h3b} |')
    (BASE / 'pairs_results.json').write_text(json.dumps(res, indent=1))
    (BASE / 'pairs_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
