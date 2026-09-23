#!/usr/bin/env python3
"""Addendum G analysis: per condition, how often the probe / native answer
report the SOURCE vs TARGET state, and the native logit-margin shift."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[1] / 'outputs/tracking_algo_v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3vl8b')
    a = ap.parse_args()
    df = pd.DataFrame([json.loads(l) for l in open(B / f'patch/{a.model}_rows.jsonl')])
    rng = np.random.default_rng(0)
    rows = []
    for c, g in df.groupby('cond', sort=False):
        shift = (g.margin_src_minus_tgt - g.base_margin_src_minus_tgt).values
        bs = [shift[rng.integers(0, len(shift), len(shift))].mean() for _ in range(5000)]
        r = {'cond': c, 'n': len(g)}
        for L in (24, 36):
            r[f'probe{L}=src'] = round(float((g[f'probe{L}_pred'] == g.y_source).mean()), 3)
            r[f'probe{L}=tgt'] = round(float((g[f'probe{L}_pred'] == g.y_target).mean()), 3)
        r['native=src'] = round(float((g.native_pred == g.y_source).mean()), 3)
        r['native=tgt'] = round(float((g.native_pred == g.y_target).mean()), 3)
        r['margin_shift'] = round(float(shift.mean()), 3)
        r['margin_shift_ci'] = [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)]
        rows.append(r)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    (B / f'patch_{a.model}.md').write_text(out.to_markdown(index=False) + '\n')


if __name__ == '__main__':
    main()
