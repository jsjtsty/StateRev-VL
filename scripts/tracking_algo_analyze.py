#!/usr/bin/env python3
"""Pre-registered G1 analysis (outputs/tracking_algo_v1/PREREG.md)."""
from pathlib import Path
import argparse
import glob
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/tracking_algo_v1'


def load(model):
    rows = [json.loads(l) for f in glob.glob(str(BASE / 'eval' / model / 'rows_shard*.jsonl')) for l in open(f)]
    if not rows:
        return None
    items = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'data/items.jsonl')}
    df = pd.DataFrame(rows).drop_duplicates(['id', 'variant'])
    for name in items[next(iter(items))]['labels']:
        df[name] = [items[i]['labels'][name] for i in df['id']]
    df['agree'] = [items[i]['agree'] for i in df['id']]
    df['k'] = [items[i]['k'] for i in df['id']]
    df['n_cups'] = [items[i]['n_cups'] for i in df['id']]
    return df


def boot_ci(x, rng, n=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = rng.integers(0, len(x), (n, len(x)))
    v = x[m].mean(1)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def boot_p_below(x, thr, rng, n=10000):
    x = np.asarray(x, float)
    v = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float((v >= thr).mean())          # one-sided p for mean < thr


def perm_null(pred, target, rng, n=10000):
    pred, target = np.array(pred), np.array(target)
    obs = (pred == target).mean()
    v = np.array([(rng.permutation(pred) == target).mean() for _ in range(n)])
    return float(obs), float(v.mean()), float((v >= obs).mean())


def analyze(df, model, rng):
    res = {'model': model}
    for variant in sorted(df.variant.unique()):
        d = df[df.variant == variant]
        R = {}
        for s in sorted(d.set.unique()):
            x = d[d.set == s]
            c = (x.pred == x.ground_truth).astype(int)
            R[s] = {'n': len(x), 'acc': float(c.mean()), 'acc_ci': boot_ci(c, rng),
                    'answer_dist': x.pred.value_counts(normalize=True).sort_index().round(3).to_dict()}
            for ag in (True, False):
                y = x[x.agree == ag]
                if len(y):
                    cc = (y.pred == y.ground_truth).astype(int)
                    R[s]['agree' if ag else 'disagree'] = {'n': len(y), 'acc': float(cc.mean()), 'acc_ci': boot_ci(cc, rng),
                                                           'p_shortcut': float((y.pred == y.last_touch_init).mean())}
            fam = [c for c in df.columns if c in ('ground_truth', 'last_touch_init', 'stale', 'first_swap_only',
                                                  'last_swap_on_init', 'last_swap_endpoint', 'stop_after_1',
                                                  'stop_after_2', 'stop_after_3')]
            R[s]['family_agreement'] = {f: round(float((x.pred == x[f]).mean()), 3) for f in fam}
            R[s]['acc_by_k'] = {f'k{k}_{"agree" if ag else "disagree"}': round(float((g.pred == g.ground_truth).mean()), 3)
                                for (k, ag), g in x.groupby(['k', 'agree'])}
        # pre-registered tests on opaque3
        if 'opaque3' in d.set.values:
            o = d[d.set == 'opaque3']
            dis, agr = o[~o.agree], o[o.agree]
            cd = (dis.pred == dis.ground_truth).astype(int).values
            ca = (agr.pred == agr.ground_truth).astype(int).values
            p_h1a = boot_p_below(cd, 1 / 3, rng)
            obs, null, p_perm = perm_null(dis.pred.values, dis.last_touch_init.values, rng)
            p_sc_vs_gt = float(np.mean([(lambda m: (dis.pred.values[m] == dis.last_touch_init.values[m]).mean()
                                         - (dis.pred.values[m] == dis.ground_truth.values[m]).mean())(
                rng.integers(0, len(dis), len(dis))) <= 0 for _ in range(10000)]))
            diff = [(ca[rng.integers(0, len(ca), len(ca))].mean() - cd[rng.integers(0, len(cd), len(cd))].mean()) for _ in range(10000)]
            T = {'H1a_disagree_acc': float(cd.mean()), 'H1a_p_(acc<1/3)': p_h1a, 'H1a': p_h1a < .05,
                 'H1b_p_shortcut': obs, 'H1b_perm_null': null, 'H1b_p_perm': p_perm,
                 'H1b_p_gt': float(cd.mean()), 'H1b_p_(shortcut<=gt)': p_sc_vs_gt,
                 'H1b': p_perm < .05 and p_sc_vs_gt < .05,
                 'H1c_agree_minus_disagree': float(ca.mean() - cd.mean()), 'H1c_p': float(np.mean(np.array(diff) <= 0)),
                 'H1c': float(np.mean(np.array(diff) <= 0)) < .05}
            if 'transp3' in d.set.values:
                t = d[d.set == 'transp3']
                td = t[~t.agree]
                ct = (td.pred == td.ground_truth).astype(int).values
                d2 = [ct[rng.integers(0, len(ct), len(ct))].mean() - cd[rng.integers(0, len(cd), len(cd))].mean() for _ in range(10000)]
                T.update({'H2_transp_disagree_acc': float(ct.mean()), 'H2_p_(acc<=1/3)': boot_p_below(ct, 1 / 3, rng) if False else float(np.mean(ct[rng.integers(0, len(ct), (10000, len(ct)))].mean(1) <= 1 / 3)),
                          'H2_p_(transp<=opaque)': float(np.mean(np.array(d2) <= 0))})
                T['H2'] = T['H2_p_(acc<=1/3)'] < .05 and T['H2_p_(transp<=opaque)'] < .05
            if 'colored3' in d.set.values:
                c3 = d[d.set == 'colored3'].assign(seq=lambda z: z.id.str.split('_').str[1]).set_index('seq')
                o3 = o.assign(seq=lambda z: z.id.str.split('_').str[1]).set_index('seq')
                common = c3.index.intersection(o3.index)
                cc = (c3.loc[common, 'pred'] == c3.loc[common, 'ground_truth']).astype(int).values
                oc = (o3.loc[common, 'pred'] == o3.loc[common, 'ground_truth']).astype(int).values
                ii = rng.integers(0, len(common), (10000, len(common)))
                dd = cc[ii].mean(1) - oc[ii].mean(1)
                T.update({'H4_colored_acc': float(cc.mean()), 'H4_opaque_acc_paired': float(oc.mean()),
                          'H4_diff_ci': [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))],
                          'H4_p': float((dd <= 0).mean()), 'H4': float((dd <= 0).mean()) < .05})
            R['tests_opaque3'] = T
        res[variant] = R
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='qwen3vl8b,qwen35_9b,qwen25vl7b,llava_video7b,qwen36_27b,qwen36_35b_a3b,qwen3vl8b_think')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    allres = {}
    lines = ['| model | variant | opaque3 acc | opaque3 disagree acc | P(ans=shortcut) disagree [null] | transp3 acc | colored3 acc | k1 acc | opaque4 / 5 acc | H1a | H1b | H2 | H4 |', '|' + '---|' * 13]
    for m in a.models.split(','):
        df = load(m)
        if df is None:
            continue
        r = analyze(df, m, rng)
        allres[m] = r
        for v in ('init', 'vis'):
            if v not in r or 'tests_opaque3' not in r[v]:
                continue
            R, T = r[v], r[v]['tests_opaque3']
            g = lambda s_: R.get(s_, {}).get('acc', float('nan'))
            lines.append(f"| {m} | {v} | {g('opaque3'):.3f} | {R['opaque3']['disagree']['acc']:.3f} | "
                         f"{T['H1b_p_shortcut']:.3f} [{T['H1b_perm_null']:.3f}] | {g('transp3'):.3f} | {g('colored3'):.3f} | "
                         f"{g('k1'):.3f} | {g('opaque4'):.3f} / {g('opaque5'):.3f} | {T['H1a']} | {T['H1b']} | {T.get('H2')} | {T.get('H4')} |")
    (BASE / 'g1_results.json').write_text(json.dumps(allres, indent=1, default=str))
    (BASE / 'g1_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
