#!/usr/bin/env python3
"""G0: shortcut-family analysis on the EXISTING VET-Bench cup data (exploratory).

Native answers (Qwen3-VL-8B canonical cache) vs every member of
staterev.shortcuts.FAMILY, with trajectory-cluster bootstrap CIs and an
answer-permutation null; hidden-state decodability of GT vs the shortcut,
with symbolic baselines (init / init+last event) as confound controls.
"""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shortcuts import FAMILY  # noqa: E402

POS = ['Left', 'Middle', 'Right']
PAIR = {'Left and Middle': (0, 1), 'Middle and Right': (1, 2), 'Left and Right': (0, 2)}
SRC = {
    'qwen': ('outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv', 'outputs/vetbench/hidden_state_probe/hidden_states.npz', (12, 18, 24, 28, 32, 36)),
    'llava': ('outputs/vetbench/llava_next_video_7b_replication_v1/behavior.csv', 'outputs/vetbench/llava_next_video_7b_replication_v1/hidden_states.npz', (8, 16, 20, 24, 32)),
}


def table(bpath):
    d = pd.read_csv(ROOT / bpath)
    rows = []
    for tid, g in d.sort_values(['trajectory_id', 't']).groupby('trajectory_id'):
        init = POS.index(g.initial_state.iloc[0].title())
        swaps = [PAIR[e] for e in g.gt_event]
        for i, r in enumerate(g.itertuples()):
            rec = {'tid': tid, 't': int(r.t), 'init': init, 'last_ev': r.gt_event,
                   'pred': POS.index(str(r.state_pred).title()) if str(r.state_pred).title() in POS else -1}
            for name, f in FAMILY.items():
                rec[name] = f(3, init, swaps[:i + 1])
            assert rec['ground_truth'] == POS.index(r.gt_state.title())
            rows.append(rec)
    return pd.DataFrame(rows)


def cluster_ci(y, col, rng, n=2000):
    ids = y.tid.unique()
    groups = {i: y[y.tid == i] for i in ids}
    v = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.choice(ids, len(ids))])
        v.append((s.pred == s[col]).mean())
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def cv(F, yy, groups, onehot=False):
    acc = []
    for tr, te in GroupKFold(5).split(F, yy, groups):
        m = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LogisticRegression(max_iter=2000)) if onehot else \
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(F[tr], yy[tr])
        acc.append((m.predict(F[te]) == yy[te]).mean())
    return round(float(np.mean(acc)), 3)


def main():
    out = ROOT / 'outputs/tracking_algo_v1/g0'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {}
    for model, (bpath, hpath, layers) in SRC.items():
        x = table(bpath)
        R = {'answer_dist_t>=3': x[x.t >= 3].pred.value_counts(normalize=True).round(3).to_dict()}
        y = x[x.t >= 3]
        if y.pred.nunique() > 1:
            fam = {}
            for c in FAMILY:
                obs = float((y.pred == y[c]).mean())
                null = [float((rng.permutation(y.pred.values) == y[c].values).mean()) for _ in range(2000)]
                fam[c] = {'agree': round(obs, 3), 'cluster_ci': cluster_ci(y, c, rng), 'perm_null': round(float(np.mean(null)), 3),
                          'p_perm': float(np.mean(np.array(null) >= obs))}
            R['native_vs_family_t>=3'] = fam
            m = y.last_touch_init != y.ground_truth
            R['disagree_subset_t>=3'] = {'n': int(m.sum()), 'p_ans=shortcut': float((y.pred[m] == y.last_touch_init[m]).mean()),
                                         'p_ans=gt': float((y.pred[m] == y.ground_truth[m]).mean())}
        z = np.load(ROOT / hpath)
        X = np.stack([z[f'{r.tid}_t{r.t}'] for r in x.itertuples()]).astype(np.float32)
        mm = (x.t >= 2).values
        g = x.tid.values[mm]
        sym = {'init': x[['init']].astype(str).values[mm], 'init+last_event': x[['init', 'last_ev']].astype(str).values[mm]}
        dec = {}
        for lab in ('ground_truth', 'last_touch_init'):
            yy = x[lab].values[mm]
            r = {k: cv(v, yy, g, onehot=True) for k, v in sym.items()}
            for L in layers:
                r[f'L{L}'] = cv(X[mm, L], yy, g)
            dec[lab] = r
        R['hidden_decodability_t>=2'] = dec
        report[model] = R
    (out / 'g0_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:4000])


if __name__ == '__main__':
    main()
