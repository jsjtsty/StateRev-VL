#!/usr/bin/env python3
"""Addendum H: causal activation patching with TWO swaps (see PREREG.md).

Based on Addendum G (tracking_algo_patch.py).

Target video B and source video A share the initial position and all timing
(one swap), but end in different ball positions. At residual-stream layer l
(hidden_states[l], i.e. the output of decoder layer l-1) we overwrite B's
video-token activations with A's, restricted to
  post : temporal groups after the swap has finished,
  mid  : temporal groups during the swap,
  all  : every video token.
Readouts on the patched forward: (1) a frozen linear probe on the last-token
hidden state (layer 24 and 36; fit on the unpatched train split) and (2) the
model's own option-letter logits. Control: a source with the SAME final
state as the target (different video) should change nothing.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, build_inputs, question, letter_ids  # noqa: E402

DATA = ROOT / 'outputs/tracking_algo_v1/data_patch2'
LABELS = ('ground_truth', 'S1', 'pair1', 'pair2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3vl8b')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--layers', default='8,16')
    ap.add_argument('--probe-layers', default='24,36')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/patch')
    a = ap.parse_args()
    global PROBE_LAYERS
    PROBE_LAYERS = tuple(int(x) for x in a.probe_layers.split(','))
    a.out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(DATA / 'items.jsonl')]
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    layers_mod = model.model.language_model.layers
    vid_tok = model.config.video_token_id
    lid = letter_ids(proc.tokenizer, 'ABC')
    patch_layers = [int(x) for x in a.layers.split(',')]

    def prep(it):
        frames = np.load(DATA / f'{it["id"]}.npz')['frames']
        q, _ = question(it, 'init')
        inp, _ = build_inputs(cfg, proc, frames, q, it['sample_fps'])
        T = int(inp['video_grid_thw'][0][0])
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        pos = (inp['input_ids'][0] == vid_tok).nonzero().flatten()
        groups = pos.view(T, -1)
        return inp, groups

    def run(inp, patch=None):
        """patch: dict layer -> (positions LongTensor, values [n, D]) applied to hidden_states[layer]."""
        hooks = []
        if patch:
            for L, (p, v) in patch.items():
                def hook(mod, args, out, p=p, v=v):
                    h = out[0] if isinstance(out, tuple) else out
                    h[0, p] = v.to(h.dtype)
                    return out
                hooks.append(layers_mod[L - 1].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                o = model(**inp, output_hidden_states=True, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        lg = o.logits[0, -1].float()
        sc = np.array([float(max(lg[i] for i in lid[l])) for l in 'ABC'])
        last = {L: o.hidden_states[L][0, -1].float().cpu().numpy() for L in PROBE_LAYERS}
        return o, sc, last

    # 1) probe on train split
    tr = [it for it in items if it['split'] == 'train']
    Xs = {L: [] for L in PROBE_LAYERS}
    for it in tr:
        inp, _ = prep(it)
        _, _, last = run(inp)
        for L in PROBE_LAYERS:
            Xs[L].append(last[L])
    from sklearn.model_selection import cross_val_score
    probes, cvacc = {}, {}
    for L in PROBE_LAYERS:
        X = np.stack(Xs[L])
        for lab in LABELS:
            y = [it['labels'][lab] for it in tr]
            mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
            cvacc[f'{lab}@L{L}'] = float(cross_val_score(mk(), X, y, cv=5).mean())
            probes[(L, lab)] = mk().fit(X, y)
    print('probes fit', json.dumps(cvacc), flush=True)
    (a.out / f'{a.model}_probe_cv_2swap.json').write_text(json.dumps(cvacc, indent=1))

    # 2) patching on test split
    te = [it for it in items if it['split'] == 'test']
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    lab = lambda it, k: it['labels'][k]
    for n, tgt in enumerate(te):
        inp_t, groups = prep(tgt)
        T = groups.shape[0]

        def gsel(t_start, t_end):
            gs = [g for g in range(T) if 0.5 * g < t_end and 0.5 * g + 0.25 >= t_start]
            return groups[gs].flatten()
        e1, e2 = tgt['swap_end_times']
        sel = {'swap1': gsel(e1 - 1.0, e1), 'swap2': gsel(e2 - 1.0, e2), 'post2': gsel(e2 + 0.01, 99)}
        # source types: differ only in pair2 (S2 differs) / only in pair1 (S1 differs)
        srcs = {
            'dp2': [s for s in te if s['init'] == tgt['init'] and lab(s, 'pair1') == lab(tgt, 'pair1')
                    and lab(s, 'pair2') != lab(tgt, 'pair2') and lab(s, 'ground_truth') != lab(tgt, 'ground_truth')],
            'dp1': [s for s in te if s['init'] == tgt['init'] and lab(s, 'pair2') == lab(tgt, 'pair2')
                    and lab(s, 'pair1') != lab(tgt, 'pair1') and lab(s, 'S1') != lab(tgt, 'S1')],
        }
        if not all(srcs.values()):
            continue
        base = run(inp_t)
        conds = [('none', None, None, None)]
        caches = {}
        for kind, cands in srcs.items():
            src = cands[int(rng.integers(len(cands)))]
            inp_s, g_s = prep(src)
            assert torch.equal(g_s, groups)
            caches[kind] = (src, run(inp_s)[0].hidden_states)
            for L in patch_layers:
                for where in ('swap1', 'swap2', 'post2'):
                    conds.append((f'{kind}_{where}_L{L}', kind, L, where))
        for cname, kind, L, where in conds:
            if kind is None:
                o, sc, last = base
                src = tgt
            else:
                src, hs = caches[kind]
                p = sel[where]
                o, sc, last = run(inp_t, {L: (p, hs[L][0, p])})
            rec = {'target': tgt['id'], 'source': src['id'], 'cond': cname, 'native_pred': int(sc.argmax())}
            for k in LABELS:
                rec[f'tgt_{k}'] = lab(tgt, k); rec[f'src_{k}'] = lab(src, k)
                for Lp in PROBE_LAYERS:
                    rec[f'p{Lp}_{k}'] = int(probes[(Lp, k)].predict(last[Lp][None])[0])
            rows.append(rec)
        if (n + 1) % 20 == 0:
            print(f'{n + 1}/{len(te)} targets {time.time() - t0:.0f}s', flush=True)
    with open(a.out / f'{a.model}_rows_2swap.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('DONE')


if __name__ == '__main__':
    main()
