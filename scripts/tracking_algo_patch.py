#!/usr/bin/env python3
"""Addendum G: causal activation patching of the one-update state.

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

DATA = ROOT / 'outputs/tracking_algo_v1/data_patch'
PROBE_LAYERS = (24, 36)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3vl8b')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--layers', default='8,16,24,32')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/patch')
    a = ap.parse_args()
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
    Xs = {L: [] for L in PROBE_LAYERS}; ys = []
    for it in tr:
        inp, _ = prep(it)
        _, _, last = run(inp)
        for L in PROBE_LAYERS:
            Xs[L].append(last[L])
        ys.append(it['labels']['ground_truth'])
    probes = {L: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(np.stack(Xs[L]), ys)
              for L in PROBE_LAYERS}
    print('probes fit', flush=True)

    # 2) patching on test split
    te = [it for it in items if it['split'] == 'test']
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for n, tgt in enumerate(te):
        y_t = tgt['labels']['ground_truth']
        diff = [s for s in te if s['init'] == tgt['init'] and s['labels']['ground_truth'] != y_t]
        same = [s for s in te if s['init'] == tgt['init'] and s['labels']['ground_truth'] == y_t and s['id'] != tgt['id']]
        if not diff or not same:
            continue
        src = diff[int(rng.integers(len(diff)))]
        ctl = same[int(rng.integers(len(same)))]
        inp_t, groups = prep(tgt)
        T = groups.shape[0]
        end_g = int(np.ceil(tgt['swap_end_times'][0] / 0.5))
        start_g = int((tgt['swap_end_times'][0] - 1.0) // 0.5)
        sel = {'post': groups[end_g:].flatten(), 'mid': groups[start_g:end_g].flatten(), 'all': groups.flatten()}
        caches = {}
        for name, s in (('src', src), ('ctl', ctl)):
            inp_s, g_s = prep(s)
            assert inp_s['input_ids'].shape == inp_t['input_ids'].shape and torch.equal(g_s, groups)
            o, _, _ = run(inp_s)
            caches[name] = o.hidden_states
        conds = [('none', None, None, None)]
        for L in patch_layers:
            for where in ('post', 'mid', 'all'):
                conds.append((f'{where}_L{L}', 'src', L, where))
            conds.append((f'post_L{L}_ctl', 'ctl', L, 'post'))
        _, base_sc, _ = run(inp_t)
        y_s = src['labels']['ground_truth']
        for cname, which, L, where in conds:
            patch = None
            if which:
                p = sel[where]
                patch = {L: (p, caches[which][L][0, p])}
            _, sc, last = run(inp_t, patch)
            rec = {'target': tgt['id'], 'source': src['id'] if which != 'ctl' else ctl['id'], 'cond': cname,
                   'y_target': y_t, 'y_source': y_s if which != 'ctl' else y_t,
                   'native_pred': int(sc.argmax()), 'margin_src_minus_tgt': float(sc[y_s] - sc[y_t]),
                   'base_margin_src_minus_tgt': float(base_sc[y_s] - base_sc[y_t])}
            for Lp in PROBE_LAYERS:
                rec[f'probe{Lp}_pred'] = int(probes[Lp].predict(last[Lp][None])[0])
            rows.append(rec)
        if (n + 1) % 20 == 0:
            print(f'{n + 1}/{len(te)} targets {time.time() - t0:.0f}s', flush=True)
    with open(a.out / f'{a.model}_rows.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('DONE')


if __name__ == '__main__':
    main()
