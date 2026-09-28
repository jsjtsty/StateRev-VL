#!/usr/bin/env python3
"""Addendum AD: timeline factors on the synthetic word tiles of Addendum AC.
m (prior states) in {1, 4} x fps in {2, 4} x dur (seconds per prior state) in {1, 4}; final dwell 0.5 s."""
from pathlib import Path
import argparse
import itertools
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B  # noqa: E402
from tracking_algo_real2 import D  # noqa: E402
from tracking_algo_synthword import render, ask, ORD  # noqa: E402

CELLS = list(itertools.product((1, 4), (2, 4), (1, 4)))   # (m, fps, dur)
FINAL = 0.5


def states(it, rng):
    """Prior words (oldest first) for m=4 and the m=1 initial word; the final word is the correct option."""
    o = it['options']
    st = it['pen'] if it['pen'] is not None else next(i for i in range(3) if i != it['gt'])
    fin, init = o[it['gt']], o[st]
    perms = {''.join(p) for p in itertools.permutations(fin)} - {fin, init}
    if len(perms) < 3:     # too few orderings (e.g. 2-letter words) for 4 distinct prior states: item is skipped
        return None
    extra = [str(x) for x in rng.permutation(sorted(perms))[:3]]
    return init, [init] + extra, fin


def questions(pen, fin, rng):
    k = next(i for i in range(len(fin)) if pen[i] != fin[i])
    third = next((c for c in fin if c not in (fin[k], pen[k])), 'x')
    q = []
    for text, opts in ((f'What letter is at the {ORD[k]} position from the left at the end?', [fin[k].upper(), pen[k].upper(), third.upper()]),
                       ('What is the order of the letters at the end?', None)):
        if opts is None:
            others = sorted({''.join(p) for p in itertools.permutations(fin)} - {fin, pen})
            opts = [fin, pen, others[int(rng.integers(len(others)))] if others else fin[::-1]]
        perm = rng.permutation(3)
        q.append((text, [opts[i] for i in perm], int(np.where(perm == 0)[0][0]), int(np.where(perm == 1)[0][0])))
    return q


def clip(prior, fin, fps, dur):
    fr = [np.repeat(render(w)[None], int(round(dur * fps)), 0) for w in prior]
    fr.append(np.repeat(render(fin)[None], max(1, int(round(FINAL * fps))), 0))
    return np.concatenate(fr)


def main():
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}

    def lp(inp):
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        return int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))

    def image(f, q, opts):
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': ask(q, opts, True)}]}],
                                        tokenize=False, add_generation_prompt=True, **tkw)
        return lp(proc(text=[text], images=[f], return_tensors='pt', **cfg.get('proc_kw', {})))

    rng = np.random.default_rng(20260929)
    out = []
    for it in map(json.loads, open(D / 'items.jsonl')):
        st = states(it, rng)
        if st is None:
            continue
        init, prior4, fin = st
        rec = {'id': it['id'], 'fin': fin, 'prior4': prior4}
        for m in (1, 4):
            prior = [init] if m == 1 else prior4
            qs = questions(prior[-1], fin, rng)
            for qn, (q, opts, g, s) in zip(('pos', 'order'), qs):
                key = f'm{m}_{qn}'
                rec[key] = {'gt': g, 'stale': s, 'opts': opts, 'img': image(render(fin), q, opts)}
                for mm, fps, dur in CELLS:
                    if mm == m:
                        rec[key][f'fps{fps}_dur{dur}'] = lp(build_inputs(cfg, proc, clip(prior, fin, fps, dur), ask(q, opts, False), float(fps))[0])
        out.append(rec)
        print(a.model, it['id'], {k: (v['gt'], v['img'], [v[c] for c in v if c.startswith('fps')]) for k, v in rec.items() if k.startswith('m')}, flush=True)
    (B / f'synthword2_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for m, fps, dur in CELLS:
        for qn in ('pos', 'order'):
            k = f'm{m}_{qn}'; c = f'fps{fps}_dur{dur}'
            img = np.mean([r[k]['img'] == r[k]['gt'] for r in out]); nat = np.mean([r[k][c] == r[k]['gt'] for r in out])
            print(a.model, f'm={m} fps={fps} dur={dur} {qn:5s} img {img:.3f} native {nat:.3f} gap {img - nat:+.3f} '
                  f"P(stale) {np.mean([r[k][c] == r[k]['stale'] for r in out]):.3f}", flush=True)


if __name__ == '__main__':
    main()
