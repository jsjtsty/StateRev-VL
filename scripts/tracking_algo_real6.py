#!/usr/bin/env python3
"""Addendum AG (revision 1): the last resting prior state held 4 s then the final state 0.5 s (2 fps), real crop frames vs\nrendered tiles of the same words; held vs frozen (final frame x9) vs image-only."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B  # noqa: E402
from tracking_algo_real2 import D, ANN, letter_times  # noqa: E402
from tracking_algo_real4 import build  # noqa: E402
from tracking_algo_synthword import ask, render  # noqa: E402

FPS, HOLD = 2.0, 8


def resting_states(v, fin, rest=0.25):
    """Letter orders at rest, from the 1-fps boxes: a sample is at rest when no letter centroid moves more than
    rest x median letter height from the previous and to the next sample; consecutive at-rest samples with the same
    order form a run; returns [(first_frame, last_frame, word)] with repeated orders merged, or None if the last run
    is not the final order."""
    lids = letter_times(v)[0]
    obs = [o for o in v['object_tracking'] if o['id'] in lids]
    if not obs or any(not o['label'].startswith('letter') for o in obs):
        return None
    frames = sorted(set.intersection(*[set(o['frame_ids']) for o in obs]))
    if len(frames) < 3:
        return None
    box = lambda o, f: o['bounding_boxes'][o['frame_ids'].index(f)]
    C = np.array([[((box(o, f)[0] + box(o, f)[2]) / 2, (box(o, f)[1] + box(o, f)[3]) / 2) for o in obs] for f in frames])
    h = np.median([box(o, f)[3] - box(o, f)[1] for o in obs for f in frames])
    mv = np.linalg.norm(np.diff(C, axis=0), axis=2).max(1)
    still = np.r_[True, mv < rest * h] & np.r_[mv < rest * h, True]
    names = [o['label'].split('-', 1)[1].lower() for o in obs]
    last = C[-1]; ax = 0 if np.ptp(last[:, 0]) >= np.ptp(last[:, 1]) else 1
    word = lambda c, rev: (lambda s: s[::-1] if rev else s)(''.join(names[i] for i in np.argsort(c[:, ax])))
    rev = next((r for r in (False, True) if word(last, r) == fin), None)
    if rev is None:
        return None
    runs = []
    for f, c, st in zip(frames, C, still):
        if not st:
            continue
        w = word(c, rev)
        if runs and runs[-1][2] == w:
            runs[-1][1] = f
        else:
            runs.append([f, f, w])
    return runs if runs and runs[-1][2] == fin else None


def domain_clips(it, rs, crop, fidx):
    """Prior resting state held 8 frames then the final frame, real and rendered; frozen = final ×9."""
    a, b, _ = rs[-2]
    prior = crop[int(np.argmin(np.abs(np.asarray(fidx) - (a + b) / 2)))]
    last = crop[list(fidx).index(it['ends']['full'])]
    pw, fw = rs[-2][2], rs[-1][2]
    sp, sf = render(pw), render(fw)
    held = lambda p, f: np.concatenate([np.repeat(p[None], HOLD, 0), f[None]])
    return {'real': {'img': last, 'held': held(prior, last), 'frozen': np.repeat(last[None], HOLD + 1, 0)},
            'syn': {'img': sf, 'held': held(sp, sf), 'frozen': np.repeat(sf[None], HOLD + 1, 0)}}


def main():
    import torch
    import zipfile
    from tracking_algo_eval import load, build_inputs, letter_ids
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    ann = json.load(zipfile.ZipFile(ANN).open('all_valid.json'))
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

    rng = np.random.default_rng(20260930)
    out = []
    for it in map(json.loads, open(D / 'items.jsonl')):
        rs = resting_states(ann[it['id']], it['options'][it['gt']])
        if rs is None or len(rs) < 2:
            continue
        pen, qs = build(it, [(f, w) for f, _, w in rs], rng)
        z = np.load(D / f'{it["id"]}.npz')
        C = domain_clips(it, rs, z['crop'], z['fidx'])
        rec = {'id': it['id'], 'rest': rs, 'pen_word': pen}
        for qn, (q, opts, g, s) in qs.items():
            r = {'gt': g, 'stale': s, 'opts': opts}
            for dom, c in C.items():
                r[dom] = {'img': image(c['img'], q, opts), **{k: lp(build_inputs(cfg, proc, c[k], ask(q, opts, False), FPS)[0]) for k in ('held', 'frozen')}}
            rec[qn] = r
        out.append(rec)
        print(a.model, it['id'], {qn: (rec[qn]['gt'], rec[qn]['stale'], rec[qn]['real'], rec[qn]['syn']) for qn in qs}, flush=True)
    (B / f'real6_{a.model}.jsonl').write_text('\n'.join(json.dumps(x) for x in out))


if __name__ == '__main__':
    main()
