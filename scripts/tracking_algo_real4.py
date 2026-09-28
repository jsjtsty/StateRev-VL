#!/usr/bin/env python3
"""Addendum AE: real letters clips (AA crop view) asked about the actual penultimate arrangement, reconstructed from
the official 1-fps letter boxes."""
from pathlib import Path
import argparse
import itertools
import json
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B, FPS  # noqa: E402
from tracking_algo_real2 import D, ANN, clip, letter_times  # noqa: E402
from tracking_algo_synthword import ask, ORD  # noqa: E402


def state_sequence(v, fin):
    """Letter orders over time (merged duplicates), in the reading direction of the correct option; None if unmatched."""
    lids = letter_times(v)[0]
    obs = [o for o in v['object_tracking'] if o['id'] in lids]
    if not obs or any(not o['label'].startswith('letter') for o in obs):
        return None
    frames = sorted(set.intersection(*[set(o['frame_ids']) for o in obs]))
    if not frames:
        return None
    seq = []
    for f in frames:
        pts = []
        for o in obs:
            b = o['bounding_boxes'][o['frame_ids'].index(f)]
            pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, o['label'].split('-', 1)[1].lower()))
        seq.append((f, pts))
    last = seq[-1][1]
    ax = 0 if np.ptp([p[0] for p in last]) >= np.ptp([p[1] for p in last]) else 1

    def word(pts, rev):
        s = ''.join(p[2] for p in sorted(pts, key=lambda p: p[ax]))
        return s[::-1] if rev else s
    rev = next((r for r in (False, True) if word(last, r) == fin), None)
    if rev is None:
        return None
    states = []
    for f, pts in seq:
        w = word(pts, rev)
        if not states or states[-1][1] != w:
            states.append((f, w))
    return states


def build(it, states, rng):
    fin = it['options'][it['gt']]
    pen = next(w for f, w in reversed(states) if w != fin)
    k = next(i for i in range(len(fin)) if pen[i] != fin[i])
    third = next((c for c in fin if c not in (fin[k], pen[k])), 'x')
    beg = states[0][1]
    others = sorted({''.join(p) for p in itertools.permutations(fin)} - {fin, pen})
    oth = beg if beg not in (fin, pen) else (others[int(rng.integers(len(others)))] if others else None)
    qs = {}
    for name, text, opts in (('pos', f'At the end, which letter is in the {ORD[k]} place of the letter sequence?', [fin[k].upper(), pen[k].upper(), third.upper()]),
                             ('order', it['question'], [fin, pen, oth])):
        if None in opts or len(set(opts)) < 3:
            continue
        perm = rng.permutation(3)
        qs[name] = (text, [opts[i] for i in perm], int(np.where(perm == 0)[0][0]), int(np.where(perm == 1)[0][0]))
    return pen, qs


def main():
    import torch
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
        states = state_sequence(ann[it['id']], it['options'][it['gt']])
        if states is None or len(states) < 2:
            continue
        pen, qs = build(it, states, rng)
        z = np.load(D / f'{it["id"]}.npz')
        rec = {'id': it['id'], 'n_changes': len(states) - 1, 'pen_word': pen, 'states': states}
        for qn, (q, opts, g, s) in qs.items():
            r = {'gt': g, 'stale': s, 'opts': opts}
            for dk in ('0.5', 'full'):
                fr = clip(it, z['crop'], z['fidx'], dk)[0]
                r[dk] = (image(fr[-1], q, opts), lp(build_inputs(cfg, proc, fr, ask(q, opts, False), FPS)[0]))
            rec[qn] = r
        out.append(rec)
        print(a.model, it['id'], rec['n_changes'], {qn: (rec[qn]['gt'], rec[qn]['0.5'], rec[qn]['full']) for qn in qs}, flush=True)
    (B / f'real4_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for qn in ('pos', 'order'):
        for dk in ('0.5', 'full'):
            r = [x for x in out if qn in x]
            print(a.model, qn, dk, 'n', len(r), 'img', round(float(np.mean([x[qn][dk][0] == x[qn]['gt'] for x in r])), 3),
                  'native', round(float(np.mean([x[qn][dk][1] == x[qn]['gt'] for x in r])), 3),
                  'P(pen) img/native', round(float(np.mean([x[qn][dk][0] == x[qn]['stale'] for x in r])), 3),
                  round(float(np.mean([x[qn][dk][1] == x[qn]['stale'] for x in r])), 3), flush=True)


if __name__ == '__main__':
    main()
