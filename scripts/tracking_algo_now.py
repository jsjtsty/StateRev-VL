#!/usr/bin/env python3
"""Addendum L: dwell time vs recency. Colored cups; reveal, then static
layouts L1 (1 s), L2 (D_pen), L3 (D_fin) with abrupt changes; ask the color
at position p at the very end. Generates data_now/ and, with --model, evaluates."""
from pathlib import Path
import argparse
import itertools
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from staterev.shellsim import Scene, render, sample_frames  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_now'
COLORS = {'blue': (200, 80, 40), 'green': (60, 170, 60), 'yellow': (40, 200, 230)}
NAMES = ['blue', 'green', 'yellow']
POS = ['left', 'middle', 'right']
PERMS = list(itertools.permutations(range(3)))
FPS = 8


def static(layout):
    fr = render(Scene(3, 0, [], cup_colors=[COLORS[NAMES[c]] for c in layout]))
    return fr


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261007)
    items = []
    for dpen in (0.5, 2.0):
        for dfin in (0.5, 1.0, 2.0, 4.0):
            for j in range(60):
                while True:
                    L = [PERMS[int(rng.integers(6))] for _ in range(4)]
                    p = int(rng.integers(3))
                    if all(L[i] != L[i + 1] for i in range(3)) and L[2][p] != L[3][p]:
                        break
                rev = static(L[0])                      # reveal + lowered static tail
                seq = [rev] + [np.repeat(static(L[i])[-1:], int(round(d * FPS)), 0) for i, d in ((1, 1.0), (2, dpen), (3, dfin))]
                frames = sample_frames(np.concatenate(seq), FPS, 4.0)
                vid = f'N_p{dpen}_f{dfin}_{j:02d}'
                np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
                items.append({'id': vid, 'set': f'p{dpen}_f{dfin}', 'dpen': dpen, 'dfin': dfin, 'p': p,
                              'layouts': [list(x) for x in L], 'gt': L[3][p], 'pen': L[2][p], 'first': L[1][p],
                              'n_frames': int(len(frames)), 'sample_fps': 4.0})
    with open(OUT / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def evaluate(model_key, device_map):
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    items = [json.loads(l) for l in open(OUT / 'items.jsonl')]
    cfg, proc, model = load(model_key, device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    out = []
    for it in items:
        q = (f'The video shows three cups of different colors (blue, green, yellow). At the very end of the video, '
             f'what color is the cup in the {POS[it["p"]]} position? (A) blue (B) green (C) yellow. Answer with only the letter.')
        frames = np.load(OUT / f'{it["id"]}.npz')['frames']
        inp, _ = build_inputs(cfg, proc, frames, q, it['sample_fps'])
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        pred = int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))
        out.append(dict(it, pred=pred))
    (ROOT / f'outputs/tracking_algo_v1/now_{model_key}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for dpen in (0.5, 2.0):
        for dfin in (0.5, 1.0, 2.0, 4.0):
            r = [x for x in out if x['dpen'] == dpen and x['dfin'] == dfin]
            print(model_key, f'D_pen={dpen} D_fin={dfin}', 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
                  'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    generate() if not a.model else evaluate(a.model, a.device_map)
