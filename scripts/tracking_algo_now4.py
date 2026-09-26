#!/usr/bin/env python3
"""Addendum O: n colored disks in 3 slots, abrupt layout changes; query what is
at position p (Q-pos) or where a given disk is (Q-obj) at the very end."""
from pathlib import Path
import argparse
import itertools
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'outputs/tracking_algo_v1/data_now4'
W, H, FPS = 448, 336, 8
POS = ['left', 'middle', 'right']
COLS = ['red', 'blue', 'green']
RGB = {'red': (220, 40, 40), 'blue': (40, 80, 220), 'green': (40, 170, 60)}


def frame(layout):   # layout[slot] = color index or -1
    img = np.empty((H, W, 3), np.uint8); img[:] = (225, 215, 200)
    img[int(H * 0.72):] = (140, 100, 60)
    xs = np.linspace(W * 0.14, W * 0.86, 3)
    for s, c in enumerate(layout):
        if c >= 0:
            cv2.circle(img, (int(xs[s]), int(H * 0.72) - 36), 34, RGB[COLS[c]], -1, cv2.LINE_AA)
    return img


def layouts(n):
    out = []
    for slots in itertools.permutations(range(3), n):
        L = [-1, -1, -1]
        for c, s in enumerate(slots):
            L[s] = c
        out.append(tuple(L))
    return out


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261010)
    items = []
    for n in (1, 2, 3):
        LS = layouts(n)
        for dfin in (0.5, 2.0):
            for query in ('pos', 'obj'):
                for j in range(50):
                    while True:
                        seq = [LS[int(rng.integers(len(LS)))] for _ in range(4)]
                        if any(seq[i] == seq[i + 1] for i in range(3)):
                            continue
                        if query == 'pos':
                            cand = [p for p in range(3) if seq[2][p] != seq[3][p]]
                        else:
                            cand = [c for c in range(n) if seq[2].index(c) != seq[3].index(c)]
                        if cand:
                            t = int(cand[int(rng.integers(len(cand)))])
                            break
                    parts = [np.repeat(frame(L)[None], FPS, 0) for L in seq[:3]] + [np.repeat(frame(seq[3])[None], int(round(dfin * FPS)), 0)]
                    frames = np.concatenate(parts)[::2]
                    vid = f'n{n}_f{dfin}_{query}_{j:02d}'
                    np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
                    if query == 'pos':
                        gt, pen = seq[3][t], seq[2][t]
                    else:
                        gt, pen = seq[3].index(t), seq[2].index(t)
                    items.append({'id': vid, 'n': n, 'dfin': dfin, 'query': query, 'target': t, 'seq': [list(x) for x in seq],
                                  'gt': int(gt), 'pen': int(pen), 'n_frames': int(len(frames)), 'sample_fps': 4.0})
    with open(OUT / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def generate_ref():
    """Addendum U disks: query the position where colour c was at the beginning."""
    D = ROOT / 'outputs/tracking_algo_v1/data_now5'
    D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261013)
    LS = layouts(3)
    items = []
    for dfin in (0.5, 2.0):
        j = 0
        while j < 100:
            seq = [LS[int(rng.integers(len(LS)))] for _ in range(4)]
            if any(seq[i] == seq[i + 1] for i in range(3)):
                continue
            ok = [(c, seq[0].index(c)) for c in range(3)
                  if seq[3][seq[0].index(c)] != seq[2][seq[0].index(c)] and c not in (seq[3][seq[0].index(c)], seq[2][seq[0].index(c)])]
            if not ok:
                continue
            c, p = ok[int(rng.integers(len(ok)))]
            parts = [np.repeat(frame(L)[None], FPS, 0) for L in seq[:3]] + [np.repeat(frame(seq[3])[None], int(round(dfin * FPS)), 0)]
            fr = np.concatenate(parts)[::2]
            vid = f'R_f{dfin}_{j:03d}'
            np.savez_compressed(D / f'{vid}.npz', frames=fr)
            items.append({'id': vid, 'dfin': dfin, 'ref_color': int(c), 'p': int(p), 'seq': [list(x) for x in seq],
                          'gt': int(seq[3][p]), 'pen': int(seq[2][p]), 'n_frames': int(len(fr)), 'sample_fps': 4.0})
            j += 1
    with open(D / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def evaluate(model_key, device_map):
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    items = [json.loads(l) for l in open(OUT / 'items.jsonl')]
    cfg, proc, model = load(model_key, device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABCD')
    out = []
    for it in items:
        if it['query'] == 'pos':
            opts = COLS[:it['n']] + (['nothing'] if it['n'] < 3 else [])
            vals = list(range(it['n'])) + ([-1] if it['n'] < 3 else [])
            letters = 'ABCD'[:len(opts)]
            q = (f'The video shows colored disks in three positions (left, middle, right) that are rearranged several times. '
                 f'At the very end of the video, what is in the {POS[it["target"]]} position? '
                 + ' '.join(f'({l}) {o}' for l, o in zip(letters, opts)) + '. Answer with only the letter.')
        else:
            letters, vals = 'ABC', [0, 1, 2]
            q = (f'The video shows colored disks in three positions that are rearranged several times. '
                 f'At the very end of the video, where is the {COLS[it["target"]]} disk? (A) Left (B) Middle (C) Right. '
                 'Answer with only the letter.')
        frames = np.load(OUT / f'{it["id"]}.npz')['frames']
        inp = build_inputs(cfg, proc, frames, q, it['sample_fps'])[0]
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        k = int(np.argmax([float(max(lg[i] for i in lid[l])) for l in letters]))
        out.append(dict(it, pred=vals[k]))
    (ROOT / f'outputs/tracking_algo_v1/now4_{model_key}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for query in ('pos', 'obj'):
        for n in (1, 2, 3):
            for dfin in (0.5, 2.0):
                r = [x for x in out if x['n'] == n and x['dfin'] == dfin and x['query'] == query]
                print(model_key, query, f'n={n} D_fin={dfin}', 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
                      'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0'); ap.add_argument('--ref', action='store_true')
    a = ap.parse_args()
    if '--ref' in sys.argv:
        generate_ref()
    else:
        generate() if not a.model else evaluate(a.model, a.device_map)
