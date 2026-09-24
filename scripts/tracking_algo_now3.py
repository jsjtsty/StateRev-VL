#!/usr/bin/env python3
"""Addendum N: minimal 'what is shown at the very end' displays (digit / ball)."""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'outputs/tracking_algo_v1/data_now3'
W, H, FPS = 448, 336, 8
POS = ['Left', 'Middle', 'Right']


def digit_frame(d):
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(img, str(d), (W // 2 - 55, H // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 12, cv2.LINE_AA)
    return img


def ball_frame(p):
    img = np.empty((H, W, 3), np.uint8); img[:] = (225, 215, 200)
    img[int(H * 0.72):] = (140, 100, 60)
    xs = np.linspace(W * 0.14, W * 0.86, 3)
    cv2.circle(img, (int(xs[p]), int(H * 0.72) - 22), 20, (220, 40, 40), -1, cv2.LINE_AA)
    return img


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261009)
    items = []
    for task in ('digit', 'ball'):
        for m in (1, 3):
            for dfin in (0.25, 0.5, 1.0, 2.0):
                for j in range(50):
                    n = 10 if task == 'digit' else 3
                    seq = [int(rng.integers(n))]
                    for _ in range(m):
                        seq.append(int(rng.choice([x for x in range(n) if x != seq[-1]])))
                    fr = digit_frame if task == 'digit' else ball_frame
                    parts = [np.repeat(fr(v)[None], FPS, 0) for v in seq[:-1]] + [np.repeat(fr(seq[-1])[None], int(round(dfin * FPS)), 0)]
                    frames = np.concatenate(parts)[::2]                       # 4 fps
                    vid = f'{task}_m{m}_f{dfin}_{j:02d}'
                    np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
                    it = {'id': vid, 'task': task, 'm': m, 'dfin': dfin, 'seq': seq, 'gt': seq[-1], 'pen': seq[-2],
                          'n_frames': int(len(frames)), 'sample_fps': 4.0}
                    if task == 'digit':
                        other = int(rng.choice([x for x in range(10) if x not in seq[-2:]]))
                        opts = [seq[-1], seq[-2], other]; rng.shuffle(opts); it['options'] = [int(x) for x in opts]
                    items.append(it)
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
        if it['task'] == 'digit':
            o = it['options']
            q = (f'The video shows a sequence of digits. What digit is shown at the very end of the video? '
                 f'(A) {o[0]} (B) {o[1]} (C) {o[2]}. Answer with only the letter.')
            ans = lambda k: o[k]
        else:
            q = ('The video shows a red ball that jumps between positions. Where is the ball at the very end of the video? '
                 '(A) Left (B) Middle (C) Right. Answer with only the letter.')
            ans = lambda k: k
        frames = np.load(OUT / f'{it["id"]}.npz')['frames']
        inp = build_inputs(cfg, proc, frames, q, it['sample_fps'])[0]
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        out.append(dict(it, pred=ans(int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC'])))))
    (ROOT / f'outputs/tracking_algo_v1/now3_{model_key}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for task in ('digit', 'ball'):
        for m in (1, 3):
            for dfin in (0.25, 0.5, 1.0, 2.0):
                r = [x for x in out if x['task'] == task and x['m'] == m and x['dfin'] == dfin]
                print(model_key, task, f'm={m} D_fin={dfin}', 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
                      'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    generate() if not a.model else evaluate(a.model, a.device_map)
