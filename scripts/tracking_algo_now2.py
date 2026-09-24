#!/usr/bin/env python3
"""Addendum M: number of prior layouts (m) x final dwell x input mode."""
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

OUT = ROOT / 'outputs/tracking_algo_v1/data_now2'
COLORS = {'blue': (200, 80, 40), 'green': (60, 170, 60), 'yellow': (40, 200, 230)}
NAMES = ['blue', 'green', 'yellow']
POS = ['left', 'middle', 'right']
PERMS = list(itertools.permutations(range(3)))
FPS = 8


def static(layout):
    return render(Scene(3, 0, [], cup_colors=[COLORS[NAMES[c]] for c in layout]))


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261008)
    items = []
    for m in (1, 2, 4):
        for dfin in (0.5, 1.0, 2.0):
            for j in range(60):
                while True:
                    L = [PERMS[int(rng.integers(6))] for _ in range(m + 1)]
                    p = int(rng.integers(3))
                    if all(L[i] != L[i + 1] for i in range(m)) and L[-2][p] != L[-1][p]:
                        break
                seq = [static(L[0])] + [np.repeat(static(L[i])[-1:], int(round(1.0 * FPS)), 0) for i in range(1, m)]
                seq.append(np.repeat(static(L[m])[-1:], int(round(dfin * FPS)), 0))
                frames = sample_frames(np.concatenate(seq), FPS, 4.0)
                vid = f'M{m}_f{dfin}_{j:02d}'
                np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
                items.append({'id': vid, 'm': m, 'dfin': dfin, 'p': p, 'layouts': [list(x) for x in L],
                              'gt': L[-1][p], 'pen': L[-2][p], 'n_frames': int(len(frames)), 'sample_fps': 4.0})
    with open(OUT / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def inputs_images(cfg, proc, frames, q, fps):
    content = []
    for i in range(len(frames)):
        content += [{'type': 'text', 'text': f'Frame {i + 1} ({i / fps:.2f} s):'}, {'type': 'image'}]
    content.append({'type': 'text', 'text': q})
    msgs = [{'role': 'user', 'content': content}]
    tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw)
    return proc(text=[text], images=list(frames), return_tensors='pt')


def evaluate(model_key, device_map, mode):
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    items = [json.loads(l) for l in open(OUT / 'items.jsonl')]
    cfg, proc, model = load(model_key, device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    out = []
    for it in items:
        what = 'video' if mode == 'video' else 'sequence of video frames'
        q = (f'The {what} shows three cups of different colors (blue, green, yellow). At the very end of the {what}, '
             f'what color is the cup in the {POS[it["p"]]} position? (A) blue (B) green (C) yellow. Answer with only the letter.')
        frames = np.load(OUT / f'{it["id"]}.npz')['frames']
        inp = build_inputs(cfg, proc, frames, q, it['sample_fps'])[0] if mode == 'video' else inputs_images(cfg, proc, frames, q, it['sample_fps'])
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        out.append(dict(it, pred=int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))))
    (ROOT / f'outputs/tracking_algo_v1/now2_{model_key}_{mode}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for m in (1, 2, 4):
        for dfin in (0.5, 1.0, 2.0):
            r = [x for x in out if x['m'] == m and x['dfin'] == dfin]
            print(model_key, mode, f'm={m} D_fin={dfin}', 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
                  'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--mode', default='video', choices=['video', 'images'])
    a = ap.parse_args()
    generate() if not a.model else evaluate(a.model, a.device_map, a.mode)
