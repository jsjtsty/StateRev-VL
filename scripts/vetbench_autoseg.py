#!/usr/bin/env python3
"""Perceive-then-reason on real VET-Bench cup videos WITHOUT metadata.

Segmentation is model-free and label-free: the mean absolute frame
difference (5-frame smoothed) is computed, its local minima (prominence
>= 5% of max, >= 1 s apart) plus the onset/offset of motion (energy above
25% of max) delimit motion segments. Every segment is shown to the VLM,
which classifies it as one of three swaps or "no swap" (cups lifted/lowered
or no movement). The number and order of swaps therefore come from the
model, not from annotations. The start cup is perceived from everything
before the first segment the model calls a swap (or given, 'init' variant).
Composition: D1 exact rule (upper bound) and D3 written reasoning (method).
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, build_inputs, letter_ids, parse_letter  # noqa: E402
from tracking_algo_decompose import text_inputs, POS, PAIRS, PAIR_TXT  # noqa: E402
from vetbench_decompose import read, VID  # noqa: E402

FPS = 30


def segments(frames, min_sep_s=1.0):
    g = frames.astype(np.float32).mean(-1)
    e = np.abs(np.diff(g, axis=0)).mean((1, 2))
    e = np.convolve(e, np.ones(5) / 5, 'same')
    thr = 0.25 * e.max()
    mins, _ = find_peaks(-e, distance=int(min_sep_s * FPS), prominence=0.05 * e.max())
    moving = e > thr
    on = [i for i in range(1, len(e)) if moving[i] and not moving[i - 1]]
    off = [i for i in range(1, len(e)) if not moving[i] and moving[i - 1]]
    b = sorted(set([0, len(e)] + list(mins) + on + off))
    segs = [(s, t) for s, t in zip(b[:-1], b[1:]) if t - s >= 10 and e[s:t].mean() > thr]
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', default='cup', choices=['cup', 'card'])
    ap.add_argument('--min-sep', type=float, default=1.0, help='minimum spacing (s) between motion minima')
    ap.add_argument('--min-dur', type=float, default=0.5, help='segments shorter than this (s) are not swaps')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/vetbench_method')
    a = ap.parse_args()
    if a.task == 'cup':
        beh = pd.read_csv(ROOT / 'outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv')
        videos = [(tid, POS.index(g.initial_state.iloc[0].title()), POS.index(g.gt_state.iloc[-1].title()), list(g.gt_event))
                  for tid, g in beh.sort_values(['trajectory_id', 't']).groupby('trajectory_id')]
    else:
        meta = json.load(open(ROOT / 'dataset/vetbench/card/card.json'))
        videos = []
        for m in meta:
            pos = lambda arr: next(x['position'] for x in arr if x['card'] == 'Q\u2665') - 1
            seq = [pos(m['initial'])] + [pos(st) for st in m['intermediate']]
            arr = [m['initial']] + m['intermediate']
            ev = []
            for u, v in zip(arr[:-1], arr[1:]):
                pu = {x['card']: x['position'] - 1 for x in u}; pv = {x['card']: x['position'] - 1 for x in v}
                moved = sorted({pu[c] for c in pu if pu[c] != pv[c]})
                ev.append(PAIR_TXT[PAIRS.index(tuple(moved))] if len(moved) == 2 else 'none')
            videos.append((m['video'][:-4], seq[0], seq[-1], ev))
    vdir = ROOT / f'dataset/vetbench/{a.task}'
    obj = 'red ball' if a.task == 'cup' else 'Queen of Hearts'
    items_word = 'cups' if a.task == 'cup' else 'face-down cards'
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid2 = letter_ids(proc.tokenizer, 'AB')
    lid3 = letter_ids(proc.tokenizer, 'ABC')
    todev = lambda d: {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in d.items()}

    def ask(clip, q, lid, letters):
        inp, _ = build_inputs(cfg, proc, clip, q, 8.0)
        with torch.inference_mode():
            lg = model(**todev(inp), logits_to_keep=1).logits[0, -1].float()
        sc = {l: float(max(lg[i] for i in lid[l])) for l in letters}
        return letters.index(max(sc, key=sc.get))

    path = a.out / f'{a.model}_autoseg2{"" if a.task == "cup" else "_card"}{"" if a.min_sep == 1.0 else f"_sep{a.min_sep}"}.jsonl'
    t0 = time.time()
    with open(path, 'w') as f:
        for n, (tid, init, gt, true_events) in enumerate(videos):
            frames = read(vdir / f'{tid}.mp4')
            segs = segments(frames, a.min_sep)
            labels = []
            w = 'cups' if a.task == 'cup' else 'cards'
            for s, t in segs:
                if t - s < a.min_dur * FPS:
                    labels.append(3)            # too short to contain a swap
                    continue
                clip = frames[max(s - 6, 0):min(t + 6, len(frames))][::FPS // 8]
                q1 = (f'The video shows 3 {items_word} in a row. Do two of the {w} swap their positions in this clip? '
                      f'(A) Yes (B) No. Answer with only the letter.')
                if ask(clip, q1, lid2, 'AB') == 1:
                    labels.append(3)
                    continue
                q2 = (f'The video shows 3 {items_word} in a row. Two of the {w} swap their positions. '
                      f'Which two positions were swapped? (A) Left and Middle (B) Middle and Right (C) Left and Right. '
                      'Answer with only the letter.')
                labels.append(ask(clip, q2, lid3, 'ABC'))
            swaps = [l for l in labels if l < 3]
            first = next((i for i, l in enumerate(labels) if l < 3), len(segs))
            intro_end = segs[first][0] if first < len(segs) else len(frames)
            intro = frames[:max(intro_end, 8)][::FPS // 8]
            if a.task == 'cup':
                qi = ('The video shows 3 cups; the cups are lifted to show a red ball, then lowered. '
                      'Under which cup is the ball? (A) Left (B) Middle (C) Right. Answer with only the letter.')
            else:
                qi = ('The video shows 3 playing cards face up, which are then turned face down. '
                      'Where is the Queen of Hearts? (A) Left (B) Middle (C) Right. Answer with only the letter.')
            init_p = ask(intro, qi, lid3, 'ABC')
            rec = {'id': tid, 'init': init, 'gt': gt, 'init_perceived': init_p, 'segments': segs, 'seg_labels': labels,
                   'n_swaps_found': len(swaps), 'swaps': swaps, 'true_events': true_events,
                   'swaps_exact': ([PAIR_TXT[p] for p in swaps] == true_events) if true_events else None}
            for tag, start in (('init', init), ('vis', init_p)):
                s = start
                for p in swaps:
                    x, y = PAIRS[p]; s = y if s == x else x if s == y else s
                rec[f'd1_{tag}'] = s
                steps = ', then '.join(f'the {PAIR_TXT[p]} cups were swapped' for p in swaps) or 'no cups were swapped'
                if a.task == 'cup':
                    head = (f'A ball is under the {POS[start]} cup of three cups (Left, Middle, Right). Then {steps}. '
                            'When two cups are swapped, the ball moves with its cup. Which cup contains the ball at the end? ')
                else:
                    steps = steps.replace('cups', 'cards')
                    head = (f'The Queen of Hearts is the {POS[start]} card of three face-down cards (Left, Middle, Right). Then {steps}. '
                            'When two cards are swapped, each card moves to the other position. Where is the Queen of Hearts at the end? ')
                q3 = (head +
                      '(A) Left (B) Middle (C) Right. Track it step by step, then give the final answer as a single letter on the last line.')
                inp3 = todev(text_inputs(cfg, proc, q3))
                with torch.inference_mode():
                    g3 = model.generate(**inp3, max_new_tokens=500, do_sample=False)
                h = parse_letter(proc.tokenizer.decode(g3[0, inp3['input_ids'].shape[1]:], skip_special_tokens=True), 'ABC')
                rec[f'd3_{tag}'] = 'ABC'.index(h[-1]) if h else -1
            f.write(json.dumps(rec, default=int) + '\n'); f.flush()
            if (n + 1) % 10 == 0:
                print(f'{a.model} {n + 1}/50 {time.time() - t0:.0f}s', flush=True)
    r = [json.loads(l) for l in open(path)]
    summ = {k: float(np.mean([x[k] == x['gt'] for x in r])) for k in ('d1_init', 'd3_init', 'd1_vis', 'd3_vis')}
    summ.update({'swap_sequence_exact': float(np.mean([bool(x['swaps_exact']) for x in r])) if r[0]['swaps_exact'] is not None else None,
                 'n_swaps_found_eq5': float(np.mean([x['n_swaps_found'] == 5 for x in r])),
                 'init_perceived_acc': float(np.mean([x['init_perceived'] == x['init'] for x in r]))})
    print(json.dumps(summ))
    print('DONE', a.model)


if __name__ == '__main__':
    main()
