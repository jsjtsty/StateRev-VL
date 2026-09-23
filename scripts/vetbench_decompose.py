#!/usr/bin/env python3
"""Perceive-then-reason on REAL VET-Bench cup videos (training-free method test).

For each of the 50 cup videos (3 cups, 5 swaps, 2 s per swap):
  native      : whole video (4 fps) + one question, option-letter logits;
                'init' prompt (start cup given in text) and 'vis' prompt.
  perceive    : each swap window (frame_end_t - 60 .. frame_end_t, +/- 6
                frames, 8 fps) -> "which two positions were swapped?";
                start cup perceived from the intro clip (for the vis variant).
  D1          : perceived swaps composed with the exact rule (upper bound);
  D3 (method) : perceived swaps written as text, the model composes them with
                written step-by-step reasoning (no rule given in code).
Swap windows come from the existing behavior table (frame_end per t).
"""
from pathlib import Path
import argparse
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, build_inputs, letter_ids, parse_letter  # noqa: E402
from tracking_algo_decompose import text_inputs, score, POS, PAIRS, PAIR_TXT  # noqa: E402

VID = ROOT / 'dataset/vetbench/cup'


def read(path):
    cap = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/vetbench_method')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    beh = pd.read_csv(ROOT / 'outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv')
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    todev = lambda d: {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in d.items()}
    path = a.out / f'{a.model}.jsonl'
    done = {json.loads(l)['id'] for l in open(path)} if path.exists() else set()
    t0 = time.time()
    with open(path, 'a') as f:
        for n, (tid, g) in enumerate(beh.sort_values(['trajectory_id', 't']).groupby('trajectory_id')):
            if tid in done:
                continue
            frames = read(VID / f'{tid}.mp4')
            init = POS.index(g.initial_state.iloc[0].title())
            gt = POS.index(g.gt_state.iloc[-1].title())
            ends = list(g.frame_end)
            rec = {'id': tid, 'init': init, 'gt': gt, 'true_events': list(g.gt_event)}
            full = frames[::30 // 4]
            for variant in ('init', 'vis'):
                q = ('The video shows a shell game with 3 identical cups. A red ball is placed under one cup, '
                     'then the cups are swapped several times. ')
                if variant == 'init':
                    q += f'At the start, the ball is under the {POS[init]} cup. '
                q += 'Which cup contains the ball at the end of the video? (A) Left (B) Middle (C) Right. Answer with only the letter.'
                inp, _ = build_inputs(cfg, proc, full, q, 4.0)
                rec[f'native_{variant}'], _ = score(model, todev(inp), lid)
            intro = frames[:max(ends[0] - 60, 8)][::30 // 8]
            qi = ('The video shows 3 cups; the cups are lifted to show a red ball, then lowered. '
                  'Under which cup is the ball? (A) Left (B) Middle (C) Right. Answer with only the letter.')
            inp, _ = build_inputs(cfg, proc, intro, qi, 8.0)
            rec['init_perceived'], _ = score(model, todev(inp), lid)
            perceived = []
            for e in ends:
                clip = frames[max(e - 66, 0):min(e + 6, len(frames))][::30 // 8]
                qs = ('The video shows 3 identical cups in a row. Two of the cups swap their positions. '
                      'Which two positions were swapped? (A) Left and Middle (B) Middle and Right (C) Left and Right. '
                      'Answer with only the letter.')
                inp, _ = build_inputs(cfg, proc, clip, qs, 8.0)
                p, _ = score(model, todev(inp), lid)
                perceived.append(p)
            rec['perceived'] = perceived
            rec['event_acc'] = float(np.mean([PAIR_TXT[p] == e for p, e in zip(perceived, rec['true_events'])]))
            for tag, start in (('init', init), ('vis', rec['init_perceived'])):
                s = start
                for p in perceived:
                    x, y = PAIRS[p]; s = y if s == x else x if s == y else s
                rec[f'd1_{tag}'] = s
                steps = ', then '.join(f'the {PAIR_TXT[p]} cups were swapped' for p in perceived)
                q3 = (f'A ball is under the {POS[start]} cup of three cups (Left, Middle, Right). Then {steps}. '
                      'When two cups are swapped, the ball moves with its cup. Which cup contains the ball at the end? '
                      '(A) Left (B) Middle (C) Right. Track the ball step by step, then give the final answer as a single letter on the last line.')
                inp3 = todev(text_inputs(cfg, proc, q3))
                with torch.inference_mode():
                    g3 = model.generate(**inp3, max_new_tokens=500, do_sample=False)
                h = parse_letter(proc.tokenizer.decode(g3[0, inp3['input_ids'].shape[1]:], skip_special_tokens=True), 'ABC')
                rec[f'd3_{tag}'] = 'ABC'.index(h[-1]) if h else -1
            f.write(json.dumps(rec) + '\n'); f.flush()
            if (n + 1) % 10 == 0:
                print(f'{a.model} {n + 1}/50 {time.time() - t0:.0f}s', flush=True)
    print('DONE', a.model)


if __name__ == '__main__':
    main()
