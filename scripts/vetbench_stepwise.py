#!/usr/bin/env python3
"""Stepwise state prompting on real VET-Bench cup videos (training-free).

Motivated by the finding that a single latent update works but chains do
not: the model is shown ONE swap clip at a time, told in text where the ball
is before the clip, and asked where it is after it. Its answer becomes the
state for the next clip. No swap labels are ever verbalised by the model.
Start state: given (init) or perceived from the intro clip (vis).
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, build_inputs, letter_ids  # noqa: E402
from tracking_algo_decompose import score, POS  # noqa: E402
from vetbench_decompose import read, VID  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/vetbench_method')
    a = ap.parse_args()
    beh = pd.read_csv(ROOT / 'outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv')
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    todev = lambda d: {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in d.items()}
    path = a.out / f'{a.model}_stepwise.jsonl'
    t0 = time.time()
    with open(path, 'w') as f:
        for n, (tid, g) in enumerate(beh.sort_values(['trajectory_id', 't']).groupby('trajectory_id')):
            frames = read(VID / f'{tid}.mp4')
            init = POS.index(g.initial_state.iloc[0].title())
            gt_states = [POS.index(x.title()) for x in g.gt_state]
            ends = list(g.frame_end)
            s, traj = init, []
            for e in ends:
                clip = frames[max(e - 66, 0):min(e + 6, len(frames))][::30 // 8]
                q = (f'The video shows 3 identical cups in a row. A red ball is hidden under the {POS[s]} cup. '
                     'Two of the cups swap their positions; the ball moves with its cup. '
                     'Where is the ball after the swap? (A) Left (B) Middle (C) Right. Answer with only the letter.')
                inp, _ = build_inputs(cfg, proc, clip, q, 8.0)
                s, _ = score(model, todev(inp), lid)
                traj.append(s)
            rec = {'id': tid, 'init': init, 'gt': gt_states[-1], 'final': s, 'traj': traj, 'gt_traj': gt_states,
                   'step_acc_given_correct_prev': None}
            f.write(json.dumps(rec) + '\n'); f.flush()
            if (n + 1) % 10 == 0:
                print(f'{a.model} {n + 1}/50 {time.time() - t0:.0f}s', flush=True)
    r = [json.loads(l) for l in open(path)]
    print('final acc', np.mean([x['final'] == x['gt'] for x in r]),
          'per-step acc', np.mean([p == q for x in r for p, q in zip(x['traj'], x['gt_traj'])]))
    print('DONE', a.model)


if __name__ == '__main__':
    main()
