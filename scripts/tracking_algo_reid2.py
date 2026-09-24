#!/usr/bin/env python3
"""Addendum K data: colored cups; five conditions differing only in the swap
period (full / steps / tele / telefinal / blank)."""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render, sample_frames, LIFT, HOLD, LOWER, SWAP, PAUSE  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_reid2'
COLORS = [(200, 80, 40), (60, 170, 60), (40, 200, 230)]   # BGR: blue, green, yellow
NAMES = ['blue', 'green', 'yellow']
PAIRS = [(0, 1), (1, 2), (0, 2)]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261006)
items = []
for j in range(300):
    k = 2 + j % 3
    init = int(rng.integers(3))
    sw = [PAIRS[int(rng.integers(3))] for _ in range(k)]
    perm = [int(x) for x in rng.permutation(3)]
    sc = Scene(3, init, sw, cup_colors=[COLORS[p] for p in perm])
    full = render(sc)
    fps, cyc = sc.fps, SWAP + PAUSE
    t0 = LIFT + HOLD + LOWER
    fi = lambda t: int(round(t * fps))
    f0, f1 = fi(t0), fi(t0 + k * cyc)
    V = {'full': full}
    steps = full.copy()
    for i in range(k):
        a, b = fi(t0 + i * cyc), fi(t0 + i * cyc + SWAP)
        steps[a:b] = full[a]
    V['steps'] = steps
    tele = full.copy(); tele[f0:f1] = full[f0]; V['tele'] = tele
    telefinal = full.copy(); telefinal[f0:f1] = full[f1]; V['telefinal'] = telefinal
    blank = full.copy()
    bg = np.empty_like(full[0]); bg[:] = sc.bg_color[::-1]
    bg[int(sc.height * 0.72):] = sc.table_color[::-1]
    blank[f0:f1] = bg; V['blank'] = blank
    gt = sc.ball_trace()[-1]
    base = {'n_cups': 3, 'init': init, 'swaps': sw, 'k': k, 'transparent': False, 'sample_fps': 4.0,
            'swap_end_times': [t0 + i * cyc + SWAP for i in range(k)], 'labels': {'ground_truth': int(gt)},
            'agree': True, 'seq': j, 'cup_color_names': [NAMES[p] for p in perm], 'duration': sc.duration()}
    for cond, fr in V.items():
        s = sample_frames(fr, fps, 4.0)
        vid = f'{cond}_{j:03d}'
        np.savez_compressed(OUT / f'{vid}.npz', frames=s)
        items.append(dict(base, id=vid, set=cond, n_frames=int(len(s))))
with open(OUT / 'items.jsonl', 'w') as f:
    for r in items:
        f.write(json.dumps(r) + '\n')
print(len(items))
