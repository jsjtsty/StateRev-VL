#!/usr/bin/env python3
"""Addendum J data: colored full / colored teleport (swap period frozen on the
pre-swap frame; final layout appears abruptly) / identical full."""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render, sample_frames, LIFT, HOLD, LOWER, SWAP, PAUSE  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_reid'
COLORS = [(200, 80, 40), (60, 170, 60), (40, 200, 230)]   # BGR: blue, green, yellow
NAMES = ['blue', 'green', 'yellow']
PAIRS = [(0, 1), (1, 2), (0, 2)]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261005)
items = []
for j in range(300):
    k = 2 + j % 3
    init = int(rng.integers(3))
    sw = [PAIRS[int(rng.integers(3))] for _ in range(k)]
    perm = [int(x) for x in rng.permutation(3)]
    sc = Scene(3, init, sw, cup_colors=[COLORS[p] for p in perm])
    full = render(sc)
    t0, t1 = LIFT + HOLD + LOWER, LIFT + HOLD + LOWER + k * (SWAP + PAUSE)
    f0, f1 = int(math.floor(t0 * sc.fps)), int(math.ceil(t1 * sc.fps))
    tele = full.copy()
    tele[f0:f1] = full[f0]   # cups fully lowered, swap not yet started
    ident = render(Scene(3, init, sw))
    gt = sc.ball_trace()[-1]
    base = {'n_cups': 3, 'init': init, 'swaps': sw, 'k': k, 'transparent': False, 'sample_fps': 4.0,
            'swap_end_times': [t0 + i * (SWAP + PAUSE) + SWAP for i in range(k)],
            'labels': {'ground_truth': int(gt)}, 'agree': True, 'seq': j}
    for cond, fr, cols in (('Cfull', full, [NAMES[p] for p in perm]), ('Ctele', tele, [NAMES[p] for p in perm]),
                           ('Ifull', ident, None)):
        s = sample_frames(fr, sc.fps, 4.0)
        vid = f'{cond}_{j:03d}'
        np.savez_compressed(OUT / f'{vid}.npz', frames=s)
        r = dict(base, id=vid, set=cond, n_frames=int(len(s)), duration=sc.duration())
        if cols:
            r['cup_color_names'] = cols
        items.append(r)
with open(OUT / 'items.jsonl', 'w') as f:
    for r in items:
        f.write(json.dumps(r) + '\n')
print(len(items))
