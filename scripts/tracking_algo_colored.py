#!/usr/bin/env python3
"""Colored-cups set: same sequences as opaque3, but each cup has a distinct
color (blue / green / yellow), so the task is solvable by binding the ball to
a cup color at the start and retrieving that cup at the end, with no need to
compose swaps. Written to data/ as set 'colored3' (appended to items.jsonl)."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render, sample_frames  # noqa: E402

DATA = ROOT / 'outputs/tracking_algo_v1/data'
COLORS = [(200, 80, 40), (60, 170, 60), (40, 200, 230)]   # BGR: blue, green, yellow
NAMES = ['blue', 'green', 'yellow']
items = [json.loads(l) for l in open(DATA / 'items.jsonl')]
assert not any(r['set'] == 'colored3' for r in items), 'colored3 already generated'
rng = np.random.default_rng(20260925)
new = []
for r in items:
    if r['set'] != 'opaque3':
        continue
    perm = [int(x) for x in rng.permutation(3)]
    cols = [COLORS[p] for p in perm]
    sc = Scene(3, r['init'], [tuple(s) for s in r['swaps']], cup_colors=cols)
    frames = sample_frames(render(sc), sc.fps, 4.0)
    vid = r['id'].replace('opaque3', 'colored3')
    np.savez_compressed(DATA / f'{vid}.npz', frames=frames)
    q = dict(r, id=vid, set='colored3', cup_color_names=[NAMES[p] for p in perm])
    new.append(q)
with open(DATA / 'items.jsonl', 'a') as f:
    for q in new:
        f.write(json.dumps(q) + '\n')
print(len(new), 'colored3 items appended')
