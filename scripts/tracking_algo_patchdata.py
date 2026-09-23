#!/usr/bin/env python3
"""Patch set (Addendum G): 3 opaque cups, reveal + ONE uniformly random swap,
identical timing for every video (so token positions match across videos).
n=480; first 240 = probe-training split, last 240 = patching test split."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_patch'
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261002)
items = []
for j in range(480):
    init = j % 3
    a, b = (int(x) for x in rng.choice(3, 2, replace=False))
    events = [('reveal',), ('swap', a, b)]
    s = b if init == a else a if init == b else init
    sc = Scene(3, init, [(a, b)], cup_color=PALETTE[int(rng.integers(len(PALETTE)))])
    frames = sample_frames(render_events(sc, events), sc.fps, 4.0)
    segs, total = timeline(events)
    vid = f'P1_{j:03d}'
    np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
    items.append({'id': vid, 'set': 'P1', 'split': 'train' if j < 240 else 'test', 'n_cups': 3, 'init': init,
                  'swaps': [(a, b)], 'k': 1, 'swap_end_times': [segs[1][0] + SWAP], 'duration': total,
                  'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
                  'labels': {'ground_truth': int(s)}, 'agree': True})
with open(OUT / 'items.jsonl', 'w') as f:
    for r in items:
        f.write(json.dumps(r) + '\n')
print(len(items), set(r['n_frames'] for r in items))
