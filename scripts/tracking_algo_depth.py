#!/usr/bin/env python3
"""Depth set (exploratory, for the open 'when do updates fail' question):
3 opaque cups, reveal + 4 uniformly random swaps, n=900 (larger n so S_j
decodability can be split by the ball's path)."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_depth'
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261001)
items = []
for j in range(900):
    init = j % 3
    events, swaps, s = [('reveal',)], [], init
    for _ in range(4):
        a, b = (int(x) for x in rng.choice(3, 2, replace=False))
        events.append(('swap', a, b)); swaps.append((a, b))
        s = b if s == a else a if s == b else s
    sc = Scene(3, init, swaps, cup_color=PALETTE[int(rng.integers(len(PALETTE)))])
    frames = sample_frames(render_events(sc, events), sc.fps, 4.0)
    segs, total = timeline(events)
    vid = f'D4_{j:03d}'
    np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
    items.append({'id': vid, 'set': 'D4', 'n_cups': 3, 'init': init, 'swaps': swaps, 'k': 4,
                  'swap_end_times': [t0 + SWAP for t0, _, e in segs if e[0] == 'swap'], 'duration': total,
                  'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
                  'labels': {'ground_truth': int(s)}, 'agree': True})
with open(OUT / 'items.jsonl', 'w') as f:
    for r in items:
        f.write(json.dumps(r) + '\n')
print(len(items))
