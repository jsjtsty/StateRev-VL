#!/usr/bin/env python3
"""Patch set for two swaps (Addendum H): 3 opaque cups, reveal + swap1 +
swap2 (independent uniform pairs), identical timing. n=960: first 480 fit
probes, last 480 are patching targets."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_patch2'
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
PAIRS = [(0, 1), (1, 2), (0, 2)]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261003)
items = []
for j in range(960):
    init = j % 3
    p1, p2 = PAIRS[int(rng.integers(3))], PAIRS[int(rng.integers(3))]
    events = [('reveal',), ('swap', *p1), ('swap', *p2)]
    s1 = p1[1] if init == p1[0] else p1[0] if init == p1[1] else init
    s2 = p2[1] if s1 == p2[0] else p2[0] if s1 == p2[1] else s1
    sc = Scene(3, init, [p1, p2], cup_color=PALETTE[int(rng.integers(len(PALETTE)))])
    frames = sample_frames(render_events(sc, events), sc.fps, 4.0)
    segs, total = timeline(events)
    vid = f'P2_{j:03d}'
    np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
    items.append({'id': vid, 'set': 'P2', 'split': 'train' if j < 480 else 'test', 'n_cups': 3, 'init': init,
                  'swaps': [p1, p2], 'k': 2, 'swap_end_times': [segs[1][0] + SWAP, segs[2][0] + SWAP],
                  'duration': total, 'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
                  'labels': {'ground_truth': int(s2), 'S1': int(s1), 'pair1': PAIRS.index(p1), 'pair2': PAIRS.index(p2)},
                  'agree': True})
with open(OUT / 'items.jsonl', 'w') as f:
    for r in items:
        f.write(json.dumps(r) + '\n')
print(len(items), set(r['n_frames'] for r in items))
