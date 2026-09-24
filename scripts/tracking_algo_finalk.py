#!/usr/bin/env python3
"""Final-state-vs-k set (Addendum I): 3 opaque cups, reveal + k uniformly
random swaps, k in {1,2,3,4}, 600 videos per k (adequately powered
last-token probes; earlier per-k analyses had n=60-90)."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_finalk'
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
PAIRS = [(0, 1), (1, 2), (0, 2)]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261004)
items = []
for k in (1, 2, 3, 4):
    for j in range(600):
        init = j % 3
        sw = [PAIRS[int(rng.integers(3))] for _ in range(k)]
        events = [('reveal',)] + [('swap', *p) for p in sw]
        s = init
        for a, b in sw:
            s = b if s == a else a if s == b else s
        sc = Scene(3, init, sw, cup_color=PALETTE[int(rng.integers(len(PALETTE)))])
        frames = sample_frames(render_events(sc, events), sc.fps, 4.0)
        segs, total = timeline(events)
        vid = f'FK{k}_{j:03d}'
        np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
        items.append({'id': vid, 'set': f'FK{k}', 'n_cups': 3, 'init': init, 'swaps': sw, 'k': k,
                      'swap_end_times': [t0 + SWAP for t0, _, e in segs if e[0] == 'swap'], 'duration': total,
                      'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
                      'labels': {'ground_truth': int(s)}, 'agree': True})
with open(OUT / 'items.jsonl', 'w') as f:
    for r in items:
        f.write(json.dumps(r) + '\n')
print(len(items))
