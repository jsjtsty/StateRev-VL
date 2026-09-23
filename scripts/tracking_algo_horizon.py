#!/usr/bin/env python3
"""Horizon set (G2): how many occluded updates since the ball was last SEEN
can the model carry?

Conditions (3 cups, opaque, 90 videos each, init balanced):
  K3_m3 : reveal, s, s, s                      (3 swaps since last sighting)
  K3_m2 : reveal, s, reveal, s, s              (2)
  K3_m1 : reveal, s, s, reveal, s              (1)
  K4_m4 : reveal, s, s, s, s                   (4)
  K4_m1 : reveal, s, s, s, reveal, s           (1)
  K1_w0 / K1_w2 / K1_w4 : reveal, wait 0/2/4 s, s   (1 swap, varying delay)
A mid-sequence reveal lifts all cups (ball visible), holds, lowers.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_horizon'
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
PAIRS3 = [(0, 1), (1, 2), (0, 2)]
COND = {
    'K3_m3': ['S', 'S', 'S'], 'K3_m2': ['S', 'R', 'S', 'S'], 'K3_m1': ['S', 'S', 'R', 'S'],
    'K4_m4': ['S', 'S', 'S', 'S'], 'K4_m1': ['S', 'S', 'S', 'R', 'S'],
    'K1_w0': ['S'], 'K1_w2': [('wait', 2.0), 'S'], 'K1_w4': [('wait', 4.0), 'S'],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260926)
    items = []
    for cond, pattern in COND.items():
        for j in range(90):
            init = j % 3
            events, swaps = [('reveal',)], []
            for p in pattern:
                if p == 'S':
                    a, b = PAIRS3[int(rng.integers(3))]
                    if rng.random() < .5:
                        a, b = b, a
                    events.append(('swap', a, b)); swaps.append((a, b))
                elif p == 'R':
                    events.append(('reveal',))
                else:
                    events.append(p)
            sc = Scene(3, init, swaps, cup_color=PALETTE[int(rng.integers(len(PALETTE)))])
            frames = sample_frames(render_events(sc, events), sc.fps, 4.0)
            segs, total = timeline(events)
            swap_end = [t0 + SWAP for t0, _, ev in segs if ev[0] == 'swap']
            vid = f'{cond}_{j:03d}'
            np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
            s = init
            for a, b in swaps:
                s = b if s == a else a if s == b else s
            items.append({'id': vid, 'set': cond, 'n_cups': 3, 'init': init, 'swaps': swaps, 'k': len(swaps),
                          'events': [list(e) for e in events], 'swap_end_times': swap_end, 'duration': total,
                          'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
                          'labels': {'ground_truth': int(s)}, 'agree': True})
    with open(OUT / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items), 'items')


if __name__ == '__main__':
    main()
