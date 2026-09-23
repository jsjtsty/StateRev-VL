#!/usr/bin/env python3
"""Binding set: exactly ONE swap moves the ball; distractor swaps (of the two
other cups) do not. Distinguishes "one effective update" from "only the first
swap event in the video is applied".
  B_ball_first   : reveal, BALL-swap, distractor
  B_dist_first   : reveal, distractor, BALL-swap
  B_dist2_first  : reveal, distractor, distractor, BALL-swap
150 videos each, 3 opaque cups. Final state = init moved by the single ball-swap.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

OUT = ROOT / 'outputs/tracking_algo_v1/data_binding'
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
PAT = {'B_ball_first': 'BD', 'B_dist_first': 'DB', 'B_dist2_first': 'DDB'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260927)
    items = []
    for cond, pat in PAT.items():
        for j in range(150):
            init = j % 3
            s = init
            events, swaps = [('reveal',)], []
            for p in pat:
                others = [c for c in range(3) if c != s]
                if p == 'B':
                    a, b = s, int(rng.choice(others))
                else:
                    a, b = others
                if rng.random() < .5:
                    a, b = b, a
                events.append(('swap', int(a), int(b))); swaps.append((int(a), int(b)))
                s = b if s == a else a if s == b else s
            sc = Scene(3, init, swaps, cup_color=PALETTE[int(rng.integers(len(PALETTE)))])
            frames = sample_frames(render_events(sc, events), sc.fps, 4.0)
            segs, total = timeline(events)
            vid = f'{cond}_{j:03d}'
            np.savez_compressed(OUT / f'{vid}.npz', frames=frames)
            items.append({'id': vid, 'set': cond, 'n_cups': 3, 'init': init, 'swaps': swaps, 'k': len(swaps),
                          'events': [list(e) for e in events], 'swap_end_times': [t0 + SWAP for t0, _, e in segs if e[0] == 'swap'],
                          'duration': total, 'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
                          'labels': {'ground_truth': int(s)}, 'agree': True})
    with open(OUT / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


if __name__ == '__main__':
    main()
