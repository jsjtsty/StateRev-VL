#!/usr/bin/env python3
"""Interference set (Addendum E), 3 opaque cups, 150 videos each.
The target swap is a RANDOM pair (it may or may not hold the ball), so the
final state is a non-linear conjunction of the initial position and that
pair. Only what precedes it varies:
  E_R  : reveal, random                    (first swap of the video)
  E_WR : reveal, wait 1.25 s, random       (same timing as E_DR, no event)
  E_DR : reveal, distractor, random        (a swap of the two non-ball cups first)
(A first design with ball-always-moving swaps was discarded before any model
run: its answer is linearly recoverable from swap-involvement features.)
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render_events, sample_frames, timeline, SWAP  # noqa: E402

import os as _os
OUT = ROOT / ('outputs/tracking_algo_v1/data_interference' if _os.environ.get('SET', 'E') == 'E' else 'outputs/tracking_algo_v1/data_second')
PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]
import os
PAT = {'E_R': 'R', 'E_WR': 'WR', 'E_DR': 'DR'} if os.environ.get('SET', 'E') == 'E' else {'F_MVR': 'MVR', 'F_MR': 'MR', 'F_DVR': 'DVR'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260929 if _os.environ.get('SET', 'E') == 'E' else 20260930)
    items = []
    for cond, pat in PAT.items():
        for j in range(150):
            init = j % 3
            s = init
            events, swaps = [('reveal',)], []
            for p in pat:
                if p == 'W':
                    events.append(('wait', 1.25))
                    continue
                if p == 'V':
                    events.append(('reveal',))
                    continue
                others = [c for c in range(3) if c != s]
                if p == 'R':
                    a, b = (int(x) for x in rng.choice(3, 2, replace=False))
                elif p == 'M':
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
