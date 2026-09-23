#!/usr/bin/env python3
"""Generate the G1 controlled shell-game set (see outputs/tracking_algo_v1/PREREG.md).

Sets
  opaque3   : 3 cups, opaque, k in 2..6, per k half "shortcut == GT", half "shortcut != GT"
  transp3   : the exact same sequences as opaque3, transparent cups (observability control)
  k1        : 3 cups, opaque, single swap (perception sanity; shortcut == GT by construction)
  opaque4/5 : 4 / 5 cups, opaque, k in {3,5}, same agree/disagree balancing
Each item stores 4-fps sampled RGB frames (npz) plus metadata (jsonl).
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.shellsim import Scene, render, sample_frames  # noqa: E402
from staterev.shortcuts import FAMILY, ground_truth, last_touch_init  # noqa: E402

PALETTE = [(40, 90, 200), (60, 160, 60), (180, 80, 40), (150, 60, 150), (40, 140, 180), (90, 90, 90)]


def draw_sequence(rng, n, k, want_agree, init):
    for _ in range(10000):
        swaps = []
        for _ in range(k):
            a, b = rng.choice(n, 2, replace=False)
            swaps.append((int(a), int(b)))
        gt, sc = ground_truth(n, init, swaps), last_touch_init(n, init, swaps)
        if (gt == sc) == want_agree:
            return swaps
    raise RuntimeError('could not satisfy constraint')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/data')
    ap.add_argument('--per-cell', type=int, default=30, help='videos per (k, agree) cell for opaque3')
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--sample-fps', type=float, default=4.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    items = []

    def add(set_name, n, init, swaps, transparent, idx):
        colors = PALETTE[int(rng.integers(len(PALETTE)))]
        sc = Scene(n, init, swaps, transparent=transparent, cup_color=colors)
        frames = sample_frames(render(sc), sc.fps, a.sample_fps)
        vid = f'{set_name}_{idx:04d}'
        np.savez_compressed(a.out / f'{vid}.npz', frames=frames)
        rec = {'id': vid, 'set': set_name, 'n_cups': n, 'init': init, 'swaps': swaps, 'k': len(swaps),
               'transparent': transparent, 'n_frames': int(len(frames)), 'sample_fps': a.sample_fps,
               'trace': sc.ball_trace(),
               'labels': {name: int(f(n, init, swaps)) for name, f in FAMILY.items()}}
        rec['agree'] = rec['labels']['ground_truth'] == rec['labels']['last_touch_init']
        items.append(rec)
        return rec

    i = 0
    seqs = []
    for k in range(2, 7):
        for agree in (True, False):
            for j in range(a.per_cell):
                init = j % 3
                seqs.append((init, draw_sequence(rng, 3, k, agree, init)))
    for init, swaps in seqs:
        add('opaque3', 3, init, swaps, False, i)
        add('transp3', 3, init, swaps, True, i)
        i += 1
    for j in range(30):
        init = j % 3
        add('k1', 3, init, draw_sequence(rng, 3, 1, True, init), False, j)
    for n in (4, 5):
        i = 0
        for k in (3, 5):
            for agree in (True, False):
                for j in range(a.per_cell):
                    init = j % n
                    add(f'opaque{n}', n, init, draw_sequence(rng, n, k, agree, init), False, i)
                    i += 1
    with open(a.out / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    import collections
    print(json.dumps({'n_items': len(items), 'by_set': collections.Counter(r['set'] for r in items)}, indent=1))


if __name__ == '__main__':
    main()
