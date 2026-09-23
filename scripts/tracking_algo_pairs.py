#!/usr/bin/env python3
"""Minimal-pair set: input-level causal test of the tracking algorithm.

For each base sequence (3 cups, opaque, k=3..5) we insert ONE extra swap at a
random position, chosen so that it changes exactly one of {ground truth,
last_touch_init shortcut}, or neither:
  gt_only : GT answer changes, shortcut answer unchanged
  sc_only : shortcut answer changes, GT answer unchanged
  neither : both unchanged (length-matched noise baseline)
Prediction under the shortcut algorithm: the model's answer flips often for
sc_only, rarely for gt_only (about as rarely as for neither). Under correct
tracking the opposite holds.
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
PAIRS3 = [(0, 1), (1, 2), (0, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/data_pairs')
    ap.add_argument('--n-base', type=int, default=120)
    ap.add_argument('--seed', type=int, default=20260924)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    items = []

    def emit(pid, kind, init, swaps, color):
        sc = Scene(3, init, swaps, cup_color=color)
        frames = sample_frames(render(sc), sc.fps, 4.0)
        vid = f'mp{pid:04d}_{kind}'
        np.savez_compressed(a.out / f'{vid}.npz', frames=frames)
        rec = {'id': vid, 'set': f'mp_{kind}', 'pair': pid, 'kind': kind, 'n_cups': 3, 'init': init, 'swaps': swaps,
               'k': len(swaps), 'transparent': False, 'n_frames': int(len(frames)), 'sample_fps': 4.0,
               'labels': {n: int(f(3, init, swaps)) for n, f in FAMILY.items()}}
        rec['agree'] = rec['labels']['ground_truth'] == rec['labels']['last_touch_init']
        items.append(rec)

    pid = 0
    while pid < a.n_base:
        k = int(rng.integers(3, 6))
        init = pid % 3
        base = [PAIRS3[int(rng.integers(3))] for _ in range(k)]
        b_gt, b_sc = ground_truth(3, init, base), last_touch_init(3, init, base)
        found = {}
        for pos in rng.permutation(k + 1):
            for pr in rng.permutation(3):
                s = base[:pos] + [PAIRS3[pr]] + base[pos:]
                dg = ground_truth(3, init, s) != b_gt
                ds = last_touch_init(3, init, s) != b_sc
                kind = 'gt_only' if dg and not ds else 'sc_only' if ds and not dg else 'neither' if not dg and not ds else None
                if kind and kind not in found:
                    found[kind] = s
        if len(found) < 3:
            continue
        color = PALETTE[int(rng.integers(len(PALETTE)))]
        emit(pid, 'base', init, base, color)
        for kind in ('gt_only', 'sc_only', 'neither'):
            emit(pid, kind, init, found[kind], color)
        pid += 1
    with open(a.out / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items), 'items')


if __name__ == '__main__':
    main()
