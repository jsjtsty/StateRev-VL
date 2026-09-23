#!/usr/bin/env python3
"""Text-only composition: which written-reasoning format composes swaps best?
P0: 'track the ball step by step' (used in the D3 runs so far)
P1: explicit per-step state lines + the update rule stated
Correct swaps are given; k in {3,5,6}; 100 problems per k."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, parse_letter  # noqa: E402
from tracking_algo_decompose import text_inputs, POS, PAIRS, PAIR_TXT  # noqa: E402

P1_TAIL = ('Rule: if the ball is under one of the two swapped cups, it moves to the other swapped position; '
           'otherwise it stays where it is. Write one line per swap in the form "After swap i: the ball is under X." '
           'Then give the final answer as a single letter on the last line.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(11)
    res = {}
    for k in (3, 5, 6):
        acc = {'P0': [], 'P1': []}
        for _ in range(100):
            init = int(rng.integers(3)); sw = [int(rng.integers(3)) for _ in range(k)]
            s = init
            for p in sw:
                x, y = PAIRS[p]; s = y if s == x else x if s == y else s
            steps = ', then '.join(f'the {PAIR_TXT[p]} cups were swapped' for p in sw)
            base = (f'A ball is under the {POS[init]} cup of three cups (Left, Middle, Right). Then {steps}. '
                    f'When two cups are swapped, the ball moves with its cup. Which cup contains the ball at the end? '
                    f'(A) Left (B) Middle (C) Right. ')
            for tag, tail in (('P0', 'Track the ball step by step, then give the final answer as a single letter on the last line.'), ('P1', P1_TAIL)):
                inp = {kk: v.to(dev) for kk, v in text_inputs(cfg, proc, base + tail).items()}
                with torch.inference_mode():
                    g = model.generate(**inp, max_new_tokens=500, do_sample=False)
                h = parse_letter(proc.tokenizer.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True), 'ABC')
                acc[tag].append(('ABC'.index(h[-1]) if h else -1) == s)
        res[k] = {t: float(np.mean(v)) for t, v in acc.items()}
        print(a.model, k, res[k], flush=True)
    out = ROOT / 'outputs/tracking_algo_v1/textcomp' / f'{a.model}_cotformat.json'
    out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
