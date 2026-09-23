#!/usr/bin/env python3
"""Text-only composition control: no video at all.

The model gets the initial position and the true list of swaps in text and
must give the final position, either answer-only (one forward, option-letter
logits) or with explicit step-by-step reasoning (greedy generation, parsed).
Separates composition ability from visual perception.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import MODEL_CFG, load, letter_ids, parse_letter  # noqa: E402
from tracking_algo_decompose import text_inputs, score, POS, PAIRS, PAIR_TXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODEL_CFG))
    ap.add_argument('--n', type=int, default=60, help='problems per k')
    ap.add_argument('--ks', default='1,2,3,4,6')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/textcomp')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    rng = np.random.default_rng(7)
    rows = []
    for k in [int(x) for x in a.ks.split(',')]:
        for _ in range(a.n):
            init = int(rng.integers(3)); sw = [int(rng.integers(3)) for _ in range(k)]
            s = init
            for p in sw:
                x, y = PAIRS[p]; s = y if s == x else x if s == y else s
            steps = ', then '.join(f'the {PAIR_TXT[p]} cups were swapped' for p in sw)
            base = (f'A ball is under the {POS[init]} cup of three cups (Left, Middle, Right). Then {steps}. '
                    f'When two cups are swapped, the ball moves with its cup. Which cup contains the ball at the end? '
                    f'(A) Left (B) Middle (C) Right. ')
            inp = text_inputs(cfg, proc, base + 'Answer with only the letter.')
            p_direct, _ = score(model, {kk: v.to(dev) for kk, v in inp.items()}, lid)
            inp = text_inputs(cfg, proc, base + 'Track the ball step by step, then give the final answer as a single letter on the last line.')
            inp = {kk: v.to(dev) for kk, v in inp.items()}
            with torch.inference_mode():
                g = model.generate(**inp, max_new_tokens=400, do_sample=False)
            txt = proc.tokenizer.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
            h = parse_letter(txt, 'ABC')
            rows.append({'k': k, 'gt': s, 'direct': p_direct, 'cot': 'ABC'.index(h[-1]) if h else -1, 'cot_text': txt[-600:]})
        d = [r for r in rows if r['k'] == k]
        print(a.model, 'k', k, 'direct', np.mean([r['direct'] == r['gt'] for r in d]), 'cot', np.mean([r['cot'] == r['gt'] for r in d]), flush=True)
    with open(a.out / f'{a.model}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
