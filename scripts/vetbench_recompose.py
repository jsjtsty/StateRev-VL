#!/usr/bin/env python3
"""Re-run only the written-composition step of perceive-then-reason, reusing
the perceived swap sequences from a vetbench_autoseg run.
  greedy : one greedy chain of thought (as in the original D3)
  sc{N}  : self-consistency, N sampled chains (T=0.7), majority vote
"""
from pathlib import Path
import argparse
import collections
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, parse_letter  # noqa: E402
from tracking_algo_decompose import text_inputs, POS, PAIR_TXT  # noqa: E402


def prompt(task, start, swaps):
    w = 'cups' if task == 'cup' else 'cards'
    steps = ', then '.join(f'the {PAIR_TXT[p]} {w} were swapped' for p in swaps) or f'no {w} were swapped'
    if task == 'cup':
        head = (f'A ball is under the {POS[start]} cup of three cups (Left, Middle, Right). Then {steps}. '
                'When two cups are swapped, the ball moves with its cup. Which cup contains the ball at the end? ')
    else:
        head = (f'The Queen of Hearts is the {POS[start]} card of three face-down cards (Left, Middle, Right). Then {steps}. '
                'When two cards are swapped, each card moves to the other position. Where is the Queen of Hearts at the end? ')
    return head + '(A) Left (B) Middle (C) Right. Track it step by step, then give the final answer as a single letter on the last line.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', default='cup', choices=['cup', 'card'])
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    B = ROOT / 'outputs/tracking_algo_v1/vetbench_method'
    src = B / f'{a.model}_autoseg2{"" if a.task == "cup" else "_card_sep1.5"}.jsonl'
    rows = [json.loads(l) for l in open(src)]
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    out = []
    for r in rows:
        inp = {k: v.to(dev) for k, v in text_inputs(cfg, proc, prompt(a.task, r['init_perceived'], r['swaps'])).items()}
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=500, do_sample=False)
        dec = lambda ids: parse_letter(proc.tokenizer.decode(ids, skip_special_tokens=True), 'ABC')
        h = dec(g[0, inp['input_ids'].shape[1]:])
        greedy = 'ABC'.index(h[-1]) if h else -1
        votes = []
        with torch.inference_mode():
            gs = model.generate(**inp, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.95,
                                num_return_sequences=a.n)
        for seq in gs:
            h = dec(seq[inp['input_ids'].shape[1]:])
            if h:
                votes.append('ABC'.index(h[-1]))
        sc = collections.Counter(votes).most_common(1)[0][0] if votes else greedy
        out.append({'id': r['id'], 'gt': r['gt'], 'd1': r['d1_vis'], 'greedy': greedy, 'sc': sc, 'votes': votes})
    res = {k: float(np.mean([x[k] == x['gt'] for x in out])) for k in ('d1', 'greedy', 'sc')}
    (B / f'{a.model}_recompose_{a.task}.json').write_text(json.dumps({'summary': res, 'rows': out}, indent=1))
    print(a.model, a.task, json.dumps(res))


if __name__ == '__main__':
    main()
