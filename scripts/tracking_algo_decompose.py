#!/usr/bin/env python3
"""Training-free decomposition: perceive each swap separately, then compose.

For each opaque 3-cup video:
  1. every swap window is cut into its own short clip (0.25 s before to 0.25 s
     after the motion) and the model is asked which two positions were swapped
     (3-way option-letter logits; this is a perception question);
  2. D1 = compose the model's own perceived swaps with the exact swap rule
     (upper bound given the model's perception);
  3. D2 = give the model, in TEXT ONLY, the initial position and its own
     perceived swap list, and ask for the final position (the model does the
     composition itself, no video).
Native single-question accuracy on the same videos comes from the G1 run.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from tracking_algo_eval import MODEL_CFG, load, build_inputs, letter_ids  # noqa: E402
from staterev.shellsim import LIFT, HOLD, LOWER, SWAP, PAUSE  # noqa: E402

POS = ['Left', 'Middle', 'Right']
PAIRS = [(0, 1), (1, 2), (0, 2)]
PAIR_TXT = ['Left and Middle', 'Middle and Right', 'Left and Right']


def swap_clip(frames, i, fps):
    t0 = LIFT + HOLD + LOWER + i * (SWAP + PAUSE)
    a, b = int(np.floor((t0 - 0.25) * fps)), int(np.ceil((t0 + SWAP + 0.25) * fps))
    return frames[max(a, 0):min(b + 1, len(frames))]


def text_inputs(cfg, proc, q):
    msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': q}]}]
    tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw)
    return proc(text=[text], return_tensors='pt')


def score(model, inputs, lid):
    with torch.inference_mode():
        lg = model(**inputs, logits_to_keep=1).logits[0, -1].float()
    s = {l: float(max(lg[i] for i in ids)) for l, ids in lid.items()}
    return 'ABC'.index(max(s, key=s.get)), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODEL_CFG))
    ap.add_argument('--sets', default='opaque3,k1')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/decompose')
    ap.add_argument('--cot', action='store_true', help='also D3: perceived swaps + written step-by-step composition')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'outputs/tracking_algo_v1/data'
    items = [json.loads(l) for l in open(data / 'items.jsonl')]
    items = [r for r in items if r['set'] in a.sets.split(',')]
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    path = a.out / f'{a.model}{"_cot" if a.cot else ""}.jsonl'
    done = {json.loads(l)['id'] for l in open(path)} if path.exists() else set()
    t0 = time.time()
    with open(path, 'a') as f:
        for n, it in enumerate(items):
            if it['id'] in done:
                continue
            frames = np.load(data / f'{it["id"]}.npz')['frames']
            perceived = []
            for i in range(it['k']):
                clip = swap_clip(frames, i, it['sample_fps'])
                q = ('The video shows 3 identical cups in a row. Two of the cups swap their positions. '
                     'Which two positions were swapped? (A) Left and Middle (B) Middle and Right (C) Left and Right. '
                     'Answer with only the letter.')
                inp, _ = build_inputs(cfg, proc, clip, q, it['sample_fps'])
                p, _ = score(model, {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}, lid)
                perceived.append(p)
            s = it['init']
            for p in perceived:
                x, y = PAIRS[p]
                s = y if s == x else x if s == y else s
            d1 = s
            steps = ', then '.join(f'the {PAIR_TXT[p]} cups were swapped' for p in perceived)
            q2 = (f'A ball is under the {POS[it["init"]]} cup of three cups (Left, Middle, Right). '
                  f'Then {steps}. When two cups are swapped, the ball moves with its cup. '
                  f'Which cup contains the ball at the end? (A) Left (B) Middle (C) Right. Answer with only the letter.')
            inp2 = text_inputs(cfg, proc, q2)
            d2, _ = score(model, {k: v.to(dev) for k, v in inp2.items()}, lid)
            d3 = None
            if a.cot:
                q3 = q2.replace('Answer with only the letter.', 'Track the ball step by step, then give the final answer as a single letter on the last line.')
                inp3 = {k: v.to(dev) for k, v in text_inputs(cfg, proc, q3).items()}
                with torch.inference_mode():
                    g3 = model.generate(**inp3, max_new_tokens=400, do_sample=False)
                t3 = proc.tokenizer.decode(g3[0, inp3['input_ids'].shape[1]:], skip_special_tokens=True)
                from tracking_algo_eval import parse_letter
                h3 = parse_letter(t3, 'ABC')
                d3 = 'ABC'.index(h3[-1]) if h3 else -1
            true_pairs = [PAIRS.index(tuple(sorted(sw))) for sw in it['swaps']]
            rec = {'id': it['id'], 'set': it['set'], 'k': it['k'], 'perceived': perceived, 'true_pairs': true_pairs,
                   'event_acc': float(np.mean([p == t for p, t in zip(perceived, true_pairs)])),
                   'd1': d1, 'd2': d2, 'd3_cot': d3, 'gt': it['labels']['ground_truth']}
            f.write(json.dumps(rec) + '\n'); f.flush()
            if (n + 1) % 25 == 0:
                print(f'{a.model} {n + 1}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    print('DONE', a.model)


if __name__ == '__main__':
    main()
