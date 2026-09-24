#!/usr/bin/env python3
"""'Is the present corrupted by the past?' Behavioural check on data_reid2:
ask for the color of the cup at a given position at the END of the video
(fully visible), under full / steps / blank. Letter-scored."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, build_inputs, letter_ids  # noqa: E402

D = ROOT / 'outputs/tracking_algo_v1/data_reid2'
POS = ['left', 'middle', 'right']
COL = ['blue', 'green', 'yellow']
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
a = ap.parse_args()
items = [json.loads(l) for l in open(D / 'items.jsonl')]
cfg, proc, model = load(a.model, a.device_map)
dev = next(model.parameters()).device
lid = letter_ids(proc.tokenizer, 'ABC')
out = []
for it in items:
    if it['set'] not in ('full', 'steps', 'blank', 'tele'):
        continue
    cup_at = list(range(3))
    for x, y in it['swaps']:
        cup_at[x], cup_at[y] = cup_at[y], cup_at[x]
    p = it['seq'] % 3
    gt = COL.index(it['cup_color_names'][cup_at[p]])
    q = (f'The video shows three cups of different colors (blue, green, yellow). At the very end of the video, '
         f'what color is the cup in the {POS[p]} position? (A) blue (B) green (C) yellow. Answer with only the letter.')
    frames = np.load(D / f'{it["id"]}.npz')['frames']
    inp, _ = build_inputs(cfg, proc, frames, q, it['sample_fps'])
    inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
    with torch.inference_mode():
        lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
    sc = [float(max(lg[i] for i in lid[l])) for l in 'ABC']
    # color at that position in the INITIAL layout (stale answer)
    out.append({'id': it['id'], 'set': it['set'], 'k': it['k'], 'pred': int(np.argmax(sc)), 'gt': gt,
                'init_color': COL.index(it['cup_color_names'][p])})
(ROOT / f'outputs/tracking_algo_v1/nowprobe_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
import collections
for s in ('blank', 'tele', 'steps', 'full'):
    r = [x for x in out if x['set'] == s]
    d = [x for x in r if x['init_color'] != x['gt']]
    print(a.model, s, 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
          '| when initial differs: acc', round(np.mean([x['pred'] == x['gt'] for x in d]), 3),
          'answers initial color', round(np.mean([x['pred'] == x['init_color'] for x in d]), 3), 'n', len(d))
