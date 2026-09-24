#!/usr/bin/env python3
"""Does written reasoning fix the stale present? CoT prompt on the hardest
cells of Addenda M and O (D_fin 0.5): describe the arrangement at each moment,
then answer with the last line."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, build_inputs, parse_letter  # noqa: E402
from tracking_algo_anchor import tasks  # noqa: E402

B = ROOT / 'outputs/tracking_algo_v1'
KEEP = {'O': ('pos_n1_f0.5', 'obj_n3_f0.5', 'pos_n3_f0.5'), 'M': ('m4_f0.5',)}
COT = (' Before answering, describe what the arrangement looks like at each moment of the video in order, '
       'then state the arrangement at the very end, and give the final answer as a single letter on the last line.')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    out = []
    for which, cells in KEEP.items():
        for it, path, q, vals, cell in tasks(which):
            if cell not in cells:
                continue
            q = q.replace(' Answer with only the letter.', COT)
            frames = np.load(path)['frames']
            inp = build_inputs(cfg, proc, frames, q, 4.0)[0]
            inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
            with torch.inference_mode():
                g = model.generate(**inp, max_new_tokens=400, do_sample=False)
            txt = proc.tokenizer.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
            letters = 'ABCD'[:len(vals)]
            h = parse_letter(txt, letters)
            pred = vals[letters.index(h[-1])] if h else None
            out.append({'id': it['id'], 'set': which, 'cell': cell, 'pred': pred, 'gt': it['gt'], 'pen': it.get('pen'), 'text': txt})
    (B / f'now_cot_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for cell in sorted(set(r['cell'] for r in out)):
        r = [x for x in out if x['cell'] == cell]
        print(a.model, 'CoT', cell, 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
              'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3), 'unparsed', sum(x['pred'] is None for x in r), flush=True)


if __name__ == '__main__':
    main()
