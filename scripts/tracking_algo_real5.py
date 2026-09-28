#!/usr/bin/env python3
"""Addendum AF: 'frozen' clips (the clip's last frame repeated to the clip's length) through the video path, for the
Addendum AE questions; native and image-only answers come from real4_<model>.jsonl."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B, FPS  # noqa: E402
from tracking_algo_real2 import D, clip  # noqa: E402
from tracking_algo_synthword import ask, ORD  # noqa: E402


def main():
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')

    def lp(inp):
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        return int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))

    items = {json.loads(l)['id']: json.loads(l) for l in open(D / 'items.jsonl')}
    out = []
    for r in map(json.loads, open(B / f'real4_{a.model}.jsonl')):
        it = items[r['id']]; z = np.load(D / f'{r["id"]}.npz')
        rec = {'id': r['id']}
        fin, pen = it['options'][it['gt']], r['pen_word']
        k = next(i for i in range(len(fin)) if pen[i] != fin[i])
        text = {'pos': f'At the end, which letter is in the {ORD[k]} place of the letter sequence?', 'order': it['question']}
        for dk in ('0.5', 'full'):
            fr = clip(it, z['crop'], z['fidx'], dk)[0]
            frozen = np.repeat(fr[-1:], len(fr), 0)
            for qn in ('pos', 'order'):
                if qn not in r:
                    continue
                rec.setdefault(qn, {})[dk] = lp(build_inputs(cfg, proc, frozen, ask(text[qn], r[qn]['opts'], False), FPS)[0])
        out.append(rec)
        print(a.model, r['id'], rec, flush=True)
    (B / f'real5_{a.model}.jsonl').write_text('\n'.join(json.dumps(x) for x in out))


if __name__ == '__main__':
    main()
