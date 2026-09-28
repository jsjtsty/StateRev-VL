#!/usr/bin/env python3
"""Addendum AB: real crop frames on a synthetic timeline (natural D=0.5 clip, splice without the manipulation, static)."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B, FPS, prompt  # noqa: E402
from tracking_algo_real2 import D, ANN, clip, letter_times  # noqa: E402


def first_action(v):
    lids = letter_times(v)[0]   # same letter objects as Addendum AA (grounded question, else 'letter-*' labels)
    s = [x['frame_ids'][0] for x in v['action_localisation'] if lids & set(x['parent_objects'])]
    return min(s) if s else None


def main():
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    ann = json.load(zipfile.ZipFile(ANN).open('all_valid.json'))
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}

    def lp(inp):
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        return torch.log_softmax(torch.tensor([float(max(lg[i] for i in lid[l])) for l in 'ABC']), 0).numpy()

    out = []
    for it in map(json.loads, open(D / 'items.jsonl')):
        A = first_action(ann[it['id']])
        if A is None or A < it['fps']:
            continue
        z = np.load(D / f'{it["id"]}.npz'); fr, fidx = z['crop'], list(z['fidx'])
        init = np.stack([fr[fidx.index(g)] for g in it['grid'] if g < A])
        last2 = np.stack([fr[fidx.index(it['grid'][-1])], fr[fidx.index(it['ends']['full'])]])
        clips = {'natural': clip(it, fr, z['fidx'], '0.5')[0],
                 'splice': np.concatenate([init, last2]),
                 'static': np.concatenate([np.repeat(init[:1], len(init), 0), np.repeat(last2[-1:], 2, 0)])}
        rec = {k: it[k] for k in ('id', 'gt', 'pen')}
        rec['n_init'] = len(init)
        for name, c in clips.items():
            p = lp(build_inputs(cfg, proc, c, prompt(it, False), FPS)[0]); rec[name] = int(p.argmax()); rec[f'lp_{name}'] = p.tolist()
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt(it, True)}]}],
                                        tokenize=False, add_generation_prompt=True, **tkw)
        p = lp(proc(text=[text], images=[last2[-1]], return_tensors='pt', **cfg.get('proc_kw', {}))); rec['img'] = int(p.argmax())
        out.append(rec)
        print(a.model, it['id'], {k: rec[k] for k in ('img', 'natural', 'splice', 'static')}, 'gt', it['gt'], flush=True)
    (B / f'real3_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for k in ('img', 'natural', 'splice', 'static'):
        s = [r for r in out if r['pen'] is not None]
        print(a.model, k, 'acc', round(float(np.mean([r[k] == r['gt'] for r in out])), 3), 'n', len(out),
              'P(stale) on', len(s), round(float(np.mean([r[k] == r['pen'] for r in s])), 3), flush=True)


if __name__ == '__main__':
    main()
