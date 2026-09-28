#!/usr/bin/env python3
"""Addendum Y: layer-resolved masking of text→image attention to a chosen state's frames (multi-image models:
Gemma-3, Idefics3). An SDPA wrapper blocks the question positions from the image tokens of the penultimate
('pen'), final ('fin') or earlier ('early') state, in chosen layers."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from transformers import AttentionInterface  # noqa: E402
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
from tracking_algo_eval import load, build_inputs, letter_ids  # noqa: E402
from tracking_algo_fix import chess_tasks  # noqa: E402
from tracking_algo_anchor import tasks  # noqa: E402

B = ROOT / 'outputs/tracking_algo_v1'
MASK = {'layers': set(), 'qstart': 0, 'keys': None}


def masked_sdpa(module, query, key, value, attention_mask, **kw):
    li = getattr(module, 'layer_idx', None)
    if MASK['keys'] is not None and li in MASK['layers'] and 'Vision' not in type(module).__name__ and 'Siglip' not in type(module).__name__:
        nq, nk = query.shape[2], key.shape[2]
        if attention_mask is None:
            attention_mask = torch.ones(nq, nk, dtype=torch.bool, device=query.device).tril(nk - nq)[None, None]
            kw['is_causal'] = False
        else:
            attention_mask = attention_mask.clone()
        if attention_mask.dtype == torch.bool:
            attention_mask[..., MASK['qstart']:, MASK['keys']] = False
        else:
            attention_mask[..., MASK['qstart']:, MASK['keys']] = torch.finfo(attention_mask.dtype).min
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register('masked_sdpa', masked_sdpa)
AttentionMaskInterface.register('masked_sdpa', sdpa_mask)


def change_starts(frames):
    """First frame of every state after the first (pixel detector of Addendum S)."""
    d = (np.abs(np.diff(frames.astype(np.int16), axis=0)).max(-1) > 40).mean((1, 2))
    return [int(i) + 1 for i in np.nonzero(d > 0.001)[0]]


def items():
    for it, path, q, vals, cell in tasks('M'):
        if cell in ('m4_f0.5', 'm4_f1.0'):
            yield it, path, q, vals, f'M_{cell}'
    for t in chess_tasks():
        if t[4] in ('chess_m1_f0.5', 'chess_m3_f0.5'):
            yield t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    cfg, proc, model = load(a.model, a.device_map)
    assert cfg['family'] == 'multiimg'
    model.set_attn_implementation('masked_sdpa')
    dev = next(model.parameters()).device
    tcfg = model.config.text_config
    NL = tcfg.num_hidden_layers
    img_tok = model.config.image_token_id
    bands = [range(round(NL * k / 8), round(NL * (k + 1) / 8)) for k in range(8)]
    conds = {'none': None, 'pen_all': ('pen', range(NL)), 'fin_all': ('fin', range(NL)), 'early_all': ('early', range(NL))}
    for b in bands:
        conds[f'pen_L{b.start}-{b.stop - 1}'] = ('pen', b)
    lt = getattr(tcfg, 'layer_types', None)
    if lt and len(set(lt)) > 1:
        conds['pen_global'] = ('pen', [i for i, t in enumerate(lt) if t == 'full_attention'])
        conds['pen_sliding'] = ('pen', [i for i, t in enumerate(lt) if t != 'full_attention'])
    rows = []
    for n_it, (it, path, q, vals, cell) in enumerate(items()):
        if a.limit and n_it >= a.limit:
            break
        frames = np.load(path)['frames']
        idx = np.arange(len(frames))
        if len(frames) > cfg['max_frames']:
            idx = np.round(np.linspace(0, len(frames) - 1, cfg['max_frames'])).astype(int)
        cs = change_starts(frames)
        tc = cs[-1] if cs else 0
        tp = cs[-2] if len(cs) >= 2 else 0
        tpp = cs[-3] if len(cs) >= 3 else (0 if len(cs) >= 2 else None)
        span = {'pen': (tp, tc), 'fin': (tc, len(frames)), 'early': (tpp, tp) if tpp is not None else (0, 0)}
        letters = 'ABCD'[:len(vals)]; lid = letter_ids(proc.tokenizer, letters)
        inp = build_inputs(cfg, proc, frames, q, 4.0)[0]
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        pos = (inp['input_ids'][0] == img_tok).nonzero().flatten()
        groups = pos.view(len(idx), -1)
        MASK['qstart'] = int(pos[-1]) + 1
        rec = {'id': it['id'], 'cell': cell, 'gt': it['gt'], 'pen': it['pen'], 'tc': tc, 'tp': tp}
        for c, spec in conds.items():
            MASK['keys'] = None
            if spec:
                lo, hi = span[spec[0]]
                fr = [j for j in range(len(idx)) if lo <= idx[j] < hi]
                if not fr:
                    rec[c] = None
                    continue
                MASK.update(keys=groups[fr].flatten(), layers=set(spec[1]))
            with torch.inference_mode():
                lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
            rec[c] = vals[int(np.argmax([float(max(lg[i] for i in lid[l])) for l in letters]))]
        MASK['keys'] = None
        rows.append(rec)
    (B / f'layermask_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    groups_ = {'M': [r for r in rows if r['cell'].startswith('M_')], 'chess': [r for r in rows if r['cell'].startswith('chess')], 'all': rows}
    for g, R in groups_.items():
        if not R:
            continue
        print(a.model, g, len(R))
        for c in conds:
            r = [x for x in R if x.get(c) is not None]
            if r:
                print(f'  {c:14s} n={len(r):3d} acc {np.mean([x[c] == x["gt"] for x in r]):.3f} P(stale) {np.mean([x[c] == x["pen"] for x in r]):.3f}', flush=True)


if __name__ == '__main__':
    main()
