#!/usr/bin/env python3
"""Stale-present mechanism (exploratory): last-token attention mass on the
video tokens of the FINAL vs the PENULTIMATE layout (Addendum M data), per
layer, for correct vs stale-error trials. Eager attention."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import MODEL_CFG, MODELS, build_inputs, letter_ids  # noqa: E402

B = ROOT / 'outputs/tracking_algo_v1'
POS = ['left', 'middle', 'right']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='qwen3vl8b'); ap.add_argument('--tag', default='_heads'); a = ap.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    cfg = MODEL_CFG[a.model]
    proc = AutoProcessor.from_pretrained(MODELS / cfg['path'])
    model = AutoModelForImageTextToText.from_pretrained(MODELS / cfg['path'], dtype=torch.bfloat16, device_map='cuda:0',
                                                        attn_implementation='eager').eval()
    lid = letter_ids(proc.tokenizer, 'ABC')
    vid_tok = model.config.video_token_id
    rows = []
    for it in map(json.loads, open(B / 'data_now2/items.jsonl')):
        if it['m'] < 2:
            continue
        q = ('The video shows three cups of different colors (blue, green, yellow). At the very end of the video, '
             f'what color is the cup in the {POS[it["p"]]} position? (A) blue (B) green (C) yellow. Answer with only the letter.')
        frames = np.load(B / 'data_now2' / f'{it["id"]}.npz')['frames']
        inp = build_inputs(cfg, proc, frames, q, 4.0)[0]
        T = int(inp['video_grid_thw'][0][0])
        inp = {k: (v.to('cuda:0') if hasattr(v, 'to') else v) for k, v in inp.items()}
        pos = (inp['input_ids'][0] == vid_tok).nonzero().flatten()
        groups = pos.view(T, -1)
        # layout index per temporal group (0.5 s each): reveal 2.5 s, then 1.0 s per non-final layout, then final
        bounds = [2.5 + 1.0 * i for i in range(it['m'])]           # start times of layouts 1..m
        lay = [sum(0.5 * g >= b - 1e-6 for b in bounds) for g in range(T)]
        with torch.inference_mode():
            o = model(**inp, output_attentions=True, logits_to_keep=1)
        pred = int(np.argmax([float(max(o.logits[0, -1, i] for i in lid[l])) for l in 'ABC']))
        fin = [g for g in range(T) if lay[g] == it['m']]; pen = [g for g in range(T) if lay[g] == it['m'] - 1]
        per_layer, heads = [], {}
        for li, att in enumerate(o.attentions):
            if 20 <= li <= 30:
                hv = att[0, :, -1].float()[:, groups].sum(-1).cpu().numpy()   # [H, T]
                heads[li] = [hv[:, fin].mean(1).tolist(), hv[:, pen].mean(1).tolist()]
            v = att[0, :, -1].float().mean(0)                         # heads-avg attention from last token
            gm = v[groups].sum(-1).cpu().numpy()                      # mass per temporal group
            per_layer.append([float(gm[fin].mean()), float(gm[pen].mean()), float(gm.sum())])
        rows.append({'id': it['id'], 'm': it['m'], 'dfin': it['dfin'], 'pred': pred, 'gt': it['gt'], 'pen': it['pen'],
                     'att': per_layer, 'heads': heads})
        del o
    (B / f'now_attn_{a.model}{a.tag}.json').write_text(json.dumps(rows))
    A = np.array([r['att'] for r in rows])                            # [N, L, 3]
    ok = np.array([r['pred'] == r['gt'] for r in rows]); st = np.array([r['pred'] == r['pen'] for r in rows])
    ratio = A[:, :, 0] / (A[:, :, 0] + A[:, :, 1] + 1e-9)
    for L in range(0, A.shape[1], 3):
        print(a.model, f'layer {L + 1}', 'final/(final+pen) per-group attention: correct', round(ratio[ok, L].mean(), 3),
              'stale', round(ratio[st, L].mean(), 3), '| video mass', round(A[:, L, 2].mean(), 3), flush=True)


if __name__ == '__main__':
    main()
