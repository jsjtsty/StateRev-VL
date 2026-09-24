#!/usr/bin/env python3
"""Addendum Q: causal test of recency heads on Addendum L data (Qwen3-VL-8B)."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import transformers.models.qwen3_vl.modeling_qwen3_vl as M  # noqa: E402
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M2  # noqa: E402
TEXT_ATTN = (M.Qwen3VLTextAttention, M2.Qwen2_5_VLAttention)
from tracking_algo_eval import MODEL_CFG, MODELS, build_inputs, letter_ids  # noqa: E402

B = ROOT / 'outputs/tracking_algo_v1'
POS = ['left', 'middle', 'right']
BIAS = {}     # layer_idx -> (heads LongTensor or None, key positions LongTensor)
QSTART = [0]
QEND = [None]


def eager_with_bias(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = M.repeat_kv(key, module.num_key_value_groups)
    value_states = M.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    if isinstance(module, TEXT_ATTN) and module.layer_idx in BIAS:
        hs, ks = BIAS[module.layer_idx]
        if hs is None:        # all heads, text query positions QSTART[0]:QEND[0]
            w[:, :, QSTART[0]:QEND[0], ks] = -1e4
        else:
            w[:, hs[:, None], -1, ks[None, :]] = -1e4
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
    return out, w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='qwen3vl8b'); ap.add_argument('--broad', action='store_true'); ap.add_argument('--layers', default=''); ap.add_argument('--qpos', action='store_true'); a = ap.parse_args()
    M.eager_attention_forward = eager_with_bias
    M2.eager_attention_forward = eager_with_bias
    from transformers import AutoProcessor, AutoModelForImageTextToText
    cfg = MODEL_CFG[a.model]
    proc = AutoProcessor.from_pretrained(MODELS / cfg['path'])
    model = AutoModelForImageTextToText.from_pretrained(MODELS / cfg['path'], dtype=torch.bfloat16, device_map='cuda:0',
                                                        attn_implementation='eager').eval()
    lid = letter_ids(proc.tokenizer, 'ABC')
    vid_tok = model.config.video_token_id
    hp = B / f'recency_heads_{a.model}.json'
    sel = [tuple(x) for x in json.load(open(hp))['heads_layeridx_head']] if hp.exists() else []
    H = model.config.text_config.num_attention_heads
    rng = np.random.default_rng(0)
    pool = [(L, h) for L in range(20, 31) for h in range(H) if (L, h) not in sel]
    rnd = [pool[i] for i in rng.choice(len(pool), 8, replace=False)]
    conds = {'none': None, 'suppress_pen': (sel, 'pen'), 'random_heads': (rnd, 'pen'), 'suppress_final': (sel, 'fin')}
    if a.broad:
        conds = {'none': None, 'text_mask_pen': ('all', 'pen'), 'text_mask_first': ('all', 'first'), 'text_mask_fin': ('all', 'fin')}
    if a.qpos:
        conds = {'none': None, 'L18-26_all': (('all', 18, 26), 'pen', 'all'), 'L18-26_last': (('all', 18, 26), 'pen', 'last'),
                 'L18-26_upto_posword': (('all', 18, 26), 'pen', 'upto'), 'L18-26_after_posword': (('all', 18, 26), 'pen', 'after')}
    if a.layers:
        conds = {'none': None}
        for rg in a.layers.split(','):
            lo, hi = map(int, rg.split('-'))
            conds[f'text_mask_pen_L{lo}-{hi}'] = (('all', lo, hi), 'pen')
    rows = []
    for it in map(json.loads, open(B / 'data_now/items.jsonl')):
        q = ('The video shows three cups of different colors (blue, green, yellow). At the very end of the video, '
             f'what color is the cup in the {POS[it["p"]]} position? (A) blue (B) green (C) yellow. Answer with only the letter.')
        frames = np.load(B / 'data_now' / f'{it["id"]}.npz')['frames']
        inp = build_inputs(cfg, proc, frames, q, 4.0)[0]
        T = int(inp['video_grid_thw'][0][0])
        inp = {k: (v.to('cuda:0') if hasattr(v, 'to') else v) for k, v in inp.items()}
        groups = (inp['input_ids'][0] == vid_tok).nonzero().flatten().view(T, -1)
        t_pen, t_fin = 3.5, 3.5 + it['dpen']
        gi = lambda lo, hi: [g for g in range(T) if lo - 1e-6 <= 0.5 * g < hi - 1e-6]
        keys = {'pen': groups[gi(t_pen, t_fin)].flatten(), 'fin': groups[gi(t_fin, 99)].flatten(), 'first': groups[gi(2.5, 3.5)].flatten()}
        QSTART[0] = int(groups[-1, -1]) + 1
        rec = {'id': it['id'], 'dpen': it['dpen'], 'dfin': it['dfin'], 'gt': it['gt'], 'pen': it['pen']}
        for c, spec in conds.items():
            BIAS.clear()
            if spec:
                heads, which = spec[:2]
                n = inp['input_ids'].shape[1]; vend = int(groups[-1, -1]) + 1
                ids = inp['input_ids'][0].tolist()
                pw = proc.tokenizer.encode(' ' + POS[it['p']], add_special_tokens=False)
                ppos = max(i for i in range(vend, n - len(pw) + 1) if ids[i:i + len(pw)] == pw) + len(pw)
                span = spec[2] if len(spec) > 2 else 'all'
                QSTART[0], QEND[0] = {'all': (vend, None), 'last': (n - 1, None), 'upto': (vend, ppos), 'after': (ppos, None)}[span]
                if heads == 'all' or (isinstance(heads, tuple) and heads[0] == 'all'):
                    lo, hi = (0, model.config.text_config.num_hidden_layers - 1) if heads == 'all' else heads[1:]
                    for L in range(lo, hi + 1):
                        BIAS[L] = (None, keys[which])
                    heads = []
                for L in sorted(set(L for L, _ in heads)):
                    BIAS[L] = (torch.tensor([h for LL, h in heads if LL == L], device='cuda:0'), keys[which])
            with torch.inference_mode():
                lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
            rec[c] = int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))
        rows.append(rec)
    BIAS.clear()
    (B / f'recency_heads_test_{a.model}{"_broad" if a.broad else ""}{"_layers" if a.layers else ""}{"_qpos" if a.qpos else ""}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    for dfin in (0.5, 1.0, 2.0, 4.0):
        r = [x for x in rows if x['dfin'] == dfin]
        print(a.model, f'D_fin={dfin}', {c: (round(np.mean([x[c] == x['gt'] for x in r]), 3), round(np.mean([x[c] == x['pen'] for x in r]), 3)) for c in conds}, flush=True)
    r = [x for x in rows if x['dfin'] <= 1.0]
    print('D_fin<=1 acc/P(pen)', {c: (round(np.mean([x[c] == x['gt'] for x in r]), 3), round(np.mean([x[c] == x['pen'] for x in r]), 3)) for c in conds})


if __name__ == '__main__':
    main()
