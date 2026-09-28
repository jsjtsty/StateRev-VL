#!/usr/bin/env python3
"""Addendum AH: layer-resolved masking of text→visual attention to the prior-state or final frames of the AG rev. 1
held clips (real and rendered, position questions). Reuses the SDPA wrapper of Addendum Y."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B  # noqa: E402
from tracking_algo_real2 import D, ANN  # noqa: E402
from tracking_algo_real4 import build  # noqa: E402
from tracking_algo_real6 import FPS, HOLD, resting_states, domain_clips  # noqa: E402
from tracking_algo_synthword import ask  # noqa: E402


def main():
    import torch
    import zipfile
    from tracking_algo_layermask import MASK  # registers 'masked_sdpa'
    from tracking_algo_eval import load, build_inputs, letter_ids
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    ann = json.load(zipfile.ZipFile(ANN).open('all_valid.json'))
    cfg, proc, model = load(a.model, a.device_map)
    model.set_attn_implementation('masked_sdpa')
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    tcfg = getattr(model.config, 'text_config', model.config)
    NL = tcfg.num_hidden_layers
    vis_tok = model.config.image_token_id if cfg['family'] == 'multiimg' else model.config.video_token_id
    nb = 8 if NL == 48 else 4
    bands = [range(round(NL * k / nb), round(NL * (k + 1) / nb)) for k in range(nb)]
    conds = {'none': None, 'prior_all': ('prior', range(NL)), 'fin_all': ('fin', range(NL))}
    for b in bands:
        conds[f'prior_L{b.start}-{b.stop - 1}'] = ('prior', b)
    lt = getattr(tcfg, 'layer_types', None)
    if lt and len(set(lt)) > 1:
        conds['prior_global'] = ('prior', [i for i, t in enumerate(lt) if t == 'full_attention'])
        conds['prior_sliding'] = ('prior', [i for i, t in enumerate(lt) if t != 'full_attention'])

    def spans(inp):
        """Visual-token positions of the prior frames and of the final frame."""
        pos = (inp['input_ids'][0] == vis_tok).nonzero().flatten()
        if cfg['family'] in ('qwen3vl', 'qwen35'):
            T = int(inp['video_grid_thw'][0][0]); g = pos.view(T, -1)
            assert 2 * (T - 1) == HOLD, T                 # groups 0..T-2 = prior frames, group T-1 = final (+ pad)
            return {'prior': g[:T - 1].flatten(), 'fin': g[T - 1]}, int(pos[-1]) + 1
        n = HOLD + 1; per = len(pos) // n              # LLaVA-OV appends one newline token after the video
        assert len(pos) - n * per <= 1, (len(pos), n)
        g = pos[:n * per].view(n, per)
        return {'prior': g[:HOLD].flatten(), 'fin': g[HOLD]}, int(pos[-1]) + 1

    def answer(inp, keys=None, layers=()):
        MASK.update(keys=keys, layers=set(layers))
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        MASK['keys'] = None
        return int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))

    rng = np.random.default_rng(20260930)
    out = []
    for it in map(json.loads, open(D / 'items.jsonl')):
        rs = resting_states(ann[it['id']], it['options'][it['gt']])
        if rs is None or len(rs) < 2:
            continue
        pen, qs = build(it, [(f, w) for f, _, w in rs], rng)
        if a.limit and len(out) >= a.limit:
            break
        z = np.load(D / f'{it["id"]}.npz')
        C = domain_clips(it, rs, z['crop'], z['fidx'])
        q, opts, g, s = qs['pos']
        rec = {'id': it['id'], 'gt': g, 'stale': s, 'opts': opts}
        for dom, c in C.items():
            inp = build_inputs(cfg, proc, c['held'], ask(q, opts, False), FPS)[0]
            inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
            sp, MASK['qstart'] = spans(inp)
            rec[dom] = {c_: answer(inp, *((sp[spec[0]], spec[1]) if spec else ())) for c_, spec in conds.items()}
        out.append(rec)
        print(a.model, it['id'], g, s, rec['real'], flush=True)
    (B / f'real7_{a.model}{"_lim" if a.limit else ""}.jsonl').write_text('\n'.join(json.dumps(x) for x in out))


if __name__ == '__main__':
    main()
