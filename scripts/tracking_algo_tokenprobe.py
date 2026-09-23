#!/usr/bin/env python3
"""Extract per-temporal-group video-token hidden states (G2 mechanism probe).

For each item: one forward with the `init` prompt; at a set of layers, the
video tokens are split into their temporal groups (video_grid_thw[0] groups,
each h*w/4 contiguous tokens) and mean-pooled per group. Saved as
[layers, T_groups, D] float16 per item. With causal attention, the tokens of
group g only see frames up to g, so the state after swap j can only be
represented at groups after that swap has finished: this asks WHERE along
the video the latent state is (or is not) carried.
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
from tracking_algo_eval import MODEL_CFG, load, build_inputs, question  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODEL_CFG))
    ap.add_argument('--sets', default='opaque3,transp3')
    ap.add_argument('--layers', default='')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/tokenprobe')
    ap.add_argument('--data', type=Path, default=ROOT / 'outputs/tracking_algo_v1/data')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = a.data
    items = [json.loads(l) for l in open(data / 'items.jsonl')]
    if a.sets != 'all':
        items = [r for r in items if r['set'] in a.sets.split(',')]
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    is_llava = cfg['family'] == 'llava'
    vid_tok = model.config.video_token_index if is_llava else model.config.video_token_id
    n_layers = model.config.text_config.num_hidden_layers or 32
    layers = [int(x) for x in a.layers.split(',')] if a.layers else sorted({int(x) for x in np.linspace(4, n_layers, 9).round()})
    store, t0 = {}, time.time()
    for n, it in enumerate(items):
        frames = np.load(data / f'{it["id"]}.npz')['frames']
        q, _ = question(it, 'init')
        if is_llava:
            # one temporal group per frame; 2 fps keeps long videos inside the 4k context
            frames = frames[::2]
            cfg_l = dict(cfg, max_frames=10 ** 6)
            inp, _ = build_inputs(cfg_l, proc, frames, q, it['sample_fps'] / 2)
            grid = [len(frames)]
        else:
            inp, _ = build_inputs(cfg, proc, frames, q, it['sample_fps'])
            grid = inp['video_grid_thw'][0].tolist()
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            o = model.model(**inp, output_hidden_states=True)
        pos = (inp['input_ids'][0] == vid_tok).nonzero().flatten()
        T = int(grid[0])
        assert len(pos) % T == 0, (len(pos), grid)
        chunks = pos.view(T, -1)
        last = int(inp['input_ids'].shape[1]) - 1
        arr = torch.stack([torch.cat([torch.stack([o.hidden_states[L][0, c.to(o.hidden_states[L].device)].float().mean(0).cpu() for c in chunks]),
                                      o.hidden_states[L][0, last].float().cpu()[None]]) for L in layers])  # [L, T+1, D]; last row = last token
        store[it['id']] = arr.cpu().numpy().astype(np.float16)
        if (n + 1) % 50 == 0:
            print(f'{a.model} {n + 1}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    np.savez(a.out / f'{a.model}{a.tag}.npz', **store)
    (a.out / f'{a.model}{a.tag}.json').write_text(json.dumps({'layers': layers, 'sets': a.sets, 'n': len(store),
                                                               'last_row_is_last_token': True,
                                                               'group_sec': 0.5}))
    print('DONE', a.model)


if __name__ == '__main__':
    main()
