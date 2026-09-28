#!/usr/bin/env python3
"""Addendum AC: the Perception Test word-shuffle items rendered synthetically (static timeline of Addendum AB),
asked as the original letter-order question and as a letter-at-position question; the position question is also
asked on the real AB static clips of the items that have a stale option."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B, FPS  # noqa: E402
from tracking_algo_real2 import D, ANN  # noqa: E402
from tracking_algo_real3 import first_action  # noqa: E402

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']


def render(word, W=448, H=252):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB', (W, H), (205, 205, 200)); d = ImageDraw.Draw(im)
    s = min(64, (W - 40) // len(word) - 8); f = ImageFont.truetype(FONT, int(s * 0.7))
    x0 = (W - len(word) * (s + 8) + 8) // 2; y0 = (H - s) // 2
    for i, ch in enumerate(word):
        x = x0 + i * (s + 8)
        d.rectangle([x, y0, x + s, y0 + s], fill=(250, 250, 250), outline=(90, 90, 90), width=2)
        d.text((x + s / 2, y0 + s / 2), ch.upper(), font=f, fill=(15, 15, 15), anchor='mm')
    return np.asarray(im)


def pos_question(init, fin, rng):
    k = next(i for i in range(min(len(init), len(fin))) if init[i] != fin[i])
    third = next((c for c in fin if c not in (fin[k], init[k])), 'x')
    opts = [fin[k].upper(), init[k].upper(), third.upper()]
    perm = rng.permutation(3)
    return (f'What letter is at the {ORD[k]} position from the left at the end?', [opts[i] for i in perm],
            int(np.where(perm == 0)[0][0]), int(np.where(perm == 1)[0][0]))


def ask(q, opts, image_only):
    head = 'The image shows the last frame of a video. ' if image_only else ''
    q = q.replace(' at the end?', '?') if image_only else q
    return head + q + ' ' + ' '.join(f'({l}) {x}' for l, x in zip('ABC', opts)) + '. Answer with only the letter.'


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
        return int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))

    def video(fr, q, opts):
        return lp(build_inputs(cfg, proc, fr, ask(q, opts, False), FPS)[0])

    def image(f, q, opts):
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': ask(q, opts, True)}]}],
                                        tokenize=False, add_generation_prompt=True, **tkw)
        return lp(proc(text=[text], images=[f], return_tensors='pt', **cfg.get('proc_kw', {})))

    rng = np.random.default_rng(20260928)
    out = []
    for it in map(json.loads, open(D / 'items.jsonl')):
        o = it['options']
        st = it['pen'] if it['pen'] is not None else next(i for i in range(3) if i != it['gt'])
        init, fin = o[st], o[it['gt']]
        A = first_action(ann[it['id']])
        n_init = sum(g < A for g in it['grid']) if A is not None and A >= it['fps'] else 16
        syn = np.concatenate([np.repeat(render(init)[None], n_init, 0), np.repeat(render(fin)[None], 2, 0)])
        pq, popts, pgt, pst = pos_question(init, fin, rng)
        rec = dict(id=it['id'], gt=it['gt'], stale=st, has_pen=it['pen'] is not None, pgt=pgt, pst=pst, n_init=n_init)
        rec['syn_order'] = (image(syn[-1], it['question'], o), video(syn, it['question'], o))
        rec['syn_pos'] = (image(syn[-1], pq, popts), video(syn, pq, popts))
        if it['pen'] is not None and A is not None and A >= it['fps']:
            z = np.load(D / f'{it["id"]}.npz'); fr, fidx = z['crop'], list(z['fidx'])
            init_r = fr[fidx.index([g for g in it['grid'] if g < A][0])]; last = fr[fidx.index(it['ends']['full'])]
            real = np.concatenate([np.repeat(init_r[None], n_init, 0), np.repeat(last[None], 2, 0)])
            rec['real_order'] = (image(last, it['question'], o), video(real, it['question'], o))
            rec['real_pos'] = (image(last, pq, popts), video(real, pq, popts))
        out.append(rec)
        print(a.model, it['id'], {k: v for k, v in rec.items() if k.startswith(('syn', 'real'))}, flush=True)
    (B / f'synthword_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for k, g, s in (('syn_order', 'gt', 'stale'), ('syn_pos', 'pgt', 'pst'), ('real_order', 'gt', 'stale'), ('real_pos', 'pgt', 'pst')):
        r = [x for x in out if k in x]
        print(a.model, k, 'n', len(r), 'img', round(float(np.mean([x[k][0] == x[g] for x in r])), 3),
              'native', round(float(np.mean([x[k][1] == x[g] for x in r])), 3),
              'P(stale) img/native', round(float(np.mean([x[k][0] == x[s] for x in r])), 3), round(float(np.mean([x[k][1] == x[s] for x in r])), 3), flush=True)


if __name__ == '__main__':
    main()
