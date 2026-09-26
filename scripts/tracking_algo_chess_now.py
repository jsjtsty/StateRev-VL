#!/usr/bin/env python3
"""Addendum R: stale present on real chess games (MET-Bench-Chess)."""
from pathlib import Path
import argparse
import json
import sys

import chess
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / ('outputs/tracking_algo_v1/data_chess_cap' if '--captures' in sys.argv else 'outputs/tracking_algo_v1/data_chess')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
S, M0 = 48, 32          # square size, margin for labels
FPS = 8
GLYPH = {'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔', 'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'}
NAME = {'p': 'pawn', 'n': 'knight', 'b': 'bishop', 'r': 'rook', 'q': 'queen', 'k': 'king'}


def pname(sym):
    return 'empty' if sym is None else f'{"white" if sym.isupper() else "black"} {NAME[sym.lower()]}'


def render(fen):
    board = chess.Board(fen)
    W = M0 + 8 * S + 16
    img = Image.new('RGB', (W, W), (245, 245, 240)); d = ImageDraw.Draw(img)
    fp = ImageFont.truetype(FONT, 40); fl = ImageFont.truetype(FONT, 18)
    for r in range(8):
        for f in range(8):
            x, y = M0 + f * S, 8 + (7 - r) * S
            d.rectangle([x, y, x + S, y + S], fill=(240, 217, 181) if (r + f) % 2 else (181, 136, 99))
            pc = board.piece_at(chess.square(f, r))
            if pc:
                d.text((x + S / 2, y + S / 2 + 2), GLYPH[pc.symbol()], font=fp, fill=(0, 0, 0), anchor='mm')
    for i in range(8):
        d.text((M0 + i * S + S / 2, 8 + 8 * S + 12), 'abcdefgh'[i], font=fl, fill=(0, 0, 0), anchor='mm')
        d.text((M0 / 2, 8 + (7 - i) * S + S / 2), str(i + 1), font=fl, fill=(0, 0, 0), anchor='mm')
    return np.array(img.resize((448, 448)))


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261011)
    games = pd.read_parquet(ROOT / 'dataset/MET-Bench-Chess/evaluation-test.parquet')
    items = []
    caps = '--captures' in sys.argv
    cells = [(m, dfin, q) for m in (1, 3) for dfin in (0.5, 2.0) for q in (('dst',) if caps else ('src', 'dst'))]
    for m, dfin, qk in cells:
        j = 0
        while j < (100 if caps else 50):
            g = games.iloc[int(rng.integers(len(games)))]
            states = list(g['states'])          # states[0] = initial, states[i] = after action i-1
            if len(states) < m + 2:
                continue
            t = int(rng.integers(m, len(states)))
            mv = chess.Move.from_uci(g['actions'][t - 1])
            sq = mv.from_square if qk == 'src' else mv.to_square
            b_fin, b_pen = chess.Board(states[t]), chess.Board(states[t - 1])
            fin = pname(b_fin.piece_at(sq).symbol() if b_fin.piece_at(sq) else None)
            pen = pname(b_pen.piece_at(sq).symbol() if b_pen.piece_at(sq) else None)
            if fin == pen or (caps and pen == 'empty'):
                continue
            others = sorted({pname(p.symbol()) for p in b_fin.piece_map().values()} - {fin, pen})
            if not others:
                continue
            opts = [fin, pen, others[int(rng.integers(len(others)))]]
            rng.shuffle(opts)
            frames = [np.repeat(render(states[i])[None], FPS, 0) for i in range(t - m, t)]
            frames.append(np.repeat(render(states[t])[None], int(round(dfin * FPS)), 0))
            fr = np.concatenate(frames)[::2]
            vid = f'C_m{m}_f{dfin}_{qk}_{j:02d}'
            np.savez_compressed(OUT / f'{vid}.npz', frames=fr)
            items.append({'id': vid, 'm': m, 'dfin': dfin, 'query': qk, 'square': chess.square_name(sq), 'options': opts,
                          'gt': opts.index(fin), 'pen': opts.index(pen), 'n_frames': int(len(fr)), 'sample_fps': 4.0})
            j += 1
    with open(OUT / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def generate_hist():
    D = ROOT / 'outputs/tracking_algo_v1/data_chess_hist'
    D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261012)
    games = pd.read_parquet(ROOT / 'dataset/MET-Bench-Chess/evaluation-test.parquet')
    items = []
    for m in (1, 3):
        for dfin in (0.5, 2.0):
            j = 0
            while j < 60:
                g = games.iloc[int(rng.integers(len(games)))]
                states = list(g['states'])
                if len(states) < m + 2:
                    continue
                t = int(rng.integers(m, len(states)))
                mv = chess.Move.from_uci(g['actions'][t - 1])
                b0, b_pen, b_fin = chess.Board(states[t - m]), chess.Board(states[t - 1]), chess.Board(states[t])
                victim = b_pen.piece_at(mv.to_square)
                if victim is None:
                    continue
                vname = pname(victim.symbol())
                if sum(pname(p.symbol()) == vname for p in b0.piece_map().values()) != 1:
                    continue
                if b0.piece_at(mv.to_square) is None or pname(b0.piece_at(mv.to_square).symbol()) != vname:
                    continue          # victim must stand on the capture square at the window start
                fin = pname(b_fin.piece_at(mv.to_square).symbol())
                others = sorted({pname(p.symbol()) for p in b_fin.piece_map().values()} - {fin, vname})
                if not others:
                    continue
                opts = [fin, vname, others[int(rng.integers(len(others)))]]
                rng.shuffle(opts)
                frames = [np.repeat(render(states[i])[None], FPS, 0) for i in range(t - m, t)]
                frames.append(np.repeat(render(states[t])[None], int(round(dfin * FPS)), 0))
                fr = np.concatenate(frames)[::2]
                vid = f'H_m{m}_f{dfin}_{j:02d}'
                np.savez_compressed(D / f'{vid}.npz', frames=fr)
                items.append({'id': vid, 'm': m, 'dfin': dfin, 'victim': vname, 'options': [str(o) for o in opts],
                              'gt': opts.index(fin), 'pen': opts.index(vname), 'n_frames': int(len(fr)), 'sample_fps': 4.0})
                j += 1
    with open(D / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def generate_exchange():
    import glob
    D = ROOT / 'outputs/tracking_algo_v1/data_chess_x'
    D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261013)
    cands = []
    for f in sorted(glob.glob(str(ROOT / 'dataset/MET-Bench-Chess/full-test-*.parquet')))[:2]:
        d = pd.read_parquet(f, columns=['actions', 'states'])
        for acts, states in zip(d['actions'], d['states']):
            for i in range(len(acts) - 1):
                if chess.Move.from_uci(acts[i]).to_square == chess.Move.from_uci(acts[i + 1]).to_square:
                    cands.append((list(states[i:i + 3]), chess.Move.from_uci(acts[i]).to_square))
    order = rng.permutation(len(cands))
    items, k = [], 0
    for dfin in (0.5, 2.0):
        j = 0
        while j < 100:
            st, sq = cands[order[k]]; k += 1
            b1, b2 = chess.Board(st[1]), chess.Board(st[2])
            stale, fin = pname(b1.piece_at(sq).symbol()), pname(b2.piece_at(sq).symbol())
            if stale == fin:
                continue
            others = sorted({pname(p.symbol()) for p in b2.piece_map().values()} - {fin, stale})
            if not others:
                continue
            opts = [fin, stale, others[int(rng.integers(len(others)))]]
            rng.shuffle(opts)
            frames = [np.repeat(render(st[0])[None], FPS, 0), np.repeat(render(st[1])[None], FPS, 0),
                      np.repeat(render(st[2])[None], int(round(dfin * FPS)), 0)]
            fr = np.concatenate(frames)[::2]
            vid = f'X_f{dfin}_{j:03d}'
            np.savez_compressed(D / f'{vid}.npz', frames=fr)
            items.append({'id': vid, 'dfin': dfin, 'options': [str(o) for o in opts], 'gt': opts.index(fin), 'pen': opts.index(stale),
                          'n_frames': int(len(fr)), 'sample_fps': 4.0})
            j += 1
    with open(D / 'items.jsonl', 'w') as f:
        for r in items:
            f.write(json.dumps(r) + '\n')
    print(len(items))


def evaluate(model_key, device_map, image_only):
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    items = [json.loads(l) for l in open(OUT / 'items.jsonl')]
    cfg, proc, model = load(model_key, device_map)
    dev = next(model.parameters()).device
    lid = letter_ids(proc.tokenizer, 'ABC')
    out = []
    for it in items:
        o = it['options']
        opt = f'(A) {o[0]} (B) {o[1]} (C) {o[2]}. Answer with only the letter.'
        frames = np.load(OUT / f'{it["id"]}.npz')['frames']
        if image_only:
            q = f'The image shows a chess board. What is on square {it["square"]}? ' + opt
            tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}
            text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}],
                                            tokenize=False, add_generation_prompt=True, **tkw)
            inp = proc(text=[text], images=[frames[-1]], return_tensors='pt')
        else:
            q = f'The video shows a chess game, one position after another. At the very end of the video, what is on square {it["square"]}? ' + opt
            inp = build_inputs(cfg, proc, frames, q, it['sample_fps'])[0]
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        out.append(dict(it, pred=int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))))
    tag = ('imgonly' if image_only else 'video') + ('_cap' if '--captures' in sys.argv else '')
    (ROOT / f'outputs/tracking_algo_v1/chess_now_{model_key}_{tag}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for m in (1, 3):
        for dfin in (0.5, 2.0):
            for qk in ('src', 'dst'):
                if not [x for x in out if x['query'] == qk]:
                    continue
                r = [x for x in out if x['m'] == m and x['dfin'] == dfin and x['query'] == qk]
                print(model_key, tag, f'm={m} D_fin={dfin} {qk}', 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
                      'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--image-only', action='store_true'); ap.add_argument('--captures', action='store_true'); ap.add_argument('--hist', action='store_true'); ap.add_argument('--exchange', action='store_true')
    a = ap.parse_args()
    if '--hist' in sys.argv:
        generate_hist()
    elif '--exchange' in sys.argv:
        generate_exchange()
    else:
        generate() if not a.model else evaluate(a.model, a.device_map, a.image_only)
