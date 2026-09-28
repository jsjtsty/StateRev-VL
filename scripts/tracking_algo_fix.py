#!/usr/bin/env python3
"""Addendum S: training-free fixes for the stale present.
CPM (change-point masking at L18-26, Qwen3-VL, eager) and PCD
(present-contrastive decoding, any model)."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import MODEL_CFG, MODELS, load, build_inputs, letter_ids  # noqa: E402
from tracking_algo_anchor import tasks  # noqa: E402

B = ROOT / 'outputs/tracking_algo_v1'
MASK = {'on': False, 'layers': range(18, 27), 'qstart': 0, 'keys': None}


def last_change(frames):
    # a frame starts a new state if > 0.1% of its pixels changed by > 40 intensity levels
    d = (np.abs(np.diff(frames.astype(np.int16), axis=0)).max(-1) > 40).mean((1, 2))
    idx = np.nonzero(d > 0.001)[0]
    return int(idx[-1]) + 1 if len(idx) else 0     # first frame of the final state


def add_noise(frames, seed):
    import cv2
    rng = np.random.default_rng(seed)
    out = []
    for f in frames:
        dx, dy = rng.integers(-2, 3, 2)
        g = cv2.warpAffine(f, np.float32([[1, 0, dx], [0, 1, dy]]), (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REFLECT)
        g = g.astype(np.float32) * (1 + rng.uniform(-0.03, 0.03)) + rng.normal(0, 6, g.shape)
        out.append(np.clip(g, 0, 255).astype(np.uint8))
    return np.stack(out)


def last_change_robust(frames, thr=0.004, max_shift=3):
    """Camera-motion-compensated change detector: align consecutive frames by the
    integer shift (|dx|,|dy| <= max_shift) that minimises their difference, blur, and
    flag a change when > thr of pixels differ by > 25."""
    import cv2
    g = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32), (7, 7), 2) for f in frames]
    g = [x * (128.0 / (x.mean() + 1e-6)) for x in g]          # remove global brightness flicker
    k = max_shift
    frac = []
    for a, b in zip(g[:-1], g[1:]):
        best = None
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                m = np.abs(b[k:-k, k:-k] - a[k + dy:a.shape[0] - k + dy, k + dx:a.shape[1] - k + dx])
                if best is None or m.mean() < best.mean():
                    best = m
        frac.append(float((best[8:-8, 8:-8] > 25).mean()))
    idx = np.nonzero(np.array(frac) > thr)[0]
    return int(idx[-1]) + 1 if len(idx) else 0


def first_static_end(frames, min_len=3):
    """End (exclusive) of the first run of >= min_len frames without change."""
    d = (np.abs(np.diff(frames.astype(np.int16), axis=0)).max(-1) > 40).mean((1, 2)) > 0.001
    run = 0
    for i, ch in enumerate(d):              # d[i]: change between frame i and i+1
        run = 0 if ch else run + 1
        if run >= min_len - 1 and (i + 1 == len(d) or d[i + 1]):
            return i + 2
    return 0


def shell_tasks():
    from tracking_algo_eval import question
    for d, keep in (('data_reid2', ('full', 'steps')), ('data_reid', ('Ifull',))):
        for it in map(json.loads, open(B / d / 'items.jsonl')):
            if it['set'] in keep:
                q, letters = question(it, 'init')
                yield dict(it, gt=it['labels']['ground_truth'], pen=None), B / d / f'{it["id"]}.npz', q, list(range(len(letters))), f'K_{it["set"]}'


def chess_tasks():
    D = B / 'data_chess_cap'
    for it in map(json.loads, open(D / 'items.jsonl')):
        o = it['options']
        q = (f'The video shows a chess game, one position after another. At the very end of the video, what is on square {it["square"]}? '
             f'(A) {o[0]} (B) {o[1]} (C) {o[2]}. Answer with only the letter.')
        yield dict(it, gt=it['gt'], pen=it['pen']), D / f'{it["id"]}.npz', q, [0, 1, 2], f'chess_m{it["m"]}_f{it["dfin"]}'


def digit_ball_tasks():
    D = B / 'data_now3'
    for it in map(json.loads, open(D / 'items.jsonl')):
        if it['m'] != 3 or it['dfin'] != 0.5:
            continue
        if it['task'] == 'digit':
            o = it['options']
            q = f'The video shows a sequence of digits. What digit is shown at the very end of the video? (A) {o[0]} (B) {o[1]} (C) {o[2]}. Answer with only the letter.'
            gt, pen = o.index(it['gt']), o.index(it['pen'])
        else:
            q = ('The video shows a red ball that jumps between positions. Where is the ball at the very end of the video? '
                 '(A) Left (B) Middle (C) Right. Answer with only the letter.')
            gt, pen = it['gt'], it['pen']
        yield dict(it, gt=gt, pen=pen), D / f'{it["id"]}.npz', q, [0, 1, 2], f'N_{it["task"]}'


def hist_tasks(image_only=False):
    D = B / 'data_chess_hist'
    for it in map(json.loads, open(D / 'items.jsonl')):
        o = it['options']
        opt = f'(A) {o[0]} (B) {o[1]} (C) {o[2]}. Answer with only the letter.'
        if image_only:
            q = f'The image shows a chess board at the end of a game segment. What is on the square where the {it["victim"]} stood before? ' + opt
        else:
            q = (f'The video shows a chess game, one position after another. At the very end of the video, what is on the square '
                 f'where the {it["victim"]} stood at the beginning of the video? ' + opt)
        yield dict(it), D / f'{it["id"]}.npz', q, [0, 1, 2], f'H_m{it["m"]}_f{it["dfin"]}'


def ref_tasks(image_only=False):
    what = 'image' if image_only else 'video'
    D = B / 'data_chess_x'
    for it in map(json.loads, open(D / 'items.jsonl')):
        o = it['options']
        q = (('The video shows a chess game, one position after another. At the very end of the video, ' if not image_only else
              'The image shows the final position of a short chess sequence. ')
             + 'what is on the square where the first move of the video ended? '
             + f'(A) {o[0]} (B) {o[1]} (C) {o[2]}. Answer with only the letter.')
        yield it, D / f'{it["id"]}.npz', q, [0, 1, 2], f'U_chess_f{it["dfin"]}'
    D = B / 'data_now5'
    cols = ['red', 'blue', 'green']
    for it in map(json.loads, open(D / 'items.jsonl')):
        q = (f'The {what} shows three colored disks (red, blue, green) in three positions that are rearranged several times. '
             f'At the very end of the video, what color is the disk in the position where the {cols[it["ref_color"]]} disk was at the beginning of the video? '
             '(A) red (B) blue (C) green. Answer with only the letter.')
        yield it, D / f'{it["id"]}.npz', q, [0, 1, 2], f'U_disks_f{it["dfin"]}'


def all_tasks(which=''):
    if which == 'W':
        for t in chess_tasks():
            if t[4].endswith('f0.5'):
                yield t
        for it, path, q, vals, cell in tasks('M'):
            if cell in ('m4_f0.5', 'm4_f1.0'):
                yield it, path, q, vals, f'M_{cell}'
        for t in ref_tasks():
            if t[4].startswith('U_chess'):
                yield t
        return
    if which == 'V':
        for t in chess_tasks():
            if t[4].endswith('f0.5'):
                yield t
        for which2, keep in (('O', ('obj_n3_f0.5', 'pos_n1_f0.5')), ('M', ('m4_f0.5', 'm4_f1.0'))):
            for it, path, q, vals, cell in tasks(which2):
                if cell in keep:
                    yield it, path, q, vals, f'{which2}_{cell}'
        for t in ref_tasks():
            if t[4].startswith('U_chess'):
                yield t
        return
    if which == 'U':
        yield from ref_tasks()
        return
    if which == 'H':
        yield from hist_tasks()
        return
    if which == 'K':
        yield from shell_tasks()
        return
    yield from chess_tasks()
    for which, keep in (('O', ('obj_n3_f0.5', 'pos_n1_f0.5', 'pos_n3_f0.5', 'obj_n3_f2.0', 'pos_n1_f2.0')), ('M', ('m4_f0.5', 'm4_f1.0'))):
        for it, path, q, vals, cell in tasks(which):
            if cell in keep:
                yield it, path, q, vals, f'{which}_{cell}'
    yield from digit_ball_tasks()


def patch_eager():
    import transformers.models.qwen3_vl.modeling_qwen3_vl as M

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        ks = M.repeat_kv(key, module.num_key_value_groups); vs = M.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        if MASK['on'] and isinstance(module, M.Qwen3VLTextAttention) and module.layer_idx in MASK['layers'] and MASK['keys'] is not None:
            w[:, :, MASK['qstart']:, MASK['keys']] = -1e4
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
    M.eager_attention_forward = eager


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--cpm', action='store_true'); ap.add_argument('--only', default=''); ap.add_argument('--image-only', action='store_true'); ap.add_argument('--delta', action='store_true')
    ap.add_argument('--dump', action='store_true', help='Addendum X: save lp_full, lp_past, lp_pres per item'); a = ap.parse_args()
    cfg = MODEL_CFG[a.model]
    if a.cpm:
        patch_eager()
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(MODELS / cfg['path'])
        model = AutoModelForImageTextToText.from_pretrained(MODELS / cfg['path'], dtype=torch.bfloat16, device_map=a.device_map,
                                                            attn_implementation='eager').eval()
    else:
        cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    vid_tok = getattr(model.config, 'video_token_id', None)

    def scores(frames, q, letters, lid):
        inp = build_inputs(cfg, proc, frames, q, 4.0)[0]
        inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
        return inp, (lambda: torch.log_softmax(torch.tensor([float(max(model(**inp, logits_to_keep=1).logits[0, -1].float()[i] for i in lid[l]))
                                                             for l in letters]), 0).numpy())

    if a.image_only:          # last frame alone, history questions (Addendum T control)
        out = []
        lid = letter_ids(proc.tokenizer, 'ABC')
        tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}
        for it, path, q, vals, cell in (ref_tasks(image_only=True) if a.only == 'U' else hist_tasks(image_only=True)):
            fr = np.load(path)['frames'][-1]
            text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}],
                                            tokenize=False, add_generation_prompt=True, **tkw)
            inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in proc(text=[text], images=[fr], return_tensors='pt', **cfg.get('proc_kw', {})).items()}
            with torch.inference_mode():
                lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
            out.append({'id': it['id'], 'cell': cell, 'gt': it['gt'], 'pen': it['pen'], 'img': int(np.argmax([float(max(lg[i] for i in lid[l])) for l in 'ABC']))})
        for cell in dict.fromkeys(r['cell'] for r in out):
            r = [x for x in out if x['cell'] == cell]
            print(a.model, 'image-only', cell, round(np.mean([x['img'] == x['gt'] for x in r]), 3), round(np.mean([x['img'] == x['pen'] for x in r]), 3), flush=True)
        (B / f'fix_img_{a.model}_{a.only or "H"}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
        return
    out = []
    for it, path, q, vals, cell in all_tasks(a.only if a.only in ('K', 'H', 'U', 'V', 'W') else ''):
        if a.only and a.only not in ('K', 'H', 'U', 'V', 'W') and not cell.startswith(a.only):
            continue
        frames = np.load(path)['frames']
        if a.only == 'W':
            frames = add_noise(frames, abs(hash(it['id'])) % (2 ** 31))
        letters = 'ABCD'[:len(vals)]; lid = letter_ids(proc.tokenizer, letters)
        tc = last_change(frames)
        tc_r = last_change_robust(frames) if a.only == 'W' else None
        rec = {'id': it['id'], 'cell': cell, 'gt': it['gt'], 'pen': it.get('pen'), 'tc': tc}
        with torch.inference_mode():
            inp, f = scores(frames, q, letters, lid)
            full = f()
            rec['native'] = vals[int(full.argmax())]
            if a.cpm:
                T = int(inp['video_grid_thw'][0][0]); g = (inp['input_ids'][0] == vid_tok).nonzero().flatten().view(T, -1)
                old = [i for i in range(T) if 2 * i + 1 < tc]          # temporal groups (2 frames) entirely before t_c
                MASK.update(on=True, qstart=int(g[-1, -1]) + 1, keys=g[old].flatten() if old else None)
                rec['cpm'] = vals[int(f().argmax())]
                fe = first_static_end(frames)
                mid = [i for i in range(T) if 2 * i >= fe and 2 * i + 1 < tc]   # groups strictly between first static run and final state
                MASK.update(keys=g[mid].flatten() if mid else None)
                rec['cpm_ends'] = vals[int(f().argmax())]
                MASK['on'] = False
            else:
                if tc >= 2:
                    _, fp = scores(frames[:tc], q, letters, lid)
                    past = fp()
                else:
                    past = np.zeros_like(full)
                for al in (0.5, 1.0):
                    rec[f'pcd{al}'] = vals[int((full - al * past).argmax())]
                if a.dump:
                    pres_fr = frames[tc:] if len(frames) - tc >= 2 else np.concatenate([frames[-1:]] * 2)
                    rec.update(vals=list(vals), lp_full=full.tolist(), lp_past=past.tolist(),
                               lp_pres=scores(pres_fr, q, letters, lid)[1]().tolist())
                if a.only == 'W':
                    past_r = scores(frames[:tc_r], q, letters, lid)[1]() if tc_r >= 2 else np.zeros_like(full)
                    rec['pcd_robust'] = vals[int((full - past_r).argmax())]
                    rec['tc'], rec['tc_r'] = tc, tc_r
                if a.delta:
                    for dl in (0.5, 1.0):
                        cut = len(frames) - int(round(dl * 4))
                        _, fd = scores(frames[:max(cut, 2)], q, letters, lid)
                        rec[f'pcdD{dl}'] = vals[int((full - fd()).argmax())]
        out.append(rec)
    tag = 'cpm' if a.cpm else ('pcdD' if a.delta else ('pcdW' if a.only == 'W' else ('dump' if a.dump else 'pcd')))
    (B / f'fix_{tag}_{a.model}{"_" + a.only if a.only else ""}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    keys = ['native'] + (['cpm', 'cpm_ends'] if a.cpm else ['pcd0.5', 'pcd1.0'] + (['pcdD0.5', 'pcdD1.0'] if a.delta else []) + (['pcd_robust'] if a.only == 'W' else []))
    for cell in dict.fromkeys(r['cell'] for r in out):
        r = [x for x in out if x['cell'] == cell]
        print(a.model, cell, {k: (round(np.mean([x[k] == x['gt'] for x in r]), 3), round(np.mean([x[k] == x['pen'] for x in r]), 3)) for k in keys}, flush=True)


if __name__ == '__main__':
    main()
