#!/usr/bin/env python3
"""Addendum P: last-frame anchoring. Video + final frame as a separate image."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_eval import load, letter_ids, question  # noqa: E402

B = ROOT / 'outputs/tracking_algo_v1'
POS = ['left', 'middle', 'right']
COLS4 = ['red', 'blue', 'green']
COLS2 = ['blue', 'green', 'yellow']


def anchored_inputs(cfg, proc, frames, q, fps, anchor=True, image_only=False):
    from transformers.video_utils import VideoMetadata
    if image_only:
        tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}
        q = q.replace('At the very end of the video, ', '').replace('The video shows', 'The image shows').replace('are rearranged several times', 'are arranged')
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}],
                                        tokenize=False, add_generation_prompt=True, **tkw)
        return proc(text=[text], images=[frames[-1]], return_tensors='pt')
    content = [{'type': 'video'}]
    if anchor:
        content += [{'type': 'image'}, {'type': 'text', 'text': 'The last frame of the video is shown above. '}]
    content.append({'type': 'text', 'text': q})
    tkw = {'enable_thinking': False} if cfg['family'] == 'qwen35' else {}
    text = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True, **tkw)
    md = VideoMetadata(total_num_frames=len(frames), fps=float(fps), frames_indices=list(range(len(frames))), duration=len(frames) / fps)
    kw = dict(text=[text], videos=[frames], video_metadata=[md], do_sample_frames=False, return_tensors='pt')
    if anchor:
        kw['images'] = [frames[-1]]
    return proc(**kw)


def tasks(which):
    if which == 'O':
        for it in map(json.loads, open(B / 'data_now4/items.jsonl')):
            if it['query'] == 'pos':
                opts = COLS4[:it['n']] + (['nothing'] if it['n'] < 3 else []); vals = list(range(it['n'])) + ([-1] if it['n'] < 3 else [])
                q = ('The video shows colored disks in three positions (left, middle, right) that are rearranged several times. '
                     f'At the very end of the video, what is in the {POS[it["target"]]} position? '
                     + ' '.join(f'({l}) {o}' for l, o in zip('ABCD', opts)) + '. Answer with only the letter.')
            else:
                vals = [0, 1, 2]
                q = ('The video shows colored disks in three positions that are rearranged several times. '
                     f'At the very end of the video, where is the {COLS4[it["target"]]} disk? (A) Left (B) Middle (C) Right. Answer with only the letter.')
            yield it, B / 'data_now4' / f'{it["id"]}.npz', q, vals, f'{it["query"]}_n{it["n"]}_f{it["dfin"]}'
    elif which == 'M':
        for it in map(json.loads, open(B / 'data_now2/items.jsonl')):
            if it['m'] != 4:
                continue
            q = ('The video shows three cups of different colors (blue, green, yellow). At the very end of the video, '
                 f'what color is the cup in the {POS[it["p"]]} position? (A) blue (B) green (C) yellow. Answer with only the letter.')
            yield it, B / 'data_now2' / f'{it["id"]}.npz', q, [0, 1, 2], f'm4_f{it["dfin"]}'
    elif which in ('K', 'J'):
        d = 'data_reid2' if which == 'K' else 'data_reid'
        keep = ('full', 'steps') if which == 'K' else ('Ifull',)
        for it in map(json.loads, open(B / d / 'items.jsonl')):
            if it['set'] not in keep:
                continue
            q, letters = question(it, 'init')
            it = dict(it, gt=it['labels']['ground_truth'])
            yield it, B / d / f'{it["id"]}.npz', q, list(range(len(letters))), it['set']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--sets', default='O,M,K,J')
    ap.add_argument('--device-map', default='cuda:0'); ap.add_argument('--image-only', action='store_true'); a = ap.parse_args()
    cfg, proc, model = load(a.model, a.device_map)
    dev = next(model.parameters()).device
    for which in a.sets.split(','):
        out = []
        for it, path, q, vals, cell in tasks(which):
            frames = np.load(path)['frames']
            letters = 'ABCD'[:len(vals)]
            lid = letter_ids(proc.tokenizer, letters)
            inp = anchored_inputs(cfg, proc, frames, q, it.get('sample_fps', 4.0), image_only=a.image_only)
            inp = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inp.items()}
            with torch.inference_mode():
                lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
            k = int(np.argmax([float(max(lg[i] for i in lid[l])) for l in letters]))
            out.append({'id': it['id'], 'cell': cell, 'pred': vals[k], 'gt': it['gt'], 'pen': it.get('pen')})
        (B / f'anchor{"_imgonly" if a.image_only else ""}_{which}_{a.model}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
        for cell in sorted(set(r['cell'] for r in out)):
            r = [x for x in out if x['cell'] == cell]
            print(a.model, which, cell, 'acc', round(np.mean([x['pred'] == x['gt'] for x in r]), 3),
                  'P(pen)', round(np.mean([x['pred'] == x['pen'] for x in r]), 3) if r[0]['pen'] is not None else '', flush=True)


if __name__ == '__main__':
    main()
