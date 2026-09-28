#!/usr/bin/env python3
"""Addendum Z: stale present on real footage (Perception Test videos shipped in MVBench's perception.zip).

--prepare: select the end-state questions (letter order at the end; is the bag empty at the end), decode each video
           at 2 fps to 448-px-wide frames, and label the stale option (the beginning order, when it is an option).
--model M: native video, last-frame image-only, and PCD with the motion-compensated change detector (exploratory).
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
B = ROOT / 'outputs/tracking_algo_v1'
D = B / 'data_real'
ANN = ROOT / 'dataset/mc_question_valid_annotations.zip'
FPS = 2.0
WIDTH = 448
ENDQ = {'What is the order of the letters at the end?': 'letters', 'What is the order of the letters on the table at the end?': 'letters',
        'Is the bag empty at the end?': 'bag'}
BEGQ = 'What was the order of the letters at the beginning?'


def decode(path):
    import cv2
    c = cv2.VideoCapture(str(path))
    fps, n = c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.round(np.arange(0, n / fps, 1 / FPS) * fps).astype(int).tolist()) | {n - 1}   # always keep the last frame
    out, i = [], 0
    while True:
        ok, f = c.read()
        if not ok:
            break
        if i in want:
            h = int(round(f.shape[0] * WIDTH / f.shape[1] / 2)) * 2
            out.append(cv2.cvtColor(cv2.resize(f, (WIDTH, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        i += 1
    return np.stack(out)


def prepare():
    import zipfile
    ann = json.load(zipfile.ZipFile(ANN).open('mc_question_valid.json'))
    items = []
    for mp4 in sorted((D / 'videos').glob('*.mp4')):
        vid = mp4.stem
        Q = {q['question']: q for q in ann[vid]['mc_question']}
        for qt, kind in ENDQ.items():
            if qt not in Q:
                continue
            q = Q[qt]
            pen = None
            if kind == 'letters' and BEGQ in Q:
                b = Q[BEGQ]['options'][Q[BEGQ]['answer_id']]
                if b in q['options'] and q['options'].index(b) != q['answer_id']:
                    pen = q['options'].index(b)
            fr = decode(mp4)
            np.savez_compressed(D / f'{vid}.npz', frames=fr)
            items.append(dict(id=vid, kind=kind, question=qt, options=q['options'], gt=q['answer_id'], pen=pen, n_frames=len(fr)))
    (D / 'items.jsonl').write_text('\n'.join(json.dumps(r) for r in items))
    print(len(items), 'items;', sum(r['pen'] is not None for r in items), 'with a stale option;',
          {k: sum(r['kind'] == k for r in items) for k in ('letters', 'bag')})


def prompt(it, image_only):
    o = it['options']
    head = 'The image shows the last frame of a video. ' if image_only else ''
    q = it['question'].replace(' at the end?', '?') if image_only else it['question']
    return head + q + ' ' + ' '.join(f'({l}) {x}' for l, x in zip('ABC', o)) + '. Answer with only the letter.'


def run(model_key, device_map):
    import torch
    from tracking_algo_eval import load, build_inputs, letter_ids
    from tracking_algo_fix import last_change_robust
    cfg, proc, model = load(model_key, device_map)
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
        fr = np.load(D / f'{it["id"]}.npz')['frames']
        full = lp(build_inputs(cfg, proc, fr, prompt(it, False), FPS)[0])
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt(it, True)}]}],
                                        tokenize=False, add_generation_prompt=True, **tkw)
        img = lp(proc(text=[text], images=[fr[-1]], return_tensors='pt', **cfg.get('proc_kw', {})))
        tc = last_change_robust(fr)
        past = lp(build_inputs(cfg, proc, fr[:tc], prompt(it, False), FPS)[0]) if tc >= 2 else np.zeros(3)
        out.append(dict(id=it['id'], kind=it['kind'], gt=it['gt'], pen=it['pen'], tc=tc, n_frames=len(fr),
                        native=int(full.argmax()), img=int(img.argmax()), pcd=int((full - past).argmax()),
                        lp_full=full.tolist(), lp_img=img.tolist(), lp_past=past.tolist()))
    (B / f'real_{model_key}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for kind in ('letters', 'bag'):
        r = [x for x in out if x['kind'] == kind]
        s = [x for x in r if x['pen'] is not None]
        print(model_key, kind, len(r), {k: round(float(np.mean([x[k] == x['gt'] for x in r])), 3) for k in ('img', 'native', 'pcd')},
              'P(stale) on', len(s), {k: round(float(np.mean([x[k] == x['pen'] for x in s])), 3) for k in ('img', 'native', 'pcd')} if s else '', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true'); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    prepare() if a.prepare else run(a.model, a.device_map)
