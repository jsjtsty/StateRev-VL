#!/usr/bin/env python3
"""Addendum AA: real footage with a controlled final dwell (Perception Test letters items of Addendum Z).

--prepare: from the official annotations, find the letter objects, the settle frame T (end of the last letter action)
           and the start S of that action; decode the 1080p source at 2 fps plus the clip end frames, as a padded
           crop around the letters and as the wide 448-px frame.
--model M: for D in (0.5, 2, full) and view in (crop, wide): native clip and last-frame image-only;
           on the crop view also PCD with the annotated change start (oracle) and with the robust detector.
"""
from pathlib import Path
import argparse
import json
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tracking_algo_real import B, D as DZ, FPS, WIDTH, prompt  # noqa: E402

D = B / 'data_real2'
ANN = ROOT / 'dataset/valid_annotations.zip'
DWELLS = {'0.5': 0.5, '2': 2.0, 'full': None}


def letter_times(v):
    lids = set(sum([g['answers'] for g in v['grounded_question'] if 'letter' in g['question'].lower()], []))
    if not lids:
        lids = {o['id'] for o in v['object_tracking'] if o['label'].startswith('letter')}
    segs = [s for s in v['action_localisation'] if lids & set(s['parent_objects'])]
    if segs:
        last = max(segs, key=lambda s: s['frame_ids'][1])
        return lids, int(last['frame_ids'][1]), int(last['frame_ids'][0]), 'action'
    T = 0
    for o in v['object_tracking']:
        if o['id'] in lids:
            b = np.array(o['bounding_boxes']); c = (b[:, :2] + b[:, 2:]) / 2; fr = np.array(o['frame_ids'])
            far = np.nonzero(np.abs(c - c[-1]).max(1) > 0.02)[0]
            if len(far):
                T = max(T, int(fr[min(far[-1] + 1, len(fr) - 1)]))
    return lids, T, max(0, T - 30), 'boxes'


def crop_box(v, lids, W, H):
    bb = np.concatenate([np.array(o['bounding_boxes']) for o in v['object_tracking'] if o['id'] in lids])
    x0, y0, x1, y1 = bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = max(0, x0 - w / 2), min(1, x1 + w / 2), max(0, y0 - h / 2), min(1, y1 + h / 2)
    return int(x0 * W), int(y0 * H), int(np.ceil(x1 * W)), int(np.ceil(y1 * H))


def prepare():
    import cv2
    ann = json.load(zipfile.ZipFile(ANN).open('all_valid.json'))
    D.mkdir(exist_ok=True)
    out = []
    for it in map(json.loads, open(DZ / 'items.jsonl')):
        if it['kind'] != 'letters':
            continue
        v = ann[it['id']]
        lids, T, S, src = letter_times(v)
        c = cv2.VideoCapture(str(DZ / 'videos' / f'{it["id"]}.mp4'))
        fps, n = c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        W, H = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x0, y0, x1, y1 = crop_box(v, lids, W, H)
        ends = {k: (n - 1 if d is None else min(n - 1, T + int(round(d * fps)))) for k, d in DWELLS.items()}
        grid = [int(round(t * fps)) for t in np.arange(0, n / fps, 1 / FPS) if int(round(t * fps)) < n]
        want = sorted(set(grid) | set(ends.values()))
        crop, wide, i, got = [], [], 0, []
        cw = WIDTH; ch = int(round((y1 - y0) * cw / (x1 - x0) / 2)) * 2
        wh = int(round(H * WIDTH / W / 2)) * 2
        while True:
            ok, f = c.read()
            if not ok:
                break
            if i in want:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                crop.append(cv2.resize(f[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_AREA))
                wide.append(cv2.resize(f, (WIDTH, wh), interpolation=cv2.INTER_AREA))
                got.append(i)
            i += 1
        ends = {k: min(e, got[-1]) for k, e in ends.items()}   # frame count metadata can overshoot the decodable frames
        np.savez_compressed(D / f'{it["id"]}.npz', crop=np.stack(crop), wide=np.stack(wide), fidx=np.array(got))
        out.append(dict(it, T=T, S=S, src=src, fps=fps, n_src=n, ends=ends, grid=[g for g in grid if g in got],
                        crop_box=[x0, y0, x1, y1], dwell_s=(n - 1 - T) / fps))
    (D / 'items.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    dw = np.array([r['dwell_s'] for r in out])
    print(len(out), 'items; T source', {s: sum(r['src'] == s for r in out) for s in ('action', 'boxes')},
          'natural final dwell (s) quartiles', np.quantile(dw, [.25, .5, .75]).round(1))


def clip(it, arr, fidx, dk):
    """Indices into the saved frames: the 2-fps grid before the clip end, then the end frame."""
    e = it['ends'][dk]
    pos = {f: j for j, f in enumerate(fidx)}
    keep = [pos[g] for g in it['grid'] if g < e] + [pos[e]]
    return arr[keep], [int(fidx[k]) for k in keep]


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

    def video(fr, it):
        return lp(build_inputs(cfg, proc, fr, prompt(it, False), FPS)[0])

    def image(f, it):
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt(it, True)}]}],
                                        tokenize=False, add_generation_prompt=True, **tkw)
        return lp(proc(text=[text], images=[f], return_tensors='pt', **cfg.get('proc_kw', {})))

    out = []
    for it in map(json.loads, open(D / 'items.jsonl')):
        z = np.load(D / f'{it["id"]}.npz')
        rec = {k: it[k] for k in ('id', 'gt', 'pen', 'T', 'S', 'src', 'dwell_s')}
        for view in ('crop', 'wide'):
            for dk in DWELLS:
                fr, fi = clip(it, z[view], z['fidx'], dk)
                full, img = video(fr, it), image(fr[-1], it)
                r = dict(lp_full=full.tolist(), lp_img=img.tolist(), native=int(full.argmax()), img=int(img.argmax()), n=len(fr))
                if view == 'crop':
                    for name, tc in (('oracle', sum(f < it['S'] for f in fi)), ('det', last_change_robust(fr))):
                        past = video(fr[:tc], it) if tc >= 2 else np.zeros(3)
                        r[f'pcd_{name}'] = int((full - past).argmax()); r[f'tc_{name}'] = int(tc)
                rec[f'{view}_{dk}'] = r
        out.append(rec)
        print(model_key, it['id'], {k: (rec[k]['img'], rec[k]['native']) for k in rec if k.startswith(('crop', 'wide'))}, 'gt', it['gt'], flush=True)
    (B / f'real2_{model_key}.jsonl').write_text('\n'.join(json.dumps(r) for r in out))
    for view in ('crop', 'wide'):
        for dk in DWELLS:
            k = f'{view}_{dk}'
            acc = {c: round(float(np.mean([r[k][c] == r['gt'] for r in out])), 3) for c in ('img', 'native', 'pcd_oracle', 'pcd_det') if c in out[0][k]}
            print(model_key, k, acc, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true'); ap.add_argument('--model'); ap.add_argument('--device-map', default='cuda:0')
    a = ap.parse_args()
    prepare() if a.prepare else run(a.model, a.device_map)
