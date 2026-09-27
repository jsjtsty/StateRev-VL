#!/usr/bin/env python3
"""Native answer evaluation for the tracking-algorithm study (G1).

For each item and prompt variant: option-letter next-token logits right after
the generation prompt (thinking disabled where the template allows), argmax
answer, a short greedy continuation (format check), and optionally last-token
hidden states for all layers. See outputs/tracking_algo_v1/PREREG.md.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path('/home/wangyf/Models')
DATA = ROOT / 'outputs/tracking_algo_v1/data'

MODEL_CFG = {
    'qwen3vl8b': dict(path='Qwen3-VL-8B-Instruct', family='qwen3vl'),
    'qwen3vl2b': dict(path='Qwen3-VL-2B-Instruct', family='qwen3vl'),
    'qwen3vl4b': dict(path='Qwen3-VL-4B-Instruct', family='qwen3vl'),
    'qwen3vl32b': dict(path='Qwen3-VL-32B-Instruct', family='qwen3vl'),
    'qwen3vl8b_think': dict(path='Qwen3-VL-8B-Thinking', family='qwen3vl', thinking=True),
    'qwen25vl7b': dict(path='Qwen2.5-VL-7B-Instruct', family='qwen25vl'),
    'llava_video7b': dict(path='LLaVA-NeXT-Video-7B-hf', family='llava', max_frames=16),
    'llava_ov7b': dict(path='llava-onevision-qwen2-7b-ov-hf', family='llava', max_frames=32),
    'internvl35_8b': dict(path='InternVL3_5-8B-HF', family='llava', max_frames=32, square=448),
    'gemma3_12b': dict(path='gemma-3-12b-it', family='multiimg', max_frames=32),
    'idefics3_8b': dict(path='Idefics3-8B-Llama3', family='multiimg', max_frames=32, proc_kw=dict(do_image_splitting=False)),
    'internvl_gptoss': dict(path='InternVL3_5-GPT-OSS-20B-A4B-Preview', family='multiimg', max_frames=32, remote=True),
    'qwen35_9b': dict(path='Qwen3.5-9B', family='qwen35'),
    'qwen36_27b': dict(path='Qwen3.6-27B', family='qwen35'),
    'qwen36_35b_a3b': dict(path='Qwen3.6-35B-A3B', family='qwen35'),
}
POS3 = ['Left', 'Middle', 'Right']


def question(item, variant):
    n = item['n_cups']
    letters = 'ABCDE'[:n]
    if n == 3:
        names = POS3
        opts = ' '.join(f'({l}) {p}' for l, p in zip(letters, names))
        start = f'At the start, the ball is under the {names[item["init"]]} cup. '
        where = ''
    else:
        names = [f'cup {i + 1}' for i in range(n)]
        opts = ' '.join(f'({l}) {p}' for l, p in zip(letters, names))
        start = f'At the start, the ball is under cup {item["init"] + 1}. '
        where = 'Cups are numbered 1 to %d from left to right. ' % n
    if item.get('task') == 'card':
        q = ('The video shows three playing cards face up, then they are turned face down and swapped several times. ')
        if variant == 'init':
            q += f'At the start, the Queen of Hearts is the {POS3[item["init"]]} card. '
        q += 'Where is the Queen of Hearts at the end of the video? (A) Left (B) Middle (C) Right. Answer with only the letter.'
        return q, 'ABC'
    kind = 'identical cups' if not item.get('cup_color_names') else 'cups of different colors'
    q = (f'The video shows a shell game with {n} {kind}. A red ball is placed under one cup, '
         f'then the cups are swapped several times. ')
    if variant == 'init':
        q += start
    q += f'{where}Which cup contains the ball at the end of the video? {opts}. Answer with only the letter.'
    return q, letters


class InternVLChatProc:
    """Processor shim for remote-code InternVLChatModel: one 448x448 tile (256 tokens) per image, as InternVL does for
    video frames; prompt in the model's own conversation template, answer channel opened (GPT-OSS harmony format)."""
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def __init__(self, tok, system, n_tok):
        self.tokenizer, self.system, self.n_tok = tok, system, n_tok

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        body = ''.join(c['text'] + ' ' if c['type'] == 'text' else '<image>\n' for c in msgs[0]['content']).strip()
        return (f'<|start|>system<|message|>{self.system}<|end|><|start|>user<|message|>{body}<|end|>'
                '<|start|>assistant<|channel|>final<|message|>')

    def __call__(self, text, images, return_tensors='pt', **kw):
        import cv2
        imgs = images[0] if isinstance(images[0], (list, tuple)) else images
        px = np.stack([(cv2.resize(np.asarray(im)[..., :3], (448, 448), interpolation=cv2.INTER_CUBIC) / 255. - self.MEAN) / self.STD
                       for im in imgs]).transpose(0, 3, 1, 2)
        t = text[0]
        assert t.count('<image>') == len(imgs)
        t = t.replace('<image>', '<img>' + '<IMG_CONTEXT>' * self.n_tok + '</img>')
        enc = self.tokenizer([t], return_tensors='pt')
        return dict(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                    pixel_values=torch.tensor(px, dtype=torch.bfloat16), image_flags=torch.ones(len(imgs), 1, dtype=torch.long))


def load_remote(cfg, device_map):
    from transformers import AutoTokenizer
    try:
        import timm  # noqa: F401
    except ImportError:                            # remote code only needs timm's DropPath (identity at inference)
        import types
        m = types.ModuleType('timm.layers'); m.DropPath = lambda *a, **k: torch.nn.Identity()
        sys.modules['timm'], sys.modules['timm.layers'] = types.ModuleType('timm'), m
    path = MODELS / cfg['path']
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module('modeling_internvl_chat.InternVLChatModel', str(path))
    if not getattr(cls, '_post_init_patched', False):   # written for transformers 4.x: __init__ never calls post_init()
        init0 = cls.__init__

        def init(self, *a, **k):
            init0(self, *a, **k)
            self.post_init()
        cls.__init__, cls._post_init_patched = init, True
    if device_map == 'auto' and torch.cuda.device_count() > 1:   # ViT, projector and embeddings must share a device
        n = torch.cuda.device_count(); split = [int(24 * (i + 1) / (n + 0.6)) for i in range(n)]
        device_map = {'vision_model': 0, 'mlp1': 0, 'language_model.model.embed_tokens': 0,
                      'language_model.model.norm': n - 1, 'language_model.lm_head': n - 1,
                      'language_model.model.rotary_emb': 0}
        for l in range(24):
            device_map[f'language_model.model.layers.{l}'] = min(sum(l >= b for b in split), n - 1)
    model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map).eval()
    model.language_model.set_attn_implementation('flex_attention')   # eager (default, no SDPA for sinks) OOMs at ~8k tokens
    model.img_context_token_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    outer, lm_fwd, keep = model.forward, model.language_model.forward, {}

    def lm_forward(*a, **k):                      # the wrapper does not take logits_to_keep; pass it to the LM
        return lm_fwd(*a, logits_to_keep=keep.get('n', 0), **k)

    def forward(*a, logits_to_keep=0, **k):
        keep['n'] = logits_to_keep
        return outer(*a, **k)
    model.language_model.forward, model.forward = lm_forward, forward
    return cfg, InternVLChatProc(tok, model.system_message, model.num_image_token), model


def load(model_key, device_map):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    cfg = MODEL_CFG[model_key]
    if cfg.get('remote'):
        return load_remote(cfg, device_map)
    path = MODELS / cfg['path']
    proc = AutoProcessor.from_pretrained(path)
    if cfg.get('square'):          # InternVL: the video processor defaults to 384, the model expects 448
        proc.video_processor.size = {'height': cfg['square'], 'width': cfg['square']}
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map)
    model.eval()
    return cfg, proc, model


def build_inputs(cfg, proc, frames, text_q, fps, think=False):
    from transformers.video_utils import VideoMetadata
    fam = cfg['family']
    if fam in ('llava', 'multiimg') and len(frames) > cfg['max_frames']:
        idx = np.round(np.linspace(0, len(frames) - 1, cfg['max_frames'])).astype(int)
        frames = frames[idx]
    if cfg.get('square'):
        import cv2
        frames = np.stack([cv2.resize(f, (cfg['square'], cfg['square'])) for f in frames])
    if fam == 'multiimg':         # no video input: frames as a list of timestamped images (as in now2 --mode images)
        content = []
        for i in range(len(frames)):
            content += [{'type': 'text', 'text': f'Frame {i + 1}:'}, {'type': 'image'}]
        content.append({'type': 'text', 'text': text_q})
        text = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        return proc(text=[text], images=[list(frames)], return_tensors='pt', **cfg.get('proc_kw', {})), text
    msgs = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': text_q}]}]
    tkw = {}
    if fam == 'qwen35':
        tkw['enable_thinking'] = bool(think)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw)
    if fam in ('qwen3vl', 'qwen35'):
        md = VideoMetadata(total_num_frames=len(frames), fps=float(fps), frames_indices=list(range(len(frames))),
                           duration=len(frames) / fps)
        return proc(text=[text], videos=[frames], video_metadata=[md], do_sample_frames=False, return_tensors='pt'), text
    if fam == 'qwen25vl':
        return proc(text=[text], videos=[frames], videos_kwargs=dict(fps=float(fps), do_sample_frames=False),
                    return_tensors='pt'), text
    return proc(text=[text], videos=[frames], return_tensors='pt'), text


def parse_letter(text, letters):
    import re
    L = f'[{letters}]'
    for pat in (rf'answer\W*(?:is\W*)?\**\(?({L})\)?\b', rf'\(({L})\)', rf'\*\*({L})\*\*', rf'(?:^|\n)\s*({L})\W*$', rf'\b({L})\W*$'):
        hits = re.findall(pat, text, flags=re.I if 'answer' in pat else 0)
        if hits:
            return [h.upper() for h in hits]
    return []


def letter_ids(tok, letters):
    out = {}
    for l in letters:
        ids = set()
        for s in (l, ' ' + l):
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                ids.add(e[0])
        out[l] = sorted(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODEL_CFG))
    ap.add_argument('--sets', default='opaque3,transp3,k1,opaque4,opaque5')
    ap.add_argument('--variants', default='init,vis')
    ap.add_argument('--device-map', default='cuda:0')
    ap.add_argument('--hidden', action='store_true', help='save last-token hidden states (all layers)')
    ap.add_argument('--hidden-sets', default='opaque3,transp3')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--max-new-tokens', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--think', action='store_true', help='long generation with reasoning; answer parsed from text')
    ap.add_argument('--think-tokens', type=int, default=2048)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/tracking_algo_v1/eval')
    ap.add_argument('--data', type=Path, default=DATA)
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split('/'))
    out = a.out / (a.model + ('_thinkgen' if a.think else ''))
    out.mkdir(parents=True, exist_ok=True)
    sets = set(a.sets.split(','))
    items = [json.loads(l) for l in open(a.data / 'items.jsonl')]
    items = [r for r in items if r['set'] in sets]
    items = items[si::sn]
    if a.limit:
        items = items[:a.limit]
    cfg, proc, model = load(a.model, a.device_map)
    tok = proc.tokenizer
    dev = next(model.parameters()).device
    rows_path = out / f'rows_shard{si}of{sn}.jsonl'
    done = set()
    if rows_path.exists():
        done = {(json.loads(l)['id'], json.loads(l)['variant']) for l in open(rows_path)}
    hidden_store = {}
    hpath = out / f'hidden_shard{si}of{sn}.npz'
    if a.hidden and hpath.exists():
        hidden_store = dict(np.load(hpath))
    hsets = set(a.hidden_sets.split(','))
    t0 = time.time()
    with open(rows_path, 'a') as fout:
        for n_done, item in enumerate(items):
            frames = np.load(a.data / f'{item["id"]}.npz')['frames']
            for variant in a.variants.split(','):
                if (item['id'], variant) in done:
                    continue
                q, letters = question(item, variant)
                if a.think:
                    q = q.replace('Answer with only the letter.', 'Think step by step, then give the final answer as a single letter.')
                inputs, text = build_inputs(cfg, proc, frames, q, item['sample_fps'], think=a.think)
                inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
                want_h = a.hidden and item['set'] in hsets and variant == 'init'
                if a.think:
                    with torch.inference_mode():
                        g = model.generate(**inputs, max_new_tokens=a.think_tokens, do_sample=False)
                    gen = tok.decode(g[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
                    final = gen.split('</think>')[-1] if '</think>' in gen else ''
                    hits = parse_letter(final, letters)
                    pred = letters.index(hits[-1]) if hits else -1
                    forced = False
                    if pred < 0:
                        # budget forcing: close the reasoning and score the option letters
                        forced = True
                        tail = ('' if '</think>' in gen else '\n</think>\n\n') + 'The final answer is ('
                        ext = torch.cat([g[0], torch.tensor(tok.encode(tail, add_special_tokens=False), device=dev)])[None]
                        fin = dict(inputs)
                        fin['input_ids'] = ext
                        fin['attention_mask'] = torch.ones_like(ext)
                        if 'mm_token_type_ids' in fin:
                            t_ = fin['mm_token_type_ids']
                            fin['mm_token_type_ids'] = torch.cat([t_, torch.zeros(1, ext.shape[1] - t_.shape[1], dtype=t_.dtype, device=dev)], 1)
                        with torch.inference_mode():
                            lg = model(**fin, logits_to_keep=1).logits[0, -1].float()
                        lid_ = letter_ids(tok, letters)
                        sc_ = {l: float(max(lg[i] for i in ids)) for l, ids in lid_.items()}
                        pred = letters.index(max(sc_, key=sc_.get))
                    rec = {'id': item['id'], 'set': item['set'], 'variant': variant, 'model': a.model + '_thinkgen',
                           'pred': pred, 'parsed': bool(hits), 'forced': forced, 'gen': gen[-3000:], 'n_gen_tokens': int(g.shape[1] - inputs['input_ids'].shape[1]),
                           'n_input_tokens': int(inputs['input_ids'].shape[1])}
                    fout.write(json.dumps(rec) + '\n'); fout.flush()
                    continue
                with torch.inference_mode():
                    o = model(**inputs, logits_to_keep=1, output_hidden_states=want_h)
                logits = o.logits[0, -1].float()
                lid = letter_ids(tok, letters)
                scores = {l: float(max(logits[i] for i in ids)) if ids else float('nan') for l, ids in lid.items()}
                pred = max(scores, key=scores.get)
                top = tok.decode([int(logits.argmax())])
                gen = ''
                if a.max_new_tokens:
                    with torch.inference_mode():
                        g = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
                    gen = tok.decode(g[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                if want_h:
                    hs = torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16)
                    hidden_store[item['id']] = hs
                rec = {'id': item['id'], 'set': item['set'], 'variant': variant, 'model': a.model,
                       'pred': letters.index(pred), 'scores': scores, 'top_token': top, 'gen': gen,
                       'n_input_tokens': int(inputs['input_ids'].shape[1])}
                fout.write(json.dumps(rec) + '\n')
                fout.flush()
            if a.hidden and (n_done + 1) % 50 == 0 and hidden_store:
                np.savez(hpath, **hidden_store)
            if (n_done + 1) % 25 == 0:
                print(f'[{a.model} {si}/{sn}] {n_done + 1}/{len(items)} items, {time.time() - t0:.0f}s', flush=True)
    if a.hidden and hidden_store:
        np.savez(hpath, **hidden_store)
    (out / f'done_shard{si}of{sn}.json').write_text(json.dumps({'items': len(items), 'sec': time.time() - t0}))
    print('DONE', a.model, si, sn, time.time() - t0)


if __name__ == '__main__':
    main()
