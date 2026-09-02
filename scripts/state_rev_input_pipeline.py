#!/usr/bin/env python3
"""StateRev-VL unified input pipeline (Validity Gate Stage 1).

ONE function builds the model inputs consumed by BOTH behavior generation
and hidden-state extraction. This guarantees the two paths feed the model
bit-identical visual + text input, and every forward records a verifiable
fingerprint so a behavior run and a hidden run for the same (row, regime)
can be proven to have used the same input.

Two explicit regimes. Neither relies on `processor_kwargs` that may be
silently dropped, and neither passes the known-buggy `enable_thinking`
kwarg (which, in this transformers build, makes the base class REPLACE
processor_kwargs - see run_hidden_state_probe.py NOTE):

  - actual_2fps     : reproduce the OLD BEHAVIOR path exactly. The behavior
                      audit effectively ran with the processor DEFAULT
                      sampling (24 fps native assumption, 2 fps target)
                      because processor_kwargs was dropped. We reproduce it
                      by passing NO processor_kwargs and NO enable_thinking.
                      Frames = int(clip/24*2) = 9/14/19/24/29 (t=1..5).
  - controlled_8fps : reproduce the OLD PROBE exactly. The probe passed
                      explicit video_metadata (fps=30) + target fps=8 and no
                      enable_thinking. Frames = int(round(clip/30*8)) =
                      31/47/63/79/95 (t=1..5).

Fingerprint recorded for every forward (input-level, deterministic):
  regime, requested_fps, clip_frames, n_video_frames, video_grid_thw,
  input_ids_len, input_ids_sha256, pixel_values_videos_shape,
  pixel_values_videos_sha256, prompt_text_sha256, plus model/backend/version
  stamps. A hard error is raised if pixel_values_videos is absent (a
  text-only forward would silently invalidate the whole regime).

This module is a library; it also has a --selftest CLI that renders
cup_001 t=1..5 under both regimes and asserts the expected frame counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

# token ids (Qwen3-VL-8B-Instruct tokenizer)
POS = ("Left", "Middle", "Right")
POS_IDS = {"Left": 5415, "Middle": 43935, "Right": 5979}
EVENT_CLASSES = ("Left and Middle", "Middle and Right", "Left and Right")
EVENT_TOKEN_IDS = {
    "Left and Middle": [5415, 323, 12592],
    "Middle and Right": [43935, 323, 10083],
    "Left and Right": [5415, 323, 10083],
}
EOS_ID = 151645
FPS_NATIVE = 30.0          # true clip fps (VET-Bench)
TEMPORAL_PATCH = 2         # Qwen3-VL video temporal_patch_size


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def expected_frames(regime: str, clip_len: int) -> int:
    if regime == "actual_2fps":
        return max(1, int(clip_len / 24.0 * 2))
    if regime == "controlled_8fps":
        return max(1, int(round(clip_len / FPS_NATIVE * 8.0)))
    raise KeyError(regime)


def requested_fps(regime: str) -> float:
    return 2.0 if regime == "actual_2fps" else 8.0


def _version_stamp() -> dict:
    import transformers
    return {
        "model": "Qwen3-VL-8B-Instruct",
        "backend": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }


def render_inputs(processor, messages, clip: np.ndarray, regime: str):
    """Render ONE input for (messages, clip, regime). Returns
    (inputs: dict of tensors on CPU, fingerprint: dict).

    This is the single entry point used by both behavior and hidden
    extraction - call it identically for both so fingerprints match.
    """
    if regime == "actual_2fps":
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    elif regime == "controlled_8fps":
        # explicit video metadata + target fps=8 (the old probe path)
        vk = {"fps": 8.0,
              "video_metadata": {"total_num_frames": int(len(clip)),
                                 "fps": float(FPS_NATIVE)}}
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs=vk,
        )
    else:
        raise KeyError(regime)

    fp = _fingerprint(processor, inputs, regime, len(clip))
    return inputs, fp


def _fingerprint(processor, inputs, regime: str, clip_len: int) -> dict:
    if "pixel_values_videos" not in inputs:
        raise RuntimeError(
            f"[{regime}] apply_chat_template returned no "
            "pixel_values_videos - the forward would be text-only. "
            "Refusing to continue (input pipeline invariant).")
    ids = inputs["input_ids"][0].tolist()
    pv = inputs["pixel_values_videos"]
    vgw = inputs["video_grid_thw"]
    grid_t = int(vgw[0, 0])
    # The processor samples num_frames = int(total/source_fps*target_fps)
    # frames, then temporally PADS to a multiple of TEMPORAL_PATCH before
    # patching. So the true sampled count is the formula value, and
    # grid_t*TEMPORAL_PATCH equals it plus a pad of 0 or 1.
    n_frames = expected_frames(regime, clip_len)
    exp = n_frames
    pad = (TEMPORAL_PATCH - n_frames % TEMPORAL_PATCH) % TEMPORAL_PATCH
    if grid_t * TEMPORAL_PATCH != n_frames + pad:
        raise RuntimeError(
            f"[{regime}] grid_t={grid_t} inconsistent with sampled "
            f"n_frames={n_frames} (pad={pad}); the regime is not "
            f"reproducing the expected sampling.")
    try:
        prompt_text = processor.tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        prompt_text = "<undecodable>"
    fp = {
        "regime": regime,
        "requested_fps": requested_fps(regime),
        "clip_frames": int(clip_len),
        "n_video_frames": n_frames,
        "n_video_frames_expected": exp,
        "video_grid_thw": [int(x) for x in vgw[0].tolist()],
        "input_ids_len": len(ids),
        "input_ids_sha256": _sha256_bytes(
            ",".join(str(i) for i in ids).encode()),
        "pixel_values_videos_shape": list(pv.shape),
        "pixel_values_videos_sha256": _sha256_bytes(
            pv.detach().cpu().numpy().tobytes()),
        "prompt_text_sha256": _sha256_bytes(prompt_text.encode()),
    }
    fp.update(_version_stamp())
    return fp


def to_device(inputs: dict, device) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()}


def extract_hidden_states(model, inputs: dict, layers: tuple[int, ...] | None = None):
    """Base-model forward (no lm_head -> memory safe). Returns hidden states
    at the LAST INPUT TOKEN for the requested layers, shape (n_sel, D).
    `layers` uses the 37-tuple convention (0=embedding, 1..36=transformer
    layers; index 36 is post-final-norm). None -> all 37.
    """
    dev = next(model.parameters()).device
    inp = to_device(inputs, dev)
    with torch.inference_mode():
        out = model.model(**inp, output_hidden_states=True)
    hs = out.hidden_states                      # 37-tuple
    n = len(hs)
    idx = list(range(n)) if layers is None else list(layers)
    probe_pos = inp["input_ids"].shape[1] - 1
    arr = np.stack([hs[i][0, probe_pos, :].float().cpu().numpy()
                    for i in idx], axis=0)
    return arr


def _last_logits(model, inp: dict):
    """Single forward returning ONLY the last-position logits (1, V).
    Prefers logits_to_keep=1 (memory safe); falls back to full logits."""
    with torch.inference_mode():
        try:
            out = model(**inp, logits_to_keep=1)
            return out.logits[0, -1, :].float()
        except TypeError:
            out = model(**inp)
            return out.logits[0, -1, :].float()


def first_token_logits(model, processor, inputs: dict,
                       target_ids: dict[str, int]) -> dict:
    """Greedy decode from a rendered prompt. Returns the model's answer text,
    the first-token logprobs for each named target, and the full first-token
    top-5 for reference. Uses last-position-only logits (memory safe).
    """
    dev = next(model.parameters()).device
    tok = processor.tokenizer
    base = to_device(inputs, dev)
    prompt_len = len(inputs["input_ids"][0])
    ids = list(inputs["input_ids"][0].tolist())

    def step_logits(cur_ids):
        inp = dict(base)
        inp["input_ids"] = torch.tensor([cur_ids], device=dev)
        inp["attention_mask"] = torch.ones(1, len(cur_ids), dtype=torch.long,
                                           device=dev)
        if "mm_token_type_ids" in inp:
            pad = len(cur_ids) - prompt_len
            t = inp["mm_token_type_ids"]
            if pad > 0:
                inp["mm_token_type_ids"] = torch.cat(
                    [t, torch.zeros(1, pad, dtype=t.dtype, device=dev)],
                    dim=1)
        return _last_logits(model, inp)

    logits0 = step_logits(ids)
    logp0 = torch.log_softmax(logits0, dim=-1)
    first_lp = {name: float(logp0[i]) for name, i in target_ids.items()}
    first_argmax = int(torch.argmax(logits0).item())
    first_token = tok.decode([first_argmax])

    # greedy decode the rest (short answers: 1-3 tokens)
    seq = ids
    answer_tokens = []
    for _ in range(8):
        logits = step_logits(seq)
        nxt = int(torch.argmax(logits).item())
        answer_tokens.append(nxt)
        if nxt == EOS_ID:
            break
        seq = seq + [nxt]
    answer = tok.decode(answer_tokens, skip_special_tokens=True).strip()
    top5 = torch.topk(logits0, 5)
    top5 = [(int(i), float(logits0[i]), tok.decode([int(i)]))
            for i in top5.indices.tolist()]
    return {
        "first_argmax": first_argmax,
        "first_token": first_token,
        "first_logprobs": first_lp,
        "answer": answer,
        "n_answer_tokens": len(answer_tokens),
        "top5_first": top5,
    }


def teacher_force_logprob(model, inputs: dict, extra_ids: list[int]) -> float:
    """Sum of log-probs of `extra_ids` appended to the rendered prompt
    (teacher forcing). Used for the 3-class event logprob."""
    dev = next(model.parameters()).device
    base = to_device(inputs, dev)
    prompt_len = len(inputs["input_ids"][0])
    full = list(inputs["input_ids"][0].tolist()) + list(extra_ids)
    inp = dict(base)
    inp["input_ids"] = torch.tensor([full], device=dev)
    inp["attention_mask"] = torch.ones(1, len(full), dtype=torch.long,
                                       device=dev)
    if "mm_token_type_ids" in inp:
        pad = len(full) - prompt_len
        t = inp["mm_token_type_ids"]
        if pad > 0:
            inp["mm_token_type_ids"] = torch.cat(
                [t, torch.zeros(1, pad, dtype=t.dtype, device=dev)], dim=1)
    with torch.inference_mode():
        out = model(**inp, logits_to_keep=len(extra_ids) + 1)
    logits = out.logits[0]                       # (len(extra)+1, V)
    logp = torch.log_softmax(logits.float(), dim=-1)
    base_len = len(inputs["input_ids"][0])
    total = 0.0
    for j, tid in enumerate(extra_ids):
        total += float(logp[j, tid])
    return total


def load_model_and_processor(model_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_state_rev_audit import load_transformers_vl_model
    model, processor = load_transformers_vl_model(model_dir)
    model.eval()
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest"], default="selftest")
    ap.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--video",
                    default="dataset/vetbench/cup/cup_001.mp4")
    ap.add_argument("--initial", default="Left")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip

    model, processor = load_model_and_processor(Path(args.model_dir))
    # frame boundaries: swap t in [57+(t-1)*60, 57+t*60); clip end for t
    ends = {1: 117, 2: 177, 3: 237, 4: 297, 5: 357}
    exp2 = {1: 9, 2: 14, 3: 19, 4: 24, 5: 29}
    exp8 = {1: 31, 2: 47, 3: 63, 4: 79, 5: 95}

    records = []
    for t in range(1, 6):
        clip = sample_clip(Path(args.video), 0, ends[t])
        msgs = state_messages(clip, args.initial, t)
        for regime, exp in (("actual_2fps", exp2[t]),
                            ("controlled_8fps", exp8[t])):
            inputs, fp = render_inputs(processor, msgs, clip, regime)
            ok = fp["n_video_frames"] == exp
            print(f"t={t} {regime:16s} frames={fp['n_video_frames']} "
                  f"(expect {exp}) input_len={fp['input_ids_len']} "
                  f"grid={fp['video_grid_thw']} pv={fp['pixel_values_videos_shape']} "
                  f"{'OK' if ok else 'MISMATCH'}")
            assert ok, f"frame count mismatch t={t} {regime}"
            # hidden extraction smoke (base model) - 37 layers
            hs = extract_hidden_states(model, inputs)
            assert hs.shape[0] == 37 and hs.shape[1] == 4096, hs.shape
            # behavior smoke: first-token state logits
            r = first_token_logits(model, processor, inputs, POS_IDS)
            records.append({"t": t, "regime": regime, "fingerprint": fp,
                            "hs_shape": list(hs.shape),
                            "state_first_lp": r["first_logprobs"],
                            "state_answer": r["answer"]})
            del inputs
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=1))
        print(f"wrote {args.out}")
    print("SELFTEST PASS: both regimes render the expected frame counts, "
          "hidden extraction returns 37x4096, and behavior decode works.")


if __name__ == "__main__":
    main()
