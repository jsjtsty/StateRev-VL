"""Probe-position consistency audit for the hidden-state extraction.

Reconstructs, for a random sample of manifest rows, EXACTLY what
run_hidden_state_probe.collect_hidden_states feeds into the model:

    messages  = state_messages(clip, initial_state, n_swaps_shown)
    inputs    = processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt", processor_kwargs=video_processor_kwargs(clip, fps))
    probe_pos = inputs["input_ids"].shape[1] - 1
    outputs   = model.model(**inputs, output_hidden_states=True)   # ONE forward

and reports, per sample:
  - total input_ids length, probe_position index, probe token id + decode
  - 5 tokens before / after the probe token
  - the last 30 tokens of the rendered chat template
  - the rendered prompt tail (verbatim text)
  - structural check that the prompt ends with the fixed template tail
    (option list + example) immediately before the assistant turn, i.e. no
    generated answer and no per-row GT label is present in the input
  - with --verify-npz: re-runs the single forward on the model and compares
    the re-extracted probe-position hidden states against the saved
    hidden_states.npz (proves the saved npz was produced by this exact
    prompt-only, pre-generation forward path)

Usage (physical GPU 3 only when --verify-npz):
  python scripts/audit_probe_position.py                      # token-level, no model
  CUDA_VISIBLE_DEVICES=3 python scripts/audit_probe_position.py --verify-npz
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR
from run_hidden_state_probe import MANIFEST, OUT_DIR, read_manifest
from run_state_rev_audit import load_transformers_vl_model, setup_seeds, state_messages
from run_vetbench_screening import decode_frames, video_processor_kwargs

# The rendered state-question prompt must end with exactly this tail, then the
# assistant generation prompt. Anything else means the prompt content drifted.
EXPECTED_PROMPT_TAIL = ('The ball is under the cup that is currently at one of the '
                        'three positions. Which position is the cup that currently '
                        'contains the ball at? '
                        '(A) Left (B) Middle (C) Right. '
                        'Answer with the option text, e.g. "Left".')


def build_inputs(processor, row: dict, sample_fps: float) -> tuple[dict, object, Path]:
    prefix = Path(row["prefix_path"])
    video_path = prefix if prefix.is_absolute() else Path.cwd() / prefix
    container_frames = decode_frames(video_path)
    clip = container_frames[row["frame_start"]: row["frame_end"]]
    messages = state_messages(clip, row["initial_state"], row["n_swaps_shown"])
    vk = video_processor_kwargs(clip, sample_fps)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", processor_kwargs=vk,
    )
    return inputs, messages, video_path


def audit_sample(processor, row: dict, sample_fps: float, tok) -> dict:
    inputs, _messages, _video_path = build_inputs(processor, row, sample_fps)
    ids = inputs["input_ids"][0]
    seq_len = int(ids.shape[0])
    probe_pos = seq_len - 1
    probe_id = int(ids[probe_pos])

    # probe_pos is the LAST input token, so "after" tokens exist only as a
    # note: nothing follows it in the input (generation starts here).
    ctx = []
    for i in range(max(0, probe_pos - 5), seq_len):
        tid = int(ids[i])
        ctx.append((i, tid, tok.decode([tid], skip_special_tokens=False)))

    last30 = [(int(ids[i]), tok.decode([int(ids[i])], skip_special_tokens=False))
              for i in range(seq_len - 30, seq_len)]

    rendered_tail = tok.decode(ids, skip_special_tokens=False)
    # structural checks on the rendered text: the prompt must end with the
    # fixed template tail followed by ONLY the assistant generation prompt
    # (no content tokens after the question).
    # Qwen3-VL template with add_generation_prompt=True renders the input tail as:
    #   <user text> + <im_end marker><NL> + <im_start marker>assistant<NL>
    # where the markers are the pipe-style special tokens (no embedded newline; the
    # newline is a separate token 198).  Built with chr() to avoid writing them raw.
    nl = "\n"
    pipe = chr(124)
    im_end = chr(60) + pipe + "im_end" + pipe + chr(62)
    im_start = chr(60) + pipe + "im_start" + pipe + chr(62)
    gen_prompt = im_start + "assistant" + nl
    user_end = im_end + nl
    tail_ok = (rendered_tail.endswith(gen_prompt)
               and rendered_tail[: -len(gen_prompt)].endswith(user_end)
               and rendered_tail[: -len(gen_prompt) - len(user_end)].endswith(EXPECTED_PROMPT_TAIL))
    asst_idx = rendered_tail.rfind("assistant")
    after_asst = rendered_tail[asst_idx:] if asst_idx >= 0 else "<assistant marker not found>"
    return {
        "key": f"{row['trajectory_id']}_t{row['t']}",
        "gt_state": row["gt_state"], "gt_event": row["gt_event"],
        "seq_len": seq_len, "probe_pos": probe_pos,
        "probe_id": probe_id, "probe_token": tok.decode([probe_id], skip_special_tokens=False),
        "context": ctx, "last30": last30,
        "tail_ok": tail_ok, "after_assistant_marker": after_asst,
        "rendered_tail_200": rendered_tail[-200:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-fps", type=float, default=8.0)
    ap.add_argument("--verify-npz", action="store_true",
                    help="load the model, re-forward the sampled rows and compare "
                         "against the saved hidden_states.npz")
    ap.add_argument("--model-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "models" / "Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = read_manifest(MANIFEST)
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(rows, args.n_samples),
                    key=lambda r: (r["trajectory_id"], r["t"]))
    print(f"Auditing {len(sample)} random samples from {len(rows)} manifest rows "
          f"(rng seed {args.seed})")

    model, processor = (None, None)
    if args.verify_npz:
        setup_seeds(42)  # same seed main() uses before extraction
        model, processor = load_transformers_vl_model(args.model_dir)
        device = torch.device(args.device)
        with np.load(OUT_DIR / "hidden_states.npz") as z:
            saved = {k: z[k] for k in z.files}
    else:
        # token-level audit needs no model: load only the processor
        from transformers import AutoProcessor
        device = None
        processor = AutoProcessor.from_pretrained(args.model_dir)

    tok = processor.tokenizer
    all_ok = True
    for row in sample:
        info = audit_sample(processor, row, args.sample_fps, tok)
        all_ok &= info["tail_ok"]
        print(f"\n=== {info['key']}  (gt_state={info['gt_state']}, gt_event={info['gt_event']}) ===")
        print(f"input_ids seq_len        : {info['seq_len']}")
        print(f"probe_position index     : {info['probe_pos']}  (= seq_len - 1, last input token)")
        print(f"probe token id / decode  : {info['probe_id']} / {info['probe_token']!r}")
        print("5 tokens before the probe (probe is the LAST input token; nothing follows it "
              "in the input - generation starts at the next token):")
        for i, tid, dec in info["context"]:
            mark = "  <-- probe (last input token)" if i == info["probe_pos"] else ""
            print(f"   [{i:>6}] id={tid:<8} {dec!r}{mark}")
        if info["probe_pos"] + 1 < info["seq_len"]:
            print(f"   [{info['probe_pos'] + 1:>6}] UNEXPECTED token after probe position!")
        print("last 30 tokens (id, decode):")
        print("   " + " ".join(f"{i}:{d!r}" for i, d in info["last30"]))
        print(f"rendered prompt ends with fixed template tail: {info['tail_ok']}")
        print(f"text from last 'assistant' marker to end: {info['after_assistant_marker']!r}")
        print(f"rendered prompt (last 200 chars): {info['rendered_tail_200']!r}")

        if args.verify_npz:
            inputs, _m, _v = build_inputs(processor, row, args.sample_fps)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.model(**inputs, output_hidden_states=True)
            hs = torch.stack([h[0, info["probe_pos"], :] for h in out.hidden_states],
                             dim=0).float().cpu().numpy()
            ref = saved[info["key"]]
            rel = float(np.linalg.norm(hs - ref) / np.linalg.norm(ref))
            maxabs = float(np.abs(hs - ref).max())
            match = rel < 1e-3
            all_ok &= match
            print(f"npz compare: layers={hs.shape[0]}  rel_L2={rel:.3e}  max_abs={maxabs:.3e}  "
                  f"{'MATCH (same prompt-only single-forward path)' if match else 'MISMATCH!'}")
            del inputs, out
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    if all_ok:
        print("AUDIT PASS: prompt ends with the fixed template tail before the "
              "assistant turn; no generated answer in the input."
              + ("  Re-forwarded hidden states match the saved npz." if args.verify_npz else ""))
    else:
        print("AUDIT FAIL: see per-sample output above.")


if __name__ == "__main__":
    main()
