"""Direct alignment debug: for a few rows compare, from ONE forward pass,
  (a) saved-npz hidden state @ finalNorm @ lm_head  (the native readout)
  (b) model.forward full-vocab logits at the probe position
  (c) model.generate greedy first token
to localize any extraction/position/dtype mismatch.
Run: CUDA_VISIBLE_DEVICES=3 python scripts/debug_native_alignment.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from run_state_rev_audit import load_transformers_vl_model, setup_seeds, state_messages
from run_vetbench_screening import (CUP_DIR, decode_frames, video_processor_kwargs,
                                    derive_ground_truth, load_metadata, POSITION_NAMES,
                                    prefix_frame_range)
from run_hidden_state_probe import MANIFEST, OUT_DIR, read_manifest

STATE_WORDS = ("Left", "Middle", "Right")

def main():
    setup_seeds(42)
    model, processor = load_transformers_vl_model(DEFAULT_MODEL_DIR)
    device = torch.device("cuda:0")
    tok = processor.tokenizer
    word_ids = {w: int(tok.encode(w, add_special_tokens=False)[0]) for w in STATE_WORDS}
    print("word ids:", word_ids)

    rows = read_manifest(MANIFEST)
    with np.load(OUT_DIR / "hidden_states.npz") as z:
        saved = {k: z[k] for k in z.files}

    norm_w = model.model.language_model.norm.weight.float()
    eps = float(model.config.text_config.rms_norm_eps)
    W = model.lm_head.weight  # (vocab, 4096)

    gt_by_video = {e["video"]: derive_ground_truth(e) for e in load_metadata()}
    picks = [r for r in rows if r["trajectory_id"] in ("cup_001", "cup_003")
             and r["t"] in (1, 2)][:4]
    for r in picks:
        video = r["video"]
        video_path = CUP_DIR / video
        frames = decode_frames(video_path)
        clip = frames[r["frame_start"]: r["frame_end"]]
        messages = state_messages(clip, r["initial_state"], r["n_swaps_shown"])
        vk = video_processor_kwargs(clip, 8.0)
        inputs = processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs=vk)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        pos = int(inputs["input_ids"].shape[1]) - 1

        # (b) real model forward logits at probe position
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        real_logit = out.logits[0, pos, :].float()          # (vocab,)
        real_hs = out.hidden_states[-1][0, pos, :].float()  # final layer, pre-norm
        # (a) native readout from SAVED npz
        key = f"{r['trajectory_id']}_t{r['t']}"
        hs_saved = torch.from_numpy(saved[key][-1]).float().to(device)
        hs_live  = real_hs.to(device)
        hs_diff = float((hs_saved - hs_live).abs().max())
        def native(hs):
            h = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + eps) * norm_w.to(device)
            return h @ W.to(device).float().T
        nat_logit = native(hs_saved)
        nat_live  = native(hs_live)
        # (c) greedy generate first token
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=3, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new_ids = gen[0, inputs["input_ids"].shape[1]:]
        first_tok_id = int(new_ids[0])
        first_tok = tok.decode([first_tok_id], skip_special_tokens=False)

        rl = real_logit.cpu()
        nl = nat_logit.cpu()
        nlive = nat_live.cpu()
        top_real = int(rl.argmax())
        top_nat  = int(nl.argmax())
        s3_real = {w: float(rl[word_ids[w]]) for w in STATE_WORDS}
        s3_nat  = {w: float(nl[word_ids[w]]) for w in STATE_WORDS}
        print(f"\n=== {key}  gt={r['gt_state']}  probe_pos={pos}")
        print(f"  saved_vs_live final-layer hs max_abs_diff = {hs_diff:.3e}")
        print(f"  native(saved) vs native(live) max_abs_diff = {float((nl-nlive).abs().max()):.3e}")
        print(f"  REAL logits  top1 id={top_real} {tok.decode([top_real],skip_special_tokens=False)!r}")
        print(f"  NATIVE(saved) top1 id={top_nat} {tok.decode([top_nat],skip_special_tokens=False)!r}")
        print(f"  REAL state-word logits: " + " ".join(f"{w}={v:.2f}" for w,v in s3_real.items()))
        print(f"  NATIVE state-word logits: " + " ".join(f"{w}={v:.2f}" for w,v in s3_nat.items()))
        all3 = [tok.decode([int(i)], skip_special_tokens=False) for i in new_ids]
        print(f"  GREEDY first token: id={first_tok_id} {first_tok!r}  (all3={all3})")

if __name__ == "__main__":
    main()
