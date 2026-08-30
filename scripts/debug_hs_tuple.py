"""Single base-model forward: inspect the hidden_states tuple structure.
CUDA_VISIBLE_DEVICES=3 python scripts/debug_hs_tuple.py
"""
import torch
from pathlib import Path
from _theory_of_space_utils import DEFAULT_MODEL_DIR
from run_state_rev_audit import load_transformers_vl_model, setup_seeds, state_messages
from run_vetbench_screening import (CUP_DIR, decode_frames, video_processor_kwargs)
from run_hidden_state_probe import read_manifest

def main():
    setup_seeds(42)
    model, processor = load_transformers_vl_model(DEFAULT_MODEL_DIR)
    device = torch.device("cuda:0")
    rows = read_manifest(Path("outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv"))
    r = next(x for x in rows if x["trajectory_id"] == "cup_001" and x["t"] == 1)
    frames = decode_frames(CUP_DIR / r["video"])
    clip = frames[r["frame_start"]: r["frame_end"]]
    messages = state_messages(clip, r["initial_state"], r["n_swaps_shown"])
    vk = video_processor_kwargs(clip, 8.0)
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", processor_kwargs=vk)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    pos = int(inputs["input_ids"].shape[1]) - 1

    with torch.inference_mode():
        base = model.model(**inputs, output_hidden_states=True)
    hs = base.hidden_states
    lhs = base.last_hidden_state
    print("tuple len:", len(hs), " dtype:", hs[0].dtype)
    print("last_hidden_state dtype:", lhs.dtype)

    norm_w = model.model.language_model.norm.weight.float()
    eps = float(model.config.text_config.rms_norm_eps)

    def mynorm(h):
        h = h.float()
        return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w).to(h.dtype)

    a = hs[-1][0, pos].float()
    b = lhs[0, pos].float()
    print("tuple[-1] vs last_hidden_state  max_abs:", float((a - b).abs().max()))
    print("tuple[-1] vs mynorm(tuple[-1])   max_abs:", float((a - mynorm(a).float()).abs().max()))
    print("mynorm(tuple[-1]) vs last_hidden max_abs:", float((mynorm(a).float() - b).abs().max()))
    print("tuple[0] vs tuple[1]  max_abs (embed vs L1):", float((hs[0][0,pos] - hs[1][0,pos]).float().abs().max()))
    print("tuple[-2] vs tuple[-1] max_abs (L35 vs L36):", float((hs[-2][0,pos] - hs[-1][0,pos]).float().abs().max()))
    print("tuple[-1] absmean:", float(a.abs().mean()), " last_hidden absmean:", float(b.abs().mean()))
    # is tuple[-1] actually POST-norm? check: does mynorm(tuple[-2]) relate to anything?
    # try: last_hidden == mynorm(hs[-1])? already above. try last_hidden == mynorm(hs[-2])?
    print("mynorm(tuple[-2]) vs last_hidden max_abs:", float((mynorm(hs[-2][0, pos]).float() - b).abs().max()))

if __name__ == "__main__":
    main()
