"""Pin down the exact logit path: compare
  R1 = model.logits (real)
  R2 = lm_head(outputs[0])            (outputs[0] = last_hidden_state, post-norm)
  R3 = lm_head(finalNorm(hidden_states[-1]))   (my native readout)
and compare the vectors outputs[0] vs finalNorm(hidden_states[-1]).
"""
import torch
from _theory_of_space_utils import DEFAULT_MODEL_DIR
from run_state_rev_audit import load_transformers_vl_model, setup_seeds, state_messages
from run_vetbench_screening import (CUP_DIR, decode_frames, video_processor_kwargs,
                                    derive_ground_truth, load_metadata)
from run_hidden_state_probe import OUT_DIR, read_manifest

STATE_WORDS = ("Left", "Middle", "Right")

def main():
    setup_seeds(42)
    model, processor = load_transformers_vl_model(DEFAULT_MODEL_DIR)
    device = torch.device("cuda:0")
    tok = processor.tokenizer
    word_ids = {w: int(tok.encode(w, add_special_tokens=False)[0]) for w in STATE_WORDS}

    from pathlib import Path
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
        out = model(**inputs, output_hidden_states=True)
        base = model.model(**{k: v for k, v in inputs.items()})
    R1 = out.logits[0, pos].float()
    last_normed = base.last_hidden_state[0, pos].float()        # Qwen3VLModel outputs[0]
    R2 = model.lm_head(last_normed.unsqueeze(0).to(model.lm_head.weight.dtype))[0].float()
    h36 = out.hidden_states[-1][0, pos].float()                  # pre-norm

    norm_w = model.model.language_model.norm.weight
    eps = float(model.config.text_config.rms_norm_eps)
    # replicate Qwen3RMSNorm exactly in fp32
    h = h36 / torch.sqrt(h36.pow(2).mean(-1, keepdim=True) + eps)
    R3vec = h * norm_w.float()
    R3 = model.lm_head(R3vec.unsqueeze(0).to(model.lm_head.weight.dtype))[0].float()

    print("max|R1-R2| (real vs lm_head(last_hidden_state)) =", float((R1-R2).abs().max()))
    print("max|R1-R3| (real vs lm_head(myFinalNorm(h36)))  =", float((R1-R3).abs().max()))
    print("max|last_hidden_state - myFinalNorm(h36)|       =", float((last_normed-R3vec).abs().max()))
    print("dtype of out.logits =", out.logits.dtype, " last_hidden_state =", out.last_hidden_state.dtype)
    print("dtype of hidden_states[-1] =", out.hidden_states[-1].dtype)
    print("norm weight norm =", float(norm_w.float().norm()), " first5 =", [float(v) for v in norm_w.float()[:5]])
    print("h36 absmean =", float(h36.abs().mean()), " last_normed absmean =", float(last_normed.abs().mean()))
    for name, v in (("R1 real", R1), ("R2 lhd", R2), ("R3 mynorm", R3)):
        s = {w: float(v[word_ids[w]]) for w in STATE_WORDS}
        print(f"  {name}: " + " ".join(f"{w}={x:.2f}" for w, x in s.items()) +
              f"  top1={tok.decode([int(v.argmax())], skip_special_tokens=False)!r}")

if __name__ == "__main__":
    main()
