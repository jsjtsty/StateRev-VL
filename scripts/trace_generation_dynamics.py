"""Experiment 3: trace WHEN the model commits to a position during generation.

Goal
----
The behavior screen shows 26 canonical "stale" failures: the model answers
S_{t-1} (the pre-swap state) even though it correctly reports the event and
the prior state. Experiments 1-2 ask whether stale information is still
linearly/nonlinearly readable after the swap and whether supplying the prior
or the event rescues the answer. This experiment asks a different question:

    During the actual answer generation, at which generation step does the
    model commit to a position word, and what do the native
    Left/Middle/Right logits look like step by step?

If the model "decides" the (wrong) answer before it has processed the swap
evidence - or if the stale position's logit is already dominant at the first
generated token - that points to an early commitment / retrieval bias rather
than a gradual inference failure. Conversely, if the correct logit leads at
early steps and is lost later, that points to a late-stage override.

Causal-safety design (the critical property)
--------------------------------------------
The decode loop is written by hand instead of calling ``model.generate``.
At generation step i the forward pass receives EXACTLY

    [prompt_tokens + generated_tokens_0..i-1]

i.e. the model never sees any token that has not already been emitted. This
makes the trace causal *by construction*: every recorded hidden state and
logit vector at step i is a function of prompt + past tokens only, so no
future information (later generated tokens, the "rest" of the video, etc.)
can leak into the trace. No KV-cache is used; the whole prefix is re-fed at
each step, so there is no cache-state aliasing between steps either.

The loop discipline itself is verified WITHOUT any GPU/model by
``--mode selftest``: a deliberately BIDIRECTIONAL toy model (every position
depends on every other token of the current input) is driven through the
same loop. Two runs share the same prompt; run B is forced to emit a
different token at step k. Because the loop only feeds past tokens:

  * records 0..k must be BIT-IDENTICAL between the two runs (same input);
  * records after k+1 may differ (the differing token is now in the past).

A "broken" variant of the loop that additionally feeds the full future
tail (simulating a leaky implementation) is run as a mutation test: it MUST
produce differing early records, proving the selftest would actually catch
a causality bug.

Modes
-----
  selftest   CPU, no model: toy-model causal discipline test (run this now).
  run        Requires --model-dir; loads Qwen3-VL on GPU (run later when the
             machine is free). Greedy decoding, identical prompt to the
             behavior audit baseline (thinking=off, same state question).
             Video sampling REPLICATES THE BEHAVIOR AUDIT'S EFFECTIVE
             SAMPLING: the apply_chat_template call below mirrors
             run_inference (including the enable_thinking kwarg, which makes
             the base class drop processor_kwargs -> processor default
             ~int(clip/24*2) frames, NOT the nominal 8 fps). Do NOT
             "fix" the kwarg without re-running the audit, or the trace
             would no longer explain the recorded answers. See
             build_temporal_intervention_manifest.py for the two sampling
             regimes (ACTUAL vs NOMINAL).

Outputs (run mode, in --out-dir)
--------------------------------
  generation_trace.csv    one row per (sample, generation step):
                          gen_index, token_id, token_text,
                          logit_Left, logit_Middle, logit_Right,
                          margin_current_minus_prev (logit S_t - logit S_{t-1}
                          at that step), top-k ids/logprobs, and flags for
                          whether the token text equals the GT current /
                          previous state word.
  generation_hidden.npz   per-step last-position hidden states for the
                          requested layers: array (n_samples, max_steps,
                          n_layers, hidden_dim), zero-padded; per-sample
                          lengths in generation_hidden_shape.json.
  generation_trace_summary.json  config + sample list + n steps per sample.

Usage
-----
  python scripts/trace_generation_dynamics.py --mode selftest
  python scripts/trace_generation_dynamics.py --mode run \
      --model-dir models/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path("outputs/vetbench/stale_origin_analysis_v1")
POSITIONS = ("Left", "Middle", "Right")
GEN_MAX_NEW_TOKENS = 32  # matches the behavior audit config (greedy, thinking off)
DEFAULT_LAYERS = (36,)  # final layer by default
DEFAULT_TOPK = 8


# ---------------------------------------------------------------------------
# Generic step-by-step decode loop (causal by construction)
# ---------------------------------------------------------------------------

def trace_decode(model_forward, prompt_ids: list[int], max_new_tokens: int,
                 topk: int, forced: dict[int, int] | None = None,
                 stop_token: int | None = None) -> list[dict]:
    """Greedy decode with per-step recording.

    ``model_forward(ids) -> (logits, [hidden_per_layer])`` where both refer
    to the LAST position of the (causal) input ``ids``.

    At step i the input is ``prompt_ids + generated[0:i]`` - past tokens
    only. ``forced`` maps a step index to a token id to emit at that step
    (used by the selftest; the logits at that step are still recorded from
    the unforced input).
    """
    records: list[dict] = []
    ids = list(prompt_ids)
    for i in range(max_new_tokens):
        logits, hiddens = model_forward(ids)
        logits = np.asarray(logits, dtype=np.float64)
        if forced is not None and i in forced:
            tok = int(forced[i])
        else:
            tok = int(np.argmax(logits))
        topk_ids = np.argsort(logits)[::-1][:topk]
        records.append({
            "gen_index": i,
            "token_id": tok,
            "logits": logits,
            "hiddens": [np.asarray(h, dtype=np.float64) for h in hiddens],
            "topk_ids": [int(x) for x in topk_ids],
            "topk_logprobs": [float(logits[x]) for x in topk_ids],
        })
        if stop_token is not None and tok == stop_token:
            break
        ids.append(tok)
    return records


def trace_decode_broken(model_forward, prompt_ids: list[int],
                        full_tail: list[int], topk: int) -> list[dict]:
    """MUTATION-TEST variant: feeds prompt + past + the FULL future tail.

    This simulates a leaky implementation (e.g. a cached/bidirectional
    forward that sees tokens that have not been generated yet). It exists
    only to prove that the selftest's bit-identity check would catch such a
    leak; it must never be used for real traces.
    """
    records: list[dict] = []
    for i in range(len(full_tail)):
        # past part, then the ENTIRE remaining tail (every future token)
        ids = list(prompt_ids) + full_tail[:i] + full_tail[i:]
        logits, hiddens = model_forward(ids)
        logits = np.asarray(logits, dtype=np.float64)
        tok = int(np.argmax(logits))
        topk_ids = np.argsort(logits)[::-1][:topk]
        records.append({
            "gen_index": i,
            "token_id": tok,
            "logits": logits,
            "hiddens": [np.asarray(h, dtype=np.float64) for h in hiddens],
            "topk_ids": [int(x) for x in topk_ids],
            "topk_logprobs": [float(logits[x]) for x in topk_ids],
        })
    return records


# ---------------------------------------------------------------------------
# Toy bidirectional model for the causal selftest
# ---------------------------------------------------------------------------

def _make_toy_model(vocab: int = 12, dim: int = 8, n_layers: int = 2,
                    seed: int = 0):
    """A small NON-CAUSAL model: every position mixes in a global sum of all
    input embeddings, so any change to any token (past or future) changes
    every position's representation. If the decode loop respected
    past-only inputs, two runs that only differ in a future token must still
    produce identical early records; if the loop leaked future tokens, the
    early records would differ. Position dependence is added via a learned
    per-position term so argmax produces a non-degenerate token stream."""
    import torch
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)

    class ToyBi(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.W = nn.Parameter(torch.randn(n_layers, dim, dim, generator=g) * 0.5)
            self.pe = nn.Parameter(torch.randn(n_layers, dim, generator=g) * 0.5)
            self.out = nn.Linear(dim, vocab)

        def forward(self, ids: list[int]):
            t = torch.tensor(ids, dtype=torch.long)
            x = self.emb(t)                       # (T, D)
            pos = torch.arange(t.size(0), dtype=torch.float32)
            mix = x.sum(dim=0, keepdim=True)      # (1, D) global -> bidirectional
            h = x
            H = []
            for l in range(n_layers):
                h = torch.tanh(h @ self.W[l].T + mix + self.pe[l] * pos[:, None])
                H.append(h)
            logits = self.out(H[-1])
            last = logits[-1].detach().numpy()
            hs = [h[-1].detach().numpy() for h in H]
            return last, hs

    return ToyBi()


def selftest_causality(topk: int = DEFAULT_TOPK, n_steps: int = 4,
                       k: int = 2, vocab: int = 12) -> bool:
    """Verify the trace_decode loop is causal by construction.

    Returns True on success, False on failure (exit code 1 from main).
    """
    import torch
    torch.manual_seed(0)
    model = _make_toy_model(vocab=vocab, seed=0)
    model.eval()
    prompt = [1, 2, 3, 4, 5]

    def fwd(ids):
        with torch.no_grad():
            return model(ids)

    run_a = trace_decode(fwd, prompt, n_steps, topk=topk)
    forced_tok = (run_a[k]["token_id"] + 1) % vocab
    run_b = trace_decode(fwd, prompt, n_steps, topk=topk, forced={k: forced_tok})

    ok = True
    # 1) steps 0..k-1: identical past => token + logits + hiddens identical
    for i in range(k):
        a, b = run_a[i], run_b[i]
        same = (a["token_id"] == b["token_id"]
                and np.array_equal(a["logits"], b["logits"])
                and all(np.array_equal(x, y)
                        for x, y in zip(a["hiddens"], b["hiddens"])))
        print(f"  step {i}: past-identical => records identical: {same}")
        ok &= same
    # step k: same input (same past) => identical logits/hiddens; the emitted
    # token differs only because run B is forced at this step
    a, b = run_a[k], run_b[k]
    same_k = (np.array_equal(a["logits"], b["logits"])
              and all(np.array_equal(x, y)
                      for x, y in zip(a["hiddens"], b["hiddens"]))
              and a["token_id"] != b["token_id"])
    print(f"  step {k}: same input => identical logits/hiddens, forced "
          f"token differs as expected: {same_k}")
    ok &= same_k
    # 2) greedy argmax consistency in run A
    for i, rec in enumerate(run_a):
        consistent = int(np.argmax(rec["logits"])) == rec["token_id"]
        print(f"  step {i}: argmax(recorded logits) == emitted token: {consistent}")
        ok &= consistent
    # 3) after the forced step, records may (and for this toy model do) differ
    differ_later = any(
        not np.array_equal(run_a[i]["logits"], run_b[i]["logits"])
        for i in range(k + 1, n_steps))
    print(f"  steps > k differ between runs (divergence visible): {differ_later}")
    ok &= differ_later
    # 4) mutation test: a leaky loop (feeds the future tail) MUST show
    #    different early records between the two continuations
    tail_a = [r["token_id"] for r in run_a]
    tail_b = tail_a[:k] + [forced_tok] + tail_a[k + 1:]
    broken_a = trace_decode_broken(fwd, prompt, tail_a, topk=topk)
    broken_b = trace_decode_broken(fwd, prompt, tail_b, topk=topk)
    early_diff = any(
        not np.array_equal(broken_a[i]["logits"], broken_b[i]["logits"])
        for i in range(k))
    print(f"  mutation (leaky loop) shows early-record divergence: {early_diff}")
    ok &= early_diff

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


def _dump_hidden(out_dir: Path, samples: list[dict], hid: dict,
                 lengths: dict, layers: tuple[int, ...]) -> np.ndarray:
    """(Re)write the hidden-state npz for the samples collected so far.
    Called every 10 samples so a crash does not lose the trace."""
    n_s = len(samples)
    max_steps = max(lengths[s["trajectory_id"] + "_t" + str(s["t"])]
                    for s in samples) if samples else 0
    n_l = len(layers)
    dim = next(iter(hid.values()))[0][0].shape[0] if hid else 0
    arr = np.zeros((n_s, max_steps, n_l, dim), dtype=np.float32)
    for si, s in enumerate(samples):
        sid = f"{s['trajectory_id']}_t{s['t']}"
        for li in range(n_l):
            for gi in range(lengths[sid]):
                arr[si, gi, li, :] = hid[sid][li][gi]
    np.savez_compressed(out_dir / "generation_hidden.npz", hidden=arr)
    return arr


# ---------------------------------------------------------------------------
# Run mode (future; requires GPU + model dir - NOT run in this session)
# ---------------------------------------------------------------------------

def _hf_forward_factory(model, layers: tuple[int, ...], base_inputs: dict,
                        prompt_len: int):
    """Adapter so the generic loop can drive a HuggingFace causal LM:
    re-feeds the full text prefix at every step (no KV cache) together
    with the (unchanging) visual inputs - pixel_values(_videos) /
    video_grid_thw MUST be passed, otherwise the model answers from
    text alone. mm_token_type_ids is zero-extended for generated tokens
    (text type)."""
    import torch

    def fwd(ids: list[int]):
        inp = torch.tensor([ids], device=model.device)
        extra = dict(base_inputs)
        extra["attention_mask"] = torch.ones(
            1, len(ids), dtype=torch.long, device=model.device)
        if "mm_token_type_ids" in extra:
            pad = len(ids) - prompt_len
            t = extra["mm_token_type_ids"]
            if pad > 0:
                extra["mm_token_type_ids"] = torch.cat(
                    [t, torch.zeros(1, pad, dtype=t.dtype,
                                     device=t.device)], dim=1)
        with torch.no_grad():
            out = model(**extra, input_ids=inp,
                        output_hidden_states=True)
        logits = out.logits[0, -1, :].float().cpu().numpy()
        hs = [out.hidden_states[L][0, -1, :].float().cpu().numpy()
              for L in layers]
        return logits, hs

    return fwd


def run_model(args) -> None:
    import torch  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_state_rev_audit import load_transformers_vl_model
    from run_vetbench_screening import sample_clip, video_processor_kwargs
    from run_state_rev_audit import state_messages
    from run_revision_rescue import load_samples

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(int(x) for x in args.layers.split(","))
    samples = load_samples(Path(args.behavior_csv),
                           controls={"maintenance", "other_failure", "rest"})
    if args.limit_samples:
        samples = samples[:args.limit_samples]
    print(f"Tracing {len(samples)} samples x {GEN_MAX_NEW_TOKENS} max steps "
          f"(layers={layers}) ...", flush=True)

    model, processor = load_transformers_vl_model(Path(args.model_dir))
    model.eval()
    tok = processor.tokenizer
    pos_ids = {}
    for name in POSITIONS:
        ids = tok.encode(name, add_special_tokens=False)
        if len(ids) != 1:
            print(f"WARNING: {name!r} encodes to {len(ids)} tokens; using first")
        pos_ids[name] = int(ids[0])
    eos_id = tok.eos_token_id

    import csv
    rows = []
    hid = {}
    lengths = {}
    input_lens = {}
    csv_path = out_dir / "generation_trace.csv"
    cf = open(csv_path, "w", newline="")
    w = None
    for si, s in enumerate(samples):
        sid = f"{s['trajectory_id']}_t{s['t']}"
        clip = sample_clip(Path(s["prefix_path"]), 0, int(s["frame_end"]))
        messages = state_messages(
            clip, s["initial_state"], int(s["n_swaps_shown"]))
        # NOTE: mirrors run_inference exactly (enable_thinking +
        # processor_kwargs). The enable_thinking kwarg makes the base class
        # drop processor_kwargs, so the effective video sampling is the
        # processor default (metadata fps=24, target fps=2) - identical to
        # the behavior audit. Intentional: the trace must use the same
        # visual input that produced the recorded answers.
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
            processor_kwargs=video_processor_kwargs(
                clip, float(s["sample_fps"])))
        prompt_ids = inputs["input_ids"][0].tolist()
        input_lens[sid] = len(prompt_ids)
        base_inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask")}
        if not any(k in base_inputs for k in
                   ("pixel_values", "pixel_values_videos")):
            raise RuntimeError(
                "apply_chat_template returned no visual tensors - the "
                "trace would be text-only; aborting")
        fwd = _hf_forward_factory(model, layers, base_inputs, len(prompt_ids))
        recs = trace_decode(fwd, prompt_ids, GEN_MAX_NEW_TOKENS,
                            topk=args.topk, stop_token=eos_id)
        lengths[sid] = len(recs)
        for r in recs:
            txt = tok.decode([r["token_id"]], skip_special_tokens=True)
            row = {
                "sample_id": sid,
                "trajectory_id": s["trajectory_id"],
                "t": s["t"],
                "group": s["group"],
                "gt_prev_state": s["gt_prev_state"],
                "gt_state": s["gt_state"],
                "gen_index": r["gen_index"],
                "token_id": r["token_id"],
                "token_text": txt,
                "logit_Left": float(r["logits"][pos_ids["Left"]]),
                "logit_Middle": float(r["logits"][pos_ids["Middle"]]),
                "logit_Right": float(r["logits"][pos_ids["Right"]]),
                "margin_current_minus_prev": float(
                    r["logits"][pos_ids[s["gt_state"]]]
                    - r["logits"][pos_ids[s["gt_prev_state"]]]),
                "topk_ids": ";".join(map(str, r["topk_ids"])),
                "topk_logprobs": ";".join(
                    f"{x:.4f}" for x in r["topk_logprobs"]),
                "is_current_state_word": txt.strip().lower()
                    == s["gt_state"].lower(),
                "is_prev_state_word": txt.strip().lower()
                    == s["gt_prev_state"].lower(),
            }
            rows.append(row)
            # incremental save: a crash must not lose finished samples
            if w is None:
                w = csv.DictWriter(cf, fieldnames=list(row.keys()))
                w.writeheader()
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
        cf.flush()
        # stow per-step hidden for this sample
        hid[sid] = [[r["hiddens"][li] for r in recs] for li in range(len(layers))]
        print(f"  [{si + 1}/{len(samples)}] {sid}: {len(recs)} steps", flush=True)
        if (si + 1) % 10 == 0:
            _dump_hidden(out_dir, samples[:si + 1], hid, lengths, layers)
    cf.close()
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    # write hidden npz: (n_samples, max_steps, n_layers, hidden_dim)
    arr = _dump_hidden(out_dir, samples, hid, lengths, layers)
    shape_meta = {
        "shape": list(arr.shape),
        "axes": ["sample (manifest order)", "gen_index (0-padded)",
                 "layer", "hidden_dim"],
        "layers": list(layers),
        "lengths": lengths,
        "note": "zero-padded to max_steps; use lengths[sample_id]",
    }
    with open(out_dir / "generation_hidden_shape.json", "w") as f:
        json.dump(shape_meta, f, indent=2)

    summary = {
        "mode": "run",
        "n_samples": len(samples),
        "groups": {g: sum(1 for s in samples if s["group"] == g)
                   for g in sorted({s["group"] for s in samples})},
        "layers": list(layers),
        "topk": args.topk,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False), thinking off, manual loop: "
                    "at step i the model sees exactly prompt + tokens 0..i-1 "
                    "(no KV cache), so every recorded state is a function of "
                    "prompt + past tokens only (causal by construction)",
        "position_token_ids": pos_ids,
        "eos_token_id": eos_id,
        "n_steps_per_sample": lengths,
        "input_len_per_sample": input_lens,
        "video_sampling": (
            "ACTUAL behavior-audit path: run_inference-style "
            "apply_chat_template (enable_thinking kwarg drops "
            "processor_kwargs) -> processor default sampling "
            "int(clip/24*2) frames (9/14/19/24/29 for t=1..5). This "
            "matches the behavior audit and the rescue baseline; the "
            "hidden-state PROBE data used the nominal 8 fps sampling "
            "(31/47/63/79/95) instead - see temporal_intervention_"
            "manifest.py. Do not 'fix' the kwarg without regenerating "
            "the audit data."),
        "files": ["generation_trace.csv", "generation_hidden.npz",
                  "generation_hidden_shape.json"],
    }
    with open(out_dir / "generation_trace_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'generation_trace_summary.json'}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["selftest", "run"], default="selftest")
    ap.add_argument("--model-dir", default=None,
                    help="HF model dir (required for --mode run)")
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                    help="comma-separated hidden-state layer indices "
                         "(0 = embedding layer; final layer of Qwen3-VL-8B "
                         "is 36)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="trace only the first N samples - for smoke tests")
    args = ap.parse_args()

    if args.mode == "selftest":
        ok = selftest_causality(topk=args.topk)
        raise SystemExit(0 if ok else 1)
    if args.mode == "run":
        if not args.model_dir:
            ap.error("--mode run requires --model-dir")
        run_model(args)


if __name__ == "__main__":
    main()
