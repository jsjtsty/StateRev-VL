"""Native logit sanity check for StateRev-VL (GPU 3; no visual re-forward).

Reads the P0-validated probe-position hidden states from
outputs/vetbench/hidden_state_probe/hidden_states.npz and, for every row and
every layer, applies the model's final RMSNorm + the original lm_head, then
reads the logits of the first tokens of "Left" / "Middle" / "Right" (plus the
full-vocabulary argmax at the final layer).

Important quirk (verified empirically, see scripts/debug_hs_tuple.py): in
this transformers build the hidden-states tuple's last element is the
POST-final-norm state (bit-equal to last_hidden_state), while indices 0..35
are pre-norm layer outputs. The final norm is therefore applied only to
layers 0..35; for layer 36 the lm_head is applied directly, which reproduces
the model's exact first-token logit distribution.

This checks whether the model's own first-token distribution at the probe
position agrees with the behavioral (greedy) state prediction. If the final
layer's native argmax disagrees with the behavioral answer at scale, the
extraction/generation alignment is suspect (do not interpret mid-layer
mechanism before resolving that).

Outputs (outputs/vetbench/probe_analysis_v2/):
  native_layer_logits.csv      250 rows x 37 layers: logit_Left/Middle/Right
  native_first_token.csv       per row: full-vocab argmax, {L,M,R} argmax,
                               behavioral state_pred, match flags
  native_canonical_margin.csv  51 canonical rows x 37 layers:
                               logit(GT S_t) - logit(GT S_{t-1})
  native_summary.json

Run from the project root (physical GPU 3 only):
  CUDA_VISIBLE_DEVICES=3 python scripts/native_logit_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from run_hidden_state_probe import MANIFEST, OUT_DIR, STATE_CLASSES, read_manifest
from run_state_rev_audit import load_transformers_vl_model

V2_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "probe_analysis_v2"
STATE_WORDS = ("Left", "Middle", "Right")


def stale_row(r: dict) -> bool:
    return bool(r["clean_revision"]) and not r["state_correct"] and r["state_pred"] == r["gt_prev_state"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-rows", type=int, default=25)
    ap.add_argument("--out-dir", type=Path, default=V2_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, processor = load_transformers_vl_model(args.model_dir)
    device = torch.device(args.device)
    tok = processor.tokenizer

    # ---- tokenizer sanity ---------------------------------------------------
    word_ids: dict[str, int] = {}
    word_ids_spaced: dict[str, int] = {}
    token_report = {}
    for w in STATE_WORDS:
        i1 = tok.encode(w, add_special_tokens=False)
        i2 = tok.encode(" " + w, add_special_tokens=False)
        assert len(i1) == 1, f"{w!r} is not a single token: {i1}"
        word_ids[w] = int(i1[0])
        word_ids_spaced[w] = int(i2[0]) if len(i2) == 1 else None
        token_report[w] = {"bare_id": word_ids[w], "spaced_id": word_ids_spaced[w]}
    print("Token ids:", token_report)

    # ---- head + final norm --------------------------------------------------
    W = model.lm_head.weight  # (vocab, 4096) bf16 (tied to embed_tokens)
    text_model = model.model.language_model
    norm_w = text_model.norm.weight.float()
    eps = float(model.config.text_config.rms_norm_eps)
    id3 = [word_ids[w] for w in STATE_WORDS]
    W3 = W[id3].float()  # (3, 4096)
    W_full = W.float()   # (vocab, 4096)
    print(f"lm_head {tuple(W.shape)}; eps={eps}")

    rows = read_manifest(MANIFEST)
    with np.load(OUT_DIR / "hidden_states.npz") as z:
        X = np.stack([z[f"{r['trajectory_id']}_t{r['t']}"] for r in rows], axis=0)
    n, L1, D = X.shape
    assert n == 250 and L1 == 37, (n, L1)

    # logits for the 3 state words, all layers: (250, 37, 3)
    logits3 = np.zeros((n, L1, 3), dtype=np.float32)
    # full-vocab argmax at the final layer per row
    argmax_full_id = np.zeros(n, dtype=np.int64)
    argmax_full_logit = np.zeros(n, dtype=np.float32)

    # NOTE: in this transformers build the hidden-states tuple's LAST element
    # (index L1-1) is the POST-final-norm state (bit-equal to last_hidden_state,
    # verified in scripts/debug_hs_tuple.py); indices 0..L1-2 are pre-norm layer
    # outputs.  So the final norm is applied only to layers 0..L1-2; the stored
    # final-layer state already includes it (lm_head on it IS the model's real
    # first-token logit distribution).
    h_all = torch.from_numpy(X).to(device).float()  # (250, 37, 4096)
    norm_w_d = norm_w.to(device)
    W3_d = W3.to(device)
    W_full_d = W_full.to(device)
    with torch.inference_mode():
        for s in range(0, n, args.chunk_rows):
            h4 = h_all[s:s + args.chunk_rows]                       # (c, 37, 4096)
            hfn = h4 * torch.rsqrt(h4.pow(2).mean(-1, keepdim=True) + eps) * norm_w_d
            hfn[:, -1, :] = h4[:, -1, :]                            # already post-norm
            hf = hfn.reshape(-1, D)                                 # (c*37, 4096)
            l3 = (hf @ W3_d.T).view(h4.shape[0], L1, 3)
            logits3[s:s + args.chunk_rows] = l3.cpu().numpy()
            hfl = hf.view(h4.shape[0], L1, D)[:, -1, :]             # (c, 4096)
            lfull = hfl @ W_full_d.T                               # (c, vocab)
            argmax_full_id[s:s + args.chunk_rows] = lfull.argmax(-1).cpu().numpy()
            argmax_full_logit[s:s + args.chunk_rows] = lfull.max(-1).values.cpu().numpy()
    del h_all, W_full_d, W_full, W3_d

    # ---- per-row outputs ------------------------------------------------------
    layer_name = lambda l: "embedding" if l == 0 else f"layer_{l:02d}"
    with open(args.out_dir / "native_layer_logits.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "trajectory_id", "t", "layer", "layer_name",
                    "logit_Left", "logit_Middle", "logit_Right"])
        for i, r in enumerate(rows):
            for l in range(L1):
                w.writerow([f"{r['trajectory_id']}_t{r['t']}", r["trajectory_id"],
                            r["t"], l, layer_name(l),
                            f"{logits3[i, l, 0]:.6f}", f"{logits3[i, l, 1]:.6f}",
                            f"{logits3[i, l, 2]:.6f}"])

    idx = {w_: i for i, w_ in enumerate(STATE_WORDS)}
    first_rows = []
    for i, r in enumerate(rows):
        a3 = int(np.argmax(logits3[i, -1]))
        a3w = STATE_WORDS[a3]
        afid = int(argmax_full_id[i])
        af_text = tok.decode([afid], skip_special_tokens=False).strip()
        first_rows.append({
            "key": f"{r['trajectory_id']}_t{r['t']}",
            "trajectory_id": r["trajectory_id"], "t": r["t"],
            "gt_state": r["gt_state"], "state_pred": r["state_pred"],
            "event_correct": r["event_correct"], "clean_revision": r["clean_revision"],
            "is_stale": stale_row(r),
            "argmax3": a3w,
            "match3": bool(a3w == r["state_pred"]),
            "full_argmax_id": afid, "full_argmax_text": af_text,
            "full_argmax_logit": float(argmax_full_logit[i]),
            "match_full_exact": bool(tok.decode([afid], skip_special_tokens=False)
                                     == r["state_pred"]),
            "match_full_stripped": bool(af_text == r["state_pred"]),
        })
    with open(args.out_dir / "native_first_token.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(first_rows[0].keys()))
        w.writeheader()
        for r in first_rows:
            w.writerow(r)

    # ---- canonical margins ----------------------------------------------------
    can = [r for r in rows if r["clean_revision"]]
    can_idx = [rows.index(r) for r in can]
    margin_rows = []
    for r_can, i in zip(can, can_idx):
        cur = idx[r_can["gt_state"]]
        prev = idx[r_can["gt_prev_state"]]
        grp = ("stale" if stale_row(r_can)
               else "success" if r_can["state_correct"] else "other_failure")
        for l in range(L1):
            margin_rows.append({
                "key": f"{r_can['trajectory_id']}_t{r_can['t']}",
                "trajectory_id": r_can["trajectory_id"], "t": r_can["t"],
                "group": grp, "gt_prev": r_can["gt_prev_state"],
                "gt_cur": r_can["gt_state"], "state_pred": r_can["state_pred"],
                "layer": l, "layer_name": layer_name(l),
                "margin": float(logits3[i, l, cur] - logits3[i, l, prev]),
            })
    with open(args.out_dir / "native_canonical_margin.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys()))
        w.writeheader()
        for m in margin_rows:
            w.writerow(m)

    # ---- summary ---------------------------------------------------------------
    def match_rate(sel: list[dict], key: str) -> tuple[int, int]:
        tot = len(sel)
        hit = sum(r[key] for r in sel)
        return hit, tot

    fr = first_rows
    can_fr = [r for r in fr if r["clean_revision"]]
    stale_fr = [r for r in fr if r["is_stale"]]
    succ_fr = [r for r in fr if r["clean_revision"] and r["state_pred"] == r["gt_state"]]

    def margin_stats(group: str, layer: int | None = None):
        ms = [m["margin"] for m in margin_rows if m["group"] == group
              and (layer is None or m["layer"] == layer)]
        if not ms:
            return None
        ms = np.array(ms)
        return {"n": len(ms), "mean": float(ms.mean()), "median": float(np.median(ms)),
                "frac_pos": float((ms > 0).mean())}

    summary = {
        "config": {
            "model_dir": str(args.model_dir),
            "hidden_states": str(OUT_DIR / "hidden_states.npz"),
            "token_ids": token_report,
            "readout": "h_probe -> final RMSNorm (float32) -> lm_head; logits "
                       "restricted to the bare first tokens of Left/Middle/Right; "
                       "full-vocab argmax at the final layer",
            "no_visual_forward": True,
        },
        "final_layer_alignment": {
            "all_250": {
                "match3": list(match_rate(fr, "match3")),
                "match_full_exact": list(match_rate(fr, "match_full_exact")),
                "match_full_stripped": list(match_rate(fr, "match_full_stripped")),
            },
            "canonical_51": {
                "match3": list(match_rate(can_fr, "match3")),
                "match_full_stripped": list(match_rate(can_fr, "match_full_stripped")),
            },
            "stale_31": {
                "match3": list(match_rate(stale_fr, "match3")),
                "match_full_stripped": list(match_rate(stale_fr, "match_full_stripped")),
            },
            "success_15": {
                "match3": list(match_rate(succ_fr, "match3")),
                "match_full_stripped": list(match_rate(succ_fr, "match_full_stripped")),
            },
            "mismatch_rows_stripped": [
                {"key": r["key"], "gt": r["gt_state"], "pred": r["state_pred"],
                 "full_argmax_text": r["full_argmax_text"]}
                for r in fr if not r["match_full_stripped"]
            ],
        },
        "native_revision_margin": {
            "stale": {
                "final_layer": margin_stats("stale", L1 - 1),
                "layer_mean_over_37": margin_stats("stale"),
            },
            "success": {
                "final_layer": margin_stats("success", L1 - 1),
                "layer_mean_over_37": margin_stats("success"),
            },
            "other_failure": {
                "final_layer": margin_stats("other_failure", L1 - 1),
                "layer_mean_over_37": margin_stats("other_failure"),
            },
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "native_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    al = summary["final_layer_alignment"]
    print(f"Done in {summary['runtime_s']}s")
    print(f"final-layer alignment all: match3={al['all_250']['match3']}, "
          f"full_stripped={al['all_250']['match_full_stripped']}")
    print(f"canonical: match3={al['canonical_51']['match3']}  "
          f"stale: {al['stale_31']['match3']}  success: {al['success_15']['match3']}")
    print("native revision margin (final layer):")
    for g in ("stale", "success", "other_failure"):
        print(f"  {g}: {summary['native_revision_margin'][g]['final_layer']}")


if __name__ == "__main__":
    main()
