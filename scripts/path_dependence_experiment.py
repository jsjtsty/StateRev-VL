#!/usr/bin/env python3
"""Held-out path-dependence / non-Markov state experiment.

The frozen pair is matched on t, initial state, S_prev, E_t, and S_t, while
target and donor have different earlier event histories. Four videos separate
history from the current-event window:

  target_real                    target history + target current window
  donor_history_target_current   donor history + target current window
  target_history_donor_current   target history + donor current window
  donor_real                      donor history + donor current window

Only the explicit ``run`` subcommand loads Qwen3-VL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAYER_SET = (24, 32, 36)
REGIME = "controlled_8fps"
SEED = 20260913
CONDITIONS = ("target_real", "donor_history_target_current",
              "target_history_donor_current", "donor_real")
STATES = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
HIDDEN_DECODER_LAYERS = (24, 32, 36)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_history_signature(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["trajectory_id", "t"]).copy()
    signatures = {}
    for trajectory, group in data.groupby("trajectory_id"):
        history = []
        for index, row in group.sort_values("t").iterrows():
            signatures[index] = ">".join(history)
            history.append(row["gt_event"])
    data["history_signature"] = [signatures[index] for index in data.index]
    return data


def build_pairs(behavior: pd.DataFrame, split: dict, min_t: int) -> list[dict]:
    behavior = add_history_signature(behavior)
    discovery = set(split["discovery_trajectories"])
    validation = set(split["validation_trajectories"])
    targets = behavior[(behavior.trajectory_id.isin(validation)) & (behavior.t >= min_t)]
    donors = behavior[behavior.trajectory_id.isin(discovery)]
    pairs = []
    for _, target in targets.sort_values("target_prefix").iterrows():
        candidates = donors[
            (donors.t == target.t)
            & (donors.initial_state == target.initial_state)
            & (donors.gt_prev_state == target.gt_prev_state)
            & (donors.gt_event == target.gt_event)
            & (donors.gt_state == target.gt_state)
            & (donors.history_signature != target.history_signature)
        ].sort_values("target_prefix")
        if candidates.empty:
            continue
        donor = candidates.iloc[0]
        pairs.append({
            "pair_id": f"{target.target_prefix}_from_{donor.trajectory_id}",
            "target_prefix": target.target_prefix,
            "target_trajectory": target.trajectory_id,
            "donor_prefix": donor.target_prefix,
            "donor_trajectory": donor.trajectory_id,
            "t": int(target.t),
            "frame_end": int(target.frame_end),
            "donor_frame_end": int(donor.frame_end),
            "initial_state": target.initial_state,
            "prev_state": target.gt_prev_state,
            "event": target.gt_event,
            "gt_state": target.gt_state,
            "target_history": target.history_signature,
            "donor_history": donor.history_signature,
            "target_state_correct": bool(target.state_correct),
            "donor_state_correct": bool(donor.state_correct),
            "target_event_correct": bool(target.event_correct),
            "donor_event_correct": bool(donor.event_correct),
        })
    return pairs


def manifest(args: argparse.Namespace) -> dict:
    behavior = pd.read_csv(args.behavior)
    behavior["target_prefix"] = behavior.trajectory_id + "_t" + behavior.t.astype(str)
    split = json.loads(args.split.read_text())
    pairs = build_pairs(behavior, split, args.min_t)
    if not args.baseline_hidden.exists():
        raise FileNotFoundError(f"missing baseline hidden cache: {args.baseline_hidden}")
    return {
        "version": 1,
        "seed": SEED,
        "regime": REGIME,
        "dataset": str(args.dataset),
        "behavior": str(args.behavior),
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "behavior_sha256": sha256(args.behavior),
        "baseline_hidden": str(args.baseline_hidden),
        "baseline_hidden_sha256": sha256(args.baseline_hidden),
        "frozen_pair_rule": "validation target and discovery donor; same t, initial state, S_prev, E_t, S_t; different earlier history signature; lexicographically first donor",
        "min_t": int(args.min_t),
        "conditions": list(CONDITIONS),
        "history_swap": {
            "target_real": "target history + target current sampled window",
            "donor_history_target_current": "donor history + target current sampled window",
            "target_history_donor_current": "target history + donor current sampled window",
            "donor_real": "donor history + donor current sampled window",
        },
        "decoder": {
            "fit": "discovery-trajectory baseline hidden states only",
            "layers_one_based": list(HIDDEN_DECODER_LAYERS),
            "state_and_event": True,
            "scaler_pca_regularization_frozen_on_discovery": True,
        },
        "statistics": "target-prefix aggregation then target-trajectory cluster bootstrap/sign permutation; pair rows descriptive",
        "independence_caveat": "All 50 trajectories were used in prior project stages; validation targets and discovery donors are disjoint but this is not a new-dataset replication.",
        "counts": {
            "pairs": len(pairs),
            "target_prefixes": len({p["target_prefix"] for p in pairs}),
            "target_trajectories": len({p["target_trajectory"] for p in pairs}),
            "by_t": {str(int(t)): int(n) for t, n in pd.DataFrame(pairs).groupby("t").size().items()} if pairs else {},
            "conditions": len(CONDITIONS),
            # Each rendered condition is evaluated once for logits and once
            # for base-model hidden states; these are separate forwards.
            "rendered_conditions": len(pairs) * len(CONDITIONS),
            "model_forwards": len(pairs) * len(CONDITIONS) * 2,
        },
        "pairs": pairs,
    }


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    if split.get("fingerprint_source"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from content_disjoint_split import assert_content_disjoint
        assert_content_disjoint(split, Path(split["fingerprint_source"]))
    m = manifest(args)
    (root / "path_dependence_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps({"status": "PATH_MANIFEST_READY", "counts": m["counts"], "min_t": args.min_t, "conditions": CONDITIONS}, indent=2))


def audit(args: argparse.Namespace) -> None:
    root = Path(args.out); m = json.loads((root / "path_dependence_manifest.json").read_text())
    assert sha256(Path(m["split"])) == m["split_sha256"]
    assert sha256(Path(m["behavior"])) == m["behavior_sha256"]
    assert sha256(Path(m["baseline_hidden"])) == m["baseline_hidden_sha256"]
    assert tuple(m["conditions"]) == CONDITIONS
    target = {p["target_trajectory"] for p in m["pairs"]}
    split = json.loads(Path(m["split"]).read_text())
    behavior = pd.read_csv(Path(m["behavior"]))
    behavior["target_prefix"] = behavior.trajectory_id + "_t" + behavior.t.astype(str)
    discovery_keys = set(behavior[behavior.trajectory_id.isin(split["discovery_trajectories"])]
                         .target_prefix)
    cache = np.load(Path(m["baseline_hidden"]), allow_pickle=False)
    assert discovery_keys <= set(cache.files), "baseline hidden cache lacks discovery keys"
    assert target <= set(split["validation_trajectories"])
    for p in m["pairs"]:
        assert p["donor_trajectory"] in set(split["discovery_trajectories"])
        assert p["target_trajectory"] != p["donor_trajectory"]
        assert p["t"] >= m["min_t"]
        assert p["target_history"] != p["donor_history"]
        assert p["frame_end"] == p["donor_frame_end"]
        assert Path(args.dataset, f"{p['target_trajectory']}.mp4").exists()
        assert Path(args.dataset, f"{p['donor_trajectory']}.mp4").exists()
    print(json.dumps({"status": "PATH_AUDIT_PASS", "counts": m["counts"], "history_pairs": len(m["pairs"])}, indent=2))


def unit() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mg_common as mg
    clip = np.arange(120 * 2 * 2 * 3, dtype=np.uint8).reshape(120, 2, 2, 3)
    source = np.flip(clip, axis=0).copy()
    n = mg.expected_frames(REGIME, len(clip))
    idx = mg.sampled_indices(len(clip), n)
    current = mg.window_sample_mask(len(clip), 1, n)
    history = ~current
    out = clip.copy(); out[idx[history]] = source[idx[history]]
    assert np.array_equal(out[idx[current]], clip[idx[current]])
    assert np.array_equal(out[idx[history]], source[idx[history]])
    out_hash = sampled_frame_hashes(out)
    source_hash = sampled_frame_hashes(source)
    clip_hash = sampled_frame_hashes(clip)
    assert all(out_hash[i] == (clip_hash[i] if current[i] else source_hash[i])
               for i in range(n))
    # Exercise the actual four-condition construction, including the current
    # window transplant helper used by the GPU runner.
    import mg_common as mg
    clips = {
        "target_real": clip,
        "donor_history_target_current": out,
        "target_history_donor_current": mg.transplant_sampled_window(clip, source, 1, REGIME)[0],
        "donor_real": source,
    }
    expected = {
        "target_real": clip_hash,
        "donor_history_target_current": [clip_hash[i] if current[i] else source_hash[i] for i in range(n)],
        "target_history_donor_current": [source_hash[i] if current[i] else clip_hash[i] for i in range(n)],
        "donor_real": source_hash,
    }
    for name, value in clips.items():
        assert sampled_frame_hashes(value) == expected[name], name
    assert all(len(sampled_frame_hashes(v)) == n for v in clips.values())
    print("PATH_UNIT_PASS: all four conditions preserve sampled slots and swap only intended window")


def fit_decoders(args, model_hidden: dict[str, np.ndarray], behavior: pd.DataFrame, split: dict):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    discovery = set(split["discovery_trajectories"])
    rows = behavior[behavior.trajectory_id.isin(discovery)].sort_values("target_prefix")
    keys = rows.target_prefix.tolist()
    missing = [key for key in keys if key not in model_hidden]
    if missing:
        raise KeyError(f"baseline hidden cache missing {len(missing)} discovery keys; first={missing[:3]}")
    decoders = {}
    for layer in HIDDEN_DECODER_LAYERS:
        x = np.vstack([model_hidden[key][layer] for key in keys])
        scaler = StandardScaler().fit(x); xs = scaler.transform(x)
        pca = PCA(n_components=min(80, xs.shape[0] - 1), random_state=0).fit(xs)
        z = pca.transform(xs)
        # scikit-learn 1.8+ removed the deprecated ``multi_class`` keyword;
        # lbfgs already selects the required multinomial/multiclass path.
        state = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(z, rows.gt_state)
        event = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(z, rows.gt_event)
        decoders[layer] = (scaler, pca, state, event)
    return decoders


def apply_history_swap(target_clip: np.ndarray, source_clip: np.ndarray, t: int) -> np.ndarray:
    import mg_common as mg
    if len(target_clip) != len(source_clip):
        raise ValueError("target/source clip lengths differ")
    n = mg.expected_frames(REGIME, len(target_clip))
    idx = mg.sampled_indices(len(target_clip), n)
    current = mg.window_sample_mask(len(target_clip), t, n)
    out = target_clip.copy()
    out[idx[~current]] = source_clip[idx[~current]]
    return out


def sampled_frame_hashes(clip: np.ndarray) -> list[str]:
    """Hashes of the actual processor-sampled frame slots, for audit only."""
    import mg_common as mg
    n = mg.expected_frames(REGIME, len(clip))
    idx = mg.sampled_indices(len(clip), n)
    return [hashlib.sha256(np.ascontiguousarray(clip[i]).tobytes()).hexdigest()
            for i in idx]


def run(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import circuit_localization
    import mg_common as mg
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS, extract_hidden_states, load_model_and_processor, render_inputs, to_device

    root = Path(args.out); m = json.loads((root / "path_dependence_manifest.json").read_text())
    behavior = pd.read_csv(args.behavior); behavior["target_prefix"] = behavior.trajectory_id + "_t" + behavior.t.astype(str)
    split = json.loads(args.split.read_text())
    base_npz = np.load(args.baseline_hidden, allow_pickle=False)
    baseline_hidden = {key: base_npz[key] for key in base_npz.files}
    decoders = fit_decoders(args, baseline_hidden, behavior, split)
    model, processor = load_model_and_processor(Path(args.model_dir)); model.eval()
    device = next(model.parameters()).device
    pairs = m["pairs"][args.shard_index::args.num_shards]
    hidden = {condition: {} for condition in CONDITIONS}
    rows = []

    def render(clip, initial, t):
        inp, fp = render_inputs(processor, state_messages(clip, initial, t), clip, REGIME)
        return to_device(inp, device), fp

    def logits(inp):
        with torch.inference_mode(): out = model(**inp, logits_to_keep=1)
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return {state: float(lp[token]) for state, token in POS_IDS.items()}

    def decode_hidden(hidden_arr, layer, p):
        scaler, pca, state_dec, event_dec = decoders[layer]
        z = pca.transform(scaler.transform(hidden_arr.reshape(1, -1)))
        sp = state_dec.predict_proba(z)[0]; ep = event_dec.predict_proba(z)[0]
        state_prob = {str(c): float(v) for c, v in zip(state_dec.classes_, sp)}
        event_prob = {str(c): float(v) for c, v in zip(event_dec.classes_, ep)}
        return state_prob, event_prob

    for p in tqdm(pairs, desc=f"path shard {args.shard_index}", unit="pair", dynamic_ncols=True):
        tc = sample_clip(Path(args.dataset) / f"{p['target_trajectory']}.mp4", 0, p["frame_end"])
        dc = sample_clip(Path(args.dataset) / f"{p['donor_trajectory']}.mp4", 0, p["donor_frame_end"])
        assert len(tc) == len(dc) == p["frame_end"] == p["donor_frame_end"]
        clips = {
            "target_real": tc,
            "donor_history_target_current": apply_history_swap(tc, dc, p["t"]),
            "target_history_donor_current": mg.transplant_sampled_window(tc, dc, p["t"], REGIME)[0],
            "donor_real": dc,
        }
        # Verify the factorial construction before any model call.  These
        # assertions make the intended history/current-window substitution
        # auditable at sampled-slot level, rather than inferring it from the
        # rendered tensor hash alone.
        n = mg.expected_frames(REGIME, len(tc))
        idx = mg.sampled_indices(len(tc), n)
        current = mg.window_sample_mask(len(tc), p["t"], n)
        th, dh = sampled_frame_hashes(tc), sampled_frame_hashes(dc)
        expected_slots = {
            "target_real": th,
            "donor_history_target_current": [th[i] if current[i] else dh[i] for i in range(n)],
            "target_history_donor_current": [dh[i] if current[i] else th[i] for i in range(n)],
            "donor_real": dh,
        }
        for condition, clip in clips.items():
            assert sampled_frame_hashes(clip) == expected_slots[condition], condition
        outputs = {}
        for condition, clip in clips.items():
            inp, fp = render(clip, p["initial_state"], p["t"])
            output_logits = logits(inp)
            hs = extract_hidden_states(model, inp, layers=HIDDEN_DECODER_LAYERS)
            hidden[condition][p["pair_id"]] = hs
            outputs[condition] = (output_logits, fp, hs)
        target = outputs["target_real"][0]; donor = outputs["donor_real"][0]
        gt = p["gt_state"]
        for condition in CONDITIONS:
            z, fp, hs = outputs[condition]
            pred = max(STATES, key=z.get)
            gt_margin = z[gt] - max(z[state] for state in STATES if state != gt)
            target_margin = target[gt] - max(target[state] for state in STATES if state != gt)
            state_probs = {}; event_probs = {}
            for i, layer in enumerate(HIDDEN_DECODER_LAYERS):
                state_probs[layer], event_probs[layer] = decode_hidden(hs[i], layer, p)
            record = {
                "pair_id": p["pair_id"], "target_prefix": p["target_prefix"],
                "target_trajectory": p["target_trajectory"], "donor_trajectory": p["donor_trajectory"],
                "t": p["t"], "condition": condition, "gt_state": gt,
                "event": p["event"], "target_history": p["target_history"], "donor_history": p["donor_history"],
                "baseline_target_pred": max(STATES, key=target.get), "prediction": pred,
                "baseline_target_gt_margin": target_margin, "gt_margin": gt_margin,
                "delta_gt_margin_vs_target_real": gt_margin - target_margin,
                "baseline_target_correct": max(STATES, key=target.get) == gt,
                "condition_correct": pred == gt,
                "wrong_to_correct_vs_target_real": max(STATES, key=target.get) != gt and pred == gt,
                "correct_to_wrong_vs_target_real": max(STATES, key=target.get) == gt and pred != gt,
                "delta_logit_Left": z["Left"] - target["Left"],
                "delta_logit_Middle": z["Middle"] - target["Middle"],
                "delta_logit_Right": z["Right"] - target["Right"],
                "condition_input_ids_sha256": fp["input_ids_sha256"],
                "condition_pixel_sha256": fp["pixel_values_videos_sha256"],
                "condition_pixel_shape": json.dumps(fp["pixel_values_videos_shape"]),
                "condition_video_grid_thw": json.dumps(fp["video_grid_thw"]),
                "condition_prompt_text_sha256": fp["prompt_text_sha256"],
                "fingerprint_model": fp["model"],
                "fingerprint_backend": fp["backend"],
                "fingerprint_transformers_version": fp["transformers_version"],
                "fingerprint_torch_version": fp["torch_version"],
                "sampled_frame_indices": json.dumps(idx.tolist()),
                "sampled_frame_slot_sha256": json.dumps(sampled_frame_hashes(clips[condition])),
                "state_probe_L24_gt": state_probs[24].get(gt),
                "state_probe_L32_gt": state_probs[32].get(gt),
                "state_probe_L36_gt": state_probs[36].get(gt),
                "event_probe_L24_current": event_probs[24].get(p["event"]),
                "event_probe_L32_current": event_probs[32].get(p["event"]),
                "event_probe_L36_current": event_probs[36].get(p["event"]),
                "hidden_L24_norm": float(np.linalg.norm(hs[0])),
                "hidden_L32_norm": float(np.linalg.norm(hs[1])),
                "hidden_L36_norm": float(np.linalg.norm(hs[2])),
            }
            # Input invariants are checked across the four rendered versions.
            if condition != "target_real":
                ref = outputs["target_real"][1]
                assert fp["input_ids_sha256"] == ref["input_ids_sha256"]
                assert fp["prompt_text_sha256"] == ref["prompt_text_sha256"]
                assert fp["video_grid_thw"] == ref["video_grid_thw"]
                assert fp["pixel_values_videos_shape"] == ref["pixel_values_videos_shape"]
            rows.append(record)
    out = root / "shards"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"path_shard_{args.shard_index}.csv", index=False)
    for condition in CONDITIONS:
        np.savez_compressed(out / f"hidden_{condition}_shard_{args.shard_index}.npz", **hidden[condition])
    (out / f"path_shard_{args.shard_index}.meta.json").write_text(json.dumps({"pairs": len(pairs), "rows": len(rows), "conditions": CONDITIONS}, indent=2) + "\n")
    print(json.dumps({"status": "PATH_SHARD_COMPLETE", "shard": args.shard_index, "pairs": len(pairs), "rows": len(rows)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=Path("outputs/vetbench/path_dependence_v1"))
    common.add_argument("--behavior", type=Path, default=Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"))
    common.add_argument("--split", type=Path, default=Path("outputs/vetbench/circuit_localization_v2/discovery_validation_split.json"))
    common.add_argument("--dataset", type=Path, default=Path("dataset/vetbench/cup"))
    common.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    common.add_argument("--baseline-hidden", type=Path, default=Path("outputs/vetbench/mechanism_gate_final/hidden_baseline.npz"))
    common.add_argument("--min-t", type=int, default=3)
    sub.add_parser("prepare", parents=[common]); sub.add_parser("audit", parents=[common]); sub.add_parser("unit", parents=[common])
    run_parser = sub.add_parser("run", parents=[common]); run_parser.add_argument("--shard-index", type=int, required=True); run_parser.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "audit": audit(args)
    elif args.command == "unit": unit()
    else: run(args)


if __name__ == "__main__": main()
