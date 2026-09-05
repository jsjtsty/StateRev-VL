#!/usr/bin/env python3
"""Final, preregistered mechanism experiment for VET-Bench cup/shell.

Modes: manifest (CPU), dry-run (CPU), run (GPU), analyze (CPU).
The runner is intentionally conservative: it never changes prompt text or
clip length.  It edits only exact sampled raw-frame slots, then calls the
unified ``render_inputs`` entry point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mg_common as mg
from state_rev_input_pipeline import (POS_IDS, extract_hidden_states,
    first_token_logits, load_model_and_processor, render_inputs)
from run_state_rev_audit import state_messages
from run_vetbench_screening import sample_clip

ROOT = Path("outputs/vetbench/mechanism_gate_final")
BEHAVIOR = Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv")
DATASET = Path("dataset/vetbench/cup")
REGIME = "controlled_8fps"
CONDITIONS = ("baseline", "no_current_event", "freeze", "shuffle",
              "matched_history_control")
PAIR_CONDITIONS = ("target_self", "same_event_source",
                   "matched_history_transplant", "source_window_shuffle",
                   "source_current_transplant")


def truth(x): return str(x).strip().lower() == "true"


def apply_event(prev, event):
    a, b = event.split(" and ")
    return b if prev == a else (a if prev == b else prev)


def load_rows(path=BEHAVIOR):
    rows = list(csv.DictReader(path.open(newline="")))
    assert len(rows) == 250, len(rows)
    for r in rows: r["t"] = int(r["t"])
    return rows


def build_manifest(rows):
    by_t = defaultdict(list)
    for r in rows:
        if truth(r["is_transition"]): by_t[r["t"]].append(r)
    pairs = []
    for t in sorted(by_t):
        for target in by_t[t]:
            candidates = []
            same = []
            for source in by_t[t]:
                if source["trajectory_id"] == target["trajectory_id"]:
                    continue
                # Required: same S_{t-1}; otherwise the transplant changes
                # both the event and the latent starting state.
                if source["gt_prev_state"] != target["gt_prev_state"]:
                    continue
                if source["gt_event"] == target["gt_event"]:
                    same.append(source)
                    continue
                cf = apply_event(target["gt_prev_state"], source["gt_event"])
                if cf != target["gt_state"]:
                    candidates.append((source, cf))
            same.sort(key=lambda r: r["trajectory_id"])
            # Every main eligible source is retained. Same-event is a fixed,
            # deterministic control source for the same target.
            if not same: continue
            for source, cf in sorted(candidates, key=lambda x: x[0]["trajectory_id"]):
                pairs.append({
                    "pair_id": f"{target['trajectory_id']}_t{t}_from_{source['trajectory_id']}",
                    "t": t, "target_traj": target["trajectory_id"],
                    "source_traj": source["trajectory_id"],
                    "same_event_traj": same[0]["trajectory_id"],
                    "prev_state": target["gt_prev_state"],
                    "target_event": target["gt_event"],
                    "source_event": source["gt_event"],
                    "target_state": target["gt_state"],
                    "counterfactual_state": cf,
                    "frame_end": int(target["frame_end"]),
                    "target_initial": target["initial_state"],
                    "source_initial": source["initial_state"],
                    "same_event_initial": same[0]["initial_state"],
                })
    return pairs


def write_manifest(args):
    rows = load_rows(args.behavior)
    pairs = build_manifest(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "regime": REGIME, "selection":
        "all eligible pairs: same t, same prev state, different event, cf != target",
        "n_rows": len(rows), "n_pairs": len(pairs), "pairs": pairs}
    (args.out / "pair_manifest.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({"n_rows": len(rows), "n_pairs": len(pairs),
                      "by_t": dict(Counter(p["t"] for p in pairs)),
                      "manifest": str(args.out / "pair_manifest.json")}, indent=2))


def sampled_frame_hashes(clip, t):
    n = mg.expected_frames(REGIME, len(clip))
    idx = mg.sampled_indices(len(clip), n)
    return idx.tolist(), [hashlib.sha256(np.ascontiguousarray(clip[i]).tobytes()).hexdigest()
                          for i in idx]


def assert_render_equal(base_fp, fp):
    for k in ("input_ids_len", "input_ids_sha256", "video_grid_thw",
              "pixel_values_videos_shape", "prompt_text_sha256", "model",
              "backend", "transformers_version", "torch_version"):
        assert base_fp[k] == fp[k], f"invariant changed: {k}"
def run(args):
    rows = load_rows(args.behavior)
    manifest_path = args.manifest or (args.out / "pair_manifest.json")
    manifest = json.loads(Path(manifest_path).read_text())
    pairs = manifest["pairs"]
    if args.limit: rows = rows[:args.limit]
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")
    rows = rows[args.shard_index::args.num_shards]
    pairs = pairs[args.shard_index::args.num_shards]
    model, processor = load_model_and_processor(args.model_dir)
    hidden, behavior, fingerprints = defaultdict(dict), [], []
    pair_by_target = {p["pair_id"]: p for p in pairs}
    args.out.mkdir(parents=True, exist_ok=True)

    def save(key, condition, clip, initial_state, t, base_fp=None):
        # Messages contain the ndarray itself; rebuilding them here prevents a
        # manipulated clip from being accidentally rendered with baseline pixels.
        msgs = state_messages(clip, initial_state, t)
        inputs, fp = render_inputs(processor, msgs, clip, REGIME)
        if base_fp is not None: assert_render_equal(base_fp, fp)
        hidden[condition][key] = extract_hidden_states(model, inputs)
        b = first_token_logits(model, processor, inputs, POS_IDS)
        fingerprints.append({"key": key, "condition": condition,
            "pixel_values_changed_vs_base": (None if base_fp is None else
                base_fp["pixel_values_videos_sha256"] != fp["pixel_values_videos_sha256"]),
            "sampled_frame_indices": sampled_frame_hashes(clip, 0)[0],
            "sampled_frame_sha256": sampled_frame_hashes(clip, 0)[1], **fp})
        behavior.append({"key": key, "condition": condition,
            "state_pred": b["answer"], "logprob_Left": b["first_logprobs"]["Left"],
            "logprob_Middle": b["first_logprobs"]["Middle"],
            "logprob_Right": b["first_logprobs"]["Right"]})
        return fp

    # Main no-event factorial: baseline and controls for every prefix.
    for r in tqdm(rows, desc=f"prefix shard {args.shard_index}", unit="prefix"):
        clip = sample_clip(args.dataset / f"{r['trajectory_id']}.mp4", 0, int(r["frame_end"]))
        base_clip, _ = mg.build_manipulated_clip(clip, r["t"], REGIME, "baseline")
        base_fp = save(f"{r['trajectory_id']}_t{r['t']}", "baseline", base_clip,
                       r["initial_state"], r["t"])
        for c in CONDITIONS[1:]:
            cc = "history_freeze" if c == "matched_history_control" else c
            altered, _ = mg.build_manipulated_clip(clip, r["t"], REGIME, cc)
            save(f"{r['trajectory_id']}_t{r['t']}", c, altered,
                 r["initial_state"], r["t"], base_fp)

    # Pair interventions keep the TARGET prompt/history and transplant only
    # source current-window sampled frames (or a matched history destination).
    selected_trajs = {r["trajectory_id"] for r in rows}
    for p in tqdm(pairs, desc=f"pair shard {args.shard_index}", unit="pair"):
        if args.limit and p["target_traj"] not in selected_trajs: continue
        tc = sample_clip(args.dataset / f"{p['target_traj']}.mp4", 0, p["frame_end"])
        sc = sample_clip(args.dataset / f"{p['source_traj']}.mp4", 0, p["frame_end"])
        ec = sample_clip(args.dataset / f"{p['same_event_traj']}.mp4", 0, p["frame_end"])
        msgs = state_messages(tc, p["target_initial"], p["t"])
        base_inputs, base_fp = render_inputs(processor, msgs, tc, REGIME)
        del base_inputs
        variants = {
            "target_self": (tc, None),
            "same_event_source": mg.transplant_sampled_window(tc, ec, p["t"], REGIME),
            "matched_history_transplant": mg.transplant_sampled_window(tc, sc, p["t"], REGIME, destination="history"),
            "source_window_shuffle": mg.transplant_sampled_window(tc, sc, p["t"], REGIME, temporal_shuffle=True),
            "source_current_transplant": mg.transplant_sampled_window(tc, sc, p["t"], REGIME),
        }
        for c, val in variants.items():
            vc = tc if val[1] is None else val[0]
            save(p["pair_id"], c, vc, p["target_initial"], p["t"],
                 None if c == "target_self" else base_fp)
    for c, vals in hidden.items(): np.savez_compressed(args.out / f"hidden_{c}.npz", **vals)
    (args.out / "behavior.csv").write_text("")
    with (args.out / "behavior.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=behavior[0].keys()); w.writeheader(); w.writerows(behavior)
    with (args.out / "fingerprints.jsonl").open("w") as f:
        for x in fingerprints: f.write(json.dumps(x) + "\n")
    (args.out / "run_manifest.json").write_text(json.dumps({"regime": REGIME,
        "n_prefix_rows": len(rows), "n_pairs": len(pairs),
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "conditions": CONDITIONS + PAIR_CONDITIONS}, indent=1))
    print(f"wrote {len(behavior)} forwards to {args.out}")


def merge(args):
    """Merge disjoint shard outputs; no model loading or inference."""
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    conditions = CONDITIONS + PAIR_CONDITIONS
    merged = {}
    for condition in conditions:
        combined = {}
        for i in range(args.num_shards):
            path = root / "shards" / f"shard_{i}" / f"hidden_{condition}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as z:
                overlap = set(combined).intersection(z.files)
                if overlap:
                    raise RuntimeError(f"duplicate {condition} keys: {sorted(overlap)[:3]}")
                combined.update({k: z[k] for k in z.files})
        np.savez_compressed(root / f"hidden_{condition}.npz", **combined)
        merged[condition] = len(combined)

    behavior_rows = []
    fingerprints = []
    for i in range(args.num_shards):
        shard = root / "shards" / f"shard_{i}"
        with (shard / "behavior.csv").open(newline="") as f:
            behavior_rows.extend(csv.DictReader(f))
        with (shard / "fingerprints.jsonl").open() as f:
            fingerprints.extend(json.loads(line) for line in f if line.strip())
    if behavior_rows:
        fields = list(behavior_rows[0].keys())
        with (root / "behavior.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(behavior_rows)
    with (root / "fingerprints.jsonl").open("w") as f:
        for item in fingerprints: f.write(json.dumps(item) + "\n")
    (root / "run_manifest.json").write_text(json.dumps({
        "regime": REGIME, "num_shards": args.num_shards,
        "n_prefix_rows": len({x["key"] for x in behavior_rows if "_from_" not in x["key"]}),
        "n_pairs": len({x["key"] for x in behavior_rows if "_from_" in x["key"]}),
        "merged_hidden_keys": merged}, indent=1))
    print(json.dumps({"merged": merged, "behavior_rows": len(behavior_rows),
                      "fingerprints": len(fingerprints)}, indent=2))


def dry_run(args):
    rows = load_rows(args.behavior); pairs = build_manifest(rows)
    print(json.dumps({"prefixes": len(rows), "pairs": len(pairs),
        "prefix_conditions": CONDITIONS, "pair_conditions": PAIR_CONDITIONS,
        "regime": REGIME, "full_run_forwards": len(rows)*len(CONDITIONS) + len(pairs)*len(PAIR_CONDITIONS),
        "model_load": False}, indent=2))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("manifest", "dry-run", "run", "merge"):
        p = sub.add_parser(mode); p.add_argument("--behavior", type=Path, default=BEHAVIOR)
        p.add_argument("--out", type=Path, default=ROOT)
        if mode == "run":
            p.add_argument("--dataset", type=Path, default=DATASET); p.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-VL-8B-Instruct")); p.add_argument("--limit", type=int, default=0)
            p.add_argument("--manifest", type=Path, default=None)
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--num-shards", type=int, default=1)
        elif mode == "merge":
            p.add_argument("--num-shards", type=int, required=True)
        p.set_defaults(func=write_manifest if mode == "manifest" else dry_run if mode == "dry-run" else run if mode == "run" else merge)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__": main()
