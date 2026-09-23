#!/usr/bin/env python3
"""Frozen L24 Head 0 causal-rescue experiment.

Only ``run`` loads Qwen3-VL. ``prepare``, ``audit``, and ``unit`` are offline.
The intervention is fixed at the final prompt token in the input of L24's
attention o_proj (the pre-o_proj Head 0 slice).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAYER = 24
HEAD = 0
SEED = 20260908
REGIME = "controlled_8fps"
STATES = ("Left", "Middle", "Right")
CONDITIONS = (
    "strong_matched_donor",
    "self_patch",
    "ordinary_matched_donor",
    "shuffled_matched_donor",
    "different_event_matched_history",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_table(metadata_path: Path, behavior_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path).copy()
    metadata["target_prefix"] = metadata["trajectory_id"] + "_t" + metadata["t"].astype(str)
    behavior = pd.read_csv(behavior_path)
    behavior = behavior[behavior["condition"] == "baseline"].set_index("key")
    records = []
    for row in metadata.itertuples(index=False):
        native = behavior.loc[row.target_prefix]
        logits = {state: float(native[f"logprob_{state}"]) for state in STATES}
        gt = row.gt_state
        others = [state for state in STATES if state != gt]
        records.append({
            **row._asdict(),
            "baseline_pred": max(STATES, key=logits.get),
            "baseline_gt_margin": logits[gt] - max(logits[state] for state in others),
            **{f"baseline_logit_{state}": logits[state] for state in STATES},
        })
    result = pd.DataFrame(records)
    if len(result) != 250 or result["target_prefix"].nunique() != 250:
        raise AssertionError("expected exactly 250 unique baseline prefixes")
    return result


def donor_record(row: pd.Series) -> dict:
    return {
        "trajectory_id": row["trajectory_id"],
        "target_prefix": row["target_prefix"],
        "t": int(row["t"]),
        "frame_end": int(row["frame_end"]),
        "initial_state": row["initial_state"],
        "prev_state": row["gt_prev_state"],
        "event": row["gt_event"],
        "state": row["gt_state"],
        "baseline_gt_margin": float(row["baseline_gt_margin"]),
    }


def build_manifest(args: argparse.Namespace) -> dict:
    split = json.loads(Path(args.split).read_text())
    if split.get("fingerprint_source"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from content_disjoint_split import assert_content_disjoint
        assert_content_disjoint(split, Path(split["fingerprint_source"]))
    table = baseline_table(Path(args.metadata), Path(args.behavior))
    target_ids = set(split["validation_trajectories"])
    donor_ids = set(split["discovery_trajectories"])
    if target_ids & donor_ids or len(target_ids | donor_ids) != 50:
        raise AssertionError("invalid fixed discovery/validation split")
    targets = table[table["trajectory_id"].isin(target_ids)].sort_values("target_prefix")
    donors = table[table["trajectory_id"].isin(donor_ids)].sort_values("target_prefix")
    rng = np.random.default_rng(args.seed)
    entries = []
    failed = []
    for _, target in targets.iterrows():
        same = donors[
            (donors["t"] == target["t"])
            & (donors["gt_prev_state"] == target["gt_prev_state"])
            & (donors["gt_event"] == target["gt_event"])
            & (donors["gt_state"] == target["gt_state"])
            & (donors["trajectory_id"] != target["trajectory_id"])
        ].sort_values("target_prefix")
        different = donors[
            (donors["t"] == target["t"])
            & (donors["gt_prev_state"] == target["gt_prev_state"])
            & (donors["gt_event"] != target["gt_event"])
            & (donors["gt_state"] != target["gt_state"])
            & (donors["trajectory_id"] != target["trajectory_id"])
        ].sort_values("target_prefix")
        if same.empty or different.empty:
            failed.append({"target_prefix": target["target_prefix"],
                           "same_match_count": len(same),
                           "different_match_count": len(different)})
            continue
        strong = same.sort_values(
            ["baseline_gt_margin", "target_prefix"], ascending=[False, True]
        ).iloc[0]
        ordinary = same.iloc[0]
        shuffled = same.iloc[int(rng.integers(0, len(same)))]
        different_strong = different.sort_values(
            ["baseline_gt_margin", "target_prefix"], ascending=[False, True]
        ).iloc[0]
        entries.append({
            "target_prefix": target["target_prefix"],
            "target_trajectory": target["trajectory_id"],
            "t": int(target["t"]),
            "frame_end": int(target["frame_end"]),
            "initial_state": target["initial_state"],
            "prev_state": target["gt_prev_state"],
            "current_event": target["gt_event"],
            "gt_state": target["gt_state"],
            "baseline_pred": target["baseline_pred"],
            "baseline_gt_margin": float(target["baseline_gt_margin"]),
            "baseline_logits": {state: float(target[f"baseline_logit_{state}"]) for state in STATES},
            "same_match_count": int(len(same)),
            "different_match_count": int(len(different)),
            "donors": {
                "strong_matched_donor": donor_record(strong),
                "self_patch": donor_record(target),
                "ordinary_matched_donor": donor_record(ordinary),
                "shuffled_matched_donor": donor_record(shuffled),
                "different_event_matched_history": donor_record(different_strong),
            },
        })
    margins = np.asarray([entry["baseline_gt_margin"] for entry in entries])
    threshold = float(np.median(margins))
    for entry in entries:
        entry["strength_group"] = "weak" if entry["baseline_gt_margin"] <= threshold else "strong"
        entry["baseline_incorrect"] = bool(entry["baseline_gt_margin"] < 0)
    model_config = Path(args.model_dir) / "config.json"
    manifest = {
        "version": 1,
        "seed": int(args.seed),
        "dataset": str(args.dataset),
        "regime": REGIME,
        "target_split": "original held-out validation trajectories",
        "donor_pool": "original discovery trajectories only",
        "independence_caveat": (
            "VET-Bench has only 50 trajectories and all were used by prior head discovery/validation. "
            "No untouched trajectory split exists. This rescue test uses a new intervention on the "
            "frozen Head 0, with prior validation trajectories as targets and disjoint discovery "
            "trajectories as donors; it is not an independent new-trajectory replication."
        ),
        "target_trajectories": sorted(target_ids),
        "donor_trajectories": sorted(donor_ids),
        "weak_rule": "baseline_gt_margin <= eligible-rescue-target median",
        "weak_threshold": threshold,
        "baseline_incorrect_rule": "baseline_gt_margin < 0",
        "donor_rules": {
            "exact_match": "same t, S_prev, current event, and GT S_t; different trajectory",
            "strong_matched_donor": "maximum baseline GT margin; lexicographic tie break",
            "ordinary_matched_donor": "lexicographically first exact-match donor; no model-result selection",
            "shuffled_matched_donor": "fixed-seed random draw from exact-match donors",
            "different_event_matched_history": "same t and S_prev, different event and different resulting state; strongest baseline donor",
        },
        "patch": {
            "model": "Qwen3-VL-8B-Instruct",
            "layer_one_based": LAYER,
            "layer_zero_based": LAYER - 1,
            "head": HEAD,
            "token": "final prompt token",
            "location": "layer[23].self_attn.o_proj forward_pre_hook",
            "slice": "pre-o_proj concatenated attention output [0:128]",
            "target_visual_input": "unchanged target video for every patched condition",
        },
        "source_files": {
            "split": str(args.split),
            "metadata": str(args.metadata),
            "behavior": str(args.behavior),
            "model_config": str(model_config),
            "split_sha256": sha256(Path(args.split)),
            "metadata_sha256": sha256(Path(args.metadata)),
            "behavior_sha256": sha256(Path(args.behavior)),
            "model_config_sha256": sha256(model_config) if model_config.exists() else None,
        },
        "counts": {
            "available_target_prefixes": int(len(targets)),
            "eligible_target_prefixes": int(len(entries)),
            "matching_success_rate": float(len(entries) / len(targets)),
            "target_trajectories": int(len({entry["target_trajectory"] for entry in entries})),
            "weak": int(sum(entry["strength_group"] == "weak" for entry in entries)),
            "strong": int(sum(entry["strength_group"] == "strong" for entry in entries)),
            "baseline_incorrect": int(sum(entry["baseline_incorrect"] for entry in entries)),
            "t": {str(t): int(sum(entry["t"] == t for entry in entries)) for t in range(1, 6)},
        },
        "failed_matches": failed,
        "targets": entries,
    }
    return manifest


def print_audit(manifest: dict) -> None:
    report = {
        "status": "RESCUE_MANIFEST_READY",
        "independence_caveat": manifest["independence_caveat"],
        **manifest["counts"],
        "weak_threshold": manifest["weak_threshold"],
        "failed_matches": manifest["failed_matches"],
        "conditions": list(CONDITIONS),
        "estimated_forwards_without_cache": len(manifest["targets"]) * 10,
    }
    print(json.dumps(report, indent=2))


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    (root / "rescue_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print_audit(manifest)


def audit(args: argparse.Namespace) -> None:
    manifest = json.loads((Path(args.out) / "rescue_manifest.json").read_text())
    target_ids = set(manifest["target_trajectories"])
    donor_ids = set(manifest["donor_trajectories"])
    assert not target_ids & donor_ids
    assert manifest["patch"]["layer_one_based"] == 24 and manifest["patch"]["head"] == 0
    for entry in manifest["targets"]:
        for condition in CONDITIONS:
            donor = entry["donors"][condition]
            assert donor["trajectory_id"] != entry["target_trajectory"] or condition == "self_patch"
            assert donor["t"] == entry["t"]
            if condition in {"strong_matched_donor", "ordinary_matched_donor", "shuffled_matched_donor"}:
                assert donor["prev_state"] == entry["prev_state"]
                assert donor["event"] == entry["current_event"]
                assert donor["state"] == entry["gt_state"]
            if condition == "different_event_matched_history":
                assert donor["prev_state"] == entry["prev_state"]
                assert donor["event"] != entry["current_event"]
                assert donor["state"] != entry["gt_state"]
    print_audit(manifest)
    print("RESCUE_AUDIT_PASS")


def unit() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from head_localization_l24 import pre_oproj_hook, run_unit
    run_unit()
    x = torch.arange(2 * 3 * 4096, dtype=torch.float32).reshape(2, 3, 4096)
    replacement = torch.randn(1, 128)
    patched = pre_oproj_hook(replacement, [0], 2)(None, (x,))[0]
    assert torch.equal(patched[:, :2], x[:, :2])
    assert torch.equal(patched[:, 2, 128:], x[:, 2, 128:])
    assert torch.equal(patched[:, 2, :128], replacement.expand(2, -1))
    print("RESCUE_UNIT_PASS: only Head 0 at final token changes")


def run(args: argparse.Namespace) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import circuit_localization
    from head_localization_l24 import pre_oproj_hook
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS, load_model_and_processor, render_inputs, to_device

    root = Path(args.out)
    manifest = json.loads((root / "rescue_manifest.json").read_text())
    print_audit(manifest)
    targets = manifest["targets"][args.shard_index::args.num_shards]
    print(json.dumps({"shard": args.shard_index, "num_shards": args.num_shards,
                      "targets": len(targets),
                      "trajectories": len({row['target_trajectory'] for row in targets})}, indent=2))
    model, processor = load_model_and_processor(Path(args.model_dir))
    model.eval()
    device = next(model.parameters()).device
    block = circuit_localization.resolve_decoder_layers(model)[LAYER - 1]
    oproj = block.self_attn.o_proj
    if int(getattr(block.self_attn, "num_heads", 32)) != 32 or int(getattr(block.self_attn, "head_dim", 128)) != 128:
        raise RuntimeError("unexpected Qwen3-VL L24 attention structure")

    def render_prefix(prefix: dict) -> tuple[dict, dict]:
        clip = sample_clip(Path(args.dataset) / f"{prefix['trajectory_id']}.mp4", 0, int(prefix["frame_end"]))
        inputs, fingerprint = render_inputs(
            processor, state_messages(clip, prefix["initial_state"], int(prefix["t"])), clip, REGIME
        )
        return to_device(inputs, device), fingerprint

    def native_logits(inputs: dict) -> dict[str, float]:
        with torch.inference_mode():
            output = model(**inputs, logits_to_keep=1)
        logits = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        return {state: float(logits[token_id]) for state, token_id in POS_IDS.items()}

    def capture(prefix: dict) -> tuple[dict[str, float], torch.Tensor, dict]:
        inputs, fingerprint = render_prefix(prefix)
        position = inputs["input_ids"].shape[1] - 1
        saved = {}
        def hook(_module, values):
            saved["head"] = values[0][0, position, :128].detach().float().cpu().clone()
            return values
        handle = oproj.register_forward_pre_hook(hook)
        try:
            logits = native_logits(inputs)
        finally:
            handle.remove()
        return logits, saved["head"], fingerprint

    def patch_target(inputs: dict, activation: torch.Tensor) -> dict[str, float]:
        position = inputs["input_ids"].shape[1] - 1
        handle = oproj.register_forward_pre_hook(pre_oproj_hook(activation.reshape(1, 128), [HEAD], position))
        try:
            return native_logits(inputs)
        finally:
            handle.remove()

    donor_cache: dict[str, tuple[dict[str, float], torch.Tensor, dict]] = {}
    records = []
    for index, entry in enumerate(targets, start=1):
        target = {
            "trajectory_id": entry["target_trajectory"], "target_prefix": entry["target_prefix"],
            "t": entry["t"], "frame_end": entry["frame_end"], "initial_state": entry["initial_state"],
        }
        target_inputs, target_fp = render_prefix(target)
        target_position = target_inputs["input_ids"].shape[1] - 1
        target_saved = {}
        def target_hook(_module, values):
            target_saved["head"] = values[0][0, target_position, :128].detach().float().cpu().clone()
            return values
        handle = oproj.register_forward_pre_hook(target_hook)
        try:
            baseline_logits = native_logits(target_inputs)
        finally:
            handle.remove()
        for condition in CONDITIONS:
            donor = entry["donors"][condition]
            if condition == "self_patch":
                donor_logits, donor_activation, donor_fp = baseline_logits, target_saved["head"], target_fp
            else:
                donor_key = donor["target_prefix"]
                if donor_key not in donor_cache:
                    donor_cache[donor_key] = capture(donor)
                donor_logits, donor_activation, donor_fp = donor_cache[donor_key]
            patched_logits = patch_target(target_inputs, donor_activation)
            gt = entry["gt_state"]
            others = [state for state in STATES if state != gt]
            base_margin = baseline_logits[gt] - max(baseline_logits[state] for state in others)
            patched_margin = patched_logits[gt] - max(patched_logits[state] for state in others)
            delta = {state: patched_logits[state] - baseline_logits[state] for state in STATES}
            cf_state = donor["state"] if condition == "different_event_matched_history" else ""
            cf_shift = ""
            if cf_state:
                cf_shift = ((patched_logits[cf_state] - patched_logits[gt])
                            - (baseline_logits[cf_state] - baseline_logits[gt]))
            records.append({
                "target_prefix": entry["target_prefix"],
                "target_trajectory": entry["target_trajectory"],
                "t": entry["t"],
                "condition": condition,
                "donor_prefix": donor["target_prefix"],
                "donor_trajectory": donor["trajectory_id"],
                "donor_baseline_gt_margin": donor["baseline_gt_margin"],
                "prev_state": entry["prev_state"],
                "target_event": entry["current_event"],
                "donor_event": donor["event"],
                "gt_state": gt,
                "counterfactual_state": cf_state,
                "strength_group": entry["strength_group"],
                "manifest_baseline_gt_margin": entry["baseline_gt_margin"],
                "baseline_gt_margin": base_margin,
                "patched_gt_margin": patched_margin,
                "delta_gt_margin": patched_margin - base_margin,
                "baseline_pred": max(STATES, key=baseline_logits.get),
                "patched_pred": max(STATES, key=patched_logits.get),
                "baseline_correct": max(STATES, key=baseline_logits.get) == gt,
                "patched_correct": max(STATES, key=patched_logits.get) == gt,
                "wrong_to_correct": max(STATES, key=baseline_logits.get) != gt and max(STATES, key=patched_logits.get) == gt,
                "correct_to_wrong": max(STATES, key=baseline_logits.get) == gt and max(STATES, key=patched_logits.get) != gt,
                "delta_logit_gt": delta[gt],
                "delta_logit_other_max": max(delta[state] for state in others),
                "gt_state_specificity": delta[gt] - max(delta[state] for state in others),
                "counterfactual_margin_shift": cf_shift,
                **{f"baseline_logit_{state}": baseline_logits[state] for state in STATES},
                **{f"patched_logit_{state}": patched_logits[state] for state in STATES},
                "target_input_ids_sha256": target_fp["input_ids_sha256"],
                "target_pixel_sha256": target_fp["pixel_values_videos_sha256"],
                "donor_input_ids_sha256": donor_fp["input_ids_sha256"],
                "donor_pixel_sha256": donor_fp["pixel_values_videos_sha256"],
                "target_video_grid_thw": json.dumps(target_fp["video_grid_thw"]),
                "donor_video_grid_thw": json.dumps(donor_fp["video_grid_thw"]),
            })
        print(f"[{index}/{len(targets)}] {entry['target_prefix']} complete", flush=True)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    output = shard_dir / f"rescue_shard_{args.shard_index}.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    print(json.dumps({"status": "RESCUE_SHARD_COMPLETE", "output": str(output),
                      "targets": len(targets), "rows": len(records),
                      "cached_donors": len(donor_cache)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="outputs/vetbench/head0_causal_rescue_v1")
    common.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    common.add_argument("--dataset", default="dataset/vetbench/cup")
    common.add_argument("--split", default="outputs/vetbench/head_localization_l24_v1/discovery_validation_split.json")
    common.add_argument("--metadata", default="outputs/vetbench/composition_analysis_v1/transformers_behavior.csv")
    common.add_argument("--behavior", default="outputs/vetbench/mechanism_gate_final/behavior.csv")
    common.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("prepare", parents=[common])
    sub.add_parser("audit", parents=[common])
    sub.add_parser("unit", parents=[common])
    run_parser = sub.add_parser("run", parents=[common])
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "audit": audit(args)
    elif args.command == "unit": unit()
    else: run(args)


if __name__ == "__main__":
    main()
