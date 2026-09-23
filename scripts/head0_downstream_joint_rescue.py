#!/usr/bin/env python3
"""Frozen L24 Head 0 plus L32/L36 block-output rescue experiment.

Only ``run`` loads Qwen3-VL. Target/donor pairs are copied verbatim from the
Head 0 rescue manifest. A downstream final-token block-output replacement
structurally overwrites the earlier final-token Head 0 intervention, so joint
rows are expected to equal block-only rows and are emitted without a redundant
forward after that equivalence is covered by unit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEAD_LAYER = 24
HEAD = 0
HEAD_DIM = 128
BLOCK_LAYERS = (32, 36)
ALPHA = 1.0
SEED = 20260910
REGIME = "controlled_8fps"
DONOR_TYPES = ("strong_same_state", "different_event")
SOURCE_DONORS = {
    "strong_same_state": "strong_matched_donor",
    "different_event": "different_event_matched_history",
}
CONDITIONS = ("baseline", "head0_only", "l32_only", "l36_only", "head0_l32", "head0_l36")
STATES = ("Left", "Middle", "Right")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(args: argparse.Namespace) -> dict:
    source_path = Path(args.rescue_manifest)
    source = json.loads(source_path.read_text())
    patch = source["patch"]
    if patch["layer_one_based"] != HEAD_LAYER or patch["head"] != HEAD:
        raise AssertionError("source manifest is not the frozen L24 Head 0 experiment")
    targets = []
    for entry in source["targets"]:
        targets.append({
            "target_prefix": entry["target_prefix"],
            "target_trajectory": entry["target_trajectory"],
            "t": entry["t"],
            "frame_end": entry["frame_end"],
            "initial_state": entry["initial_state"],
            "prev_state": entry["prev_state"],
            "current_event": entry["current_event"],
            "gt_state": entry["gt_state"],
            "strength_group": entry["strength_group"],
            "baseline_incorrect": bool(entry["baseline_gt_margin"] < 0),
            "baseline_gt_margin": entry["baseline_gt_margin"],
            "donors": {
                donor_type: entry["donors"][source_name]
                for donor_type, source_name in SOURCE_DONORS.items()
            },
        })
    return {
        "version": 1,
        "seed": SEED,
        "source_rescue_manifest": str(source_path),
        "source_rescue_manifest_sha256": file_sha256(source_path),
        "frozen_pairs": True,
        "donor_types": list(DONOR_TYPES),
        "conditions": list(CONDITIONS),
        "head_alpha": ALPHA,
        "patches": {
            "head0": patch,
            "downstream": {
                "layers_one_based": list(BLOCK_LAYERS),
                "token": "final prompt token",
                "location": "decoder block output",
                "operation": "replace target activation with clean donor activation",
            },
        },
        "structural_occlusion": {
            "expected": True,
            "reason": (
                "Replacing the complete final-token L32/L36 block output overwrites every upstream "
                "contribution at that token, including the L24 Head 0 patch. Under deterministic "
                "inference, Head0+block must therefore equal block-only. Positive synergy is not "
                "identifiable without changing the frozen block intervention (for example to an "
                "additive delta intervention)."
            ),
            "execution": "joint rows reuse corresponding block-only logits; no redundant GPU forward",
            "interaction_expectation": "delta_joint - delta_head0 - delta_block = -delta_head0",
        },
        "statistics": (
            "one frozen donor per target and donor type; target prefix first, then target trajectory "
            "cluster bootstrap and paired trajectory sign permutation"
        ),
        "regime": source["regime"],
        "dataset": source["dataset"],
        "target_split": source["target_split"],
        "donor_pool": source["donor_pool"],
        "independence_caveat": source["independence_caveat"],
        "weak_rule": source["weak_rule"],
        "weak_threshold": source["weak_threshold"],
        "baseline_incorrect_rule": "baseline GT margin < 0",
        "counts": {
            "target_prefixes": len(targets),
            "target_trajectories": len({entry["target_trajectory"] for entry in targets}),
            "donor_types": len(DONOR_TYPES),
            "conditions": len(CONDITIONS),
            "rows_expected": len(targets) * len(DONOR_TYPES) * len(CONDITIONS),
            "model_forwards_per_target_without_donor_cache": 9,
            "patched_target_forwards_per_target": 3,
            "weak": sum(entry["strength_group"] == "weak" for entry in targets),
            "strong": sum(entry["strength_group"] == "strong" for entry in targets),
            "baseline_margin_lt_zero": sum(entry["baseline_gt_margin"] < 0 for entry in targets),
        },
        "targets": targets,
    }


def print_audit(manifest: dict) -> None:
    print(json.dumps({
        "status": "JOINT_RESCUE_MANIFEST_READY",
        "frozen_pairs": manifest["frozen_pairs"],
        "donor_types": manifest["donor_types"],
        "conditions": manifest["conditions"],
        "patches": manifest["patches"],
        "counts": manifest["counts"],
        "structural_occlusion": manifest["structural_occlusion"],
    }, indent=2))


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    (root / "joint_rescue_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print_audit(manifest)


def audit(args: argparse.Namespace) -> None:
    root = Path(args.out)
    manifest = json.loads((root / "joint_rescue_manifest.json").read_text())
    source_path = Path(manifest["source_rescue_manifest"])
    if file_sha256(source_path) != manifest["source_rescue_manifest_sha256"]:
        raise AssertionError("source Head 0 rescue manifest changed after pair freeze")
    source = json.loads(source_path.read_text())
    originals = {entry["target_prefix"]: entry for entry in source["targets"]}
    assert tuple(manifest["conditions"]) == CONDITIONS
    assert tuple(manifest["patches"]["downstream"]["layers_one_based"]) == BLOCK_LAYERS
    for entry in manifest["targets"]:
        original = originals[entry["target_prefix"]]
        for donor_type, source_name in SOURCE_DONORS.items():
            donor = entry["donors"][donor_type]
            assert donor == original["donors"][source_name]
            assert donor["trajectory_id"] != entry["target_trajectory"]
            assert donor["t"] == entry["t"] and donor["prev_state"] == entry["prev_state"]
        same = entry["donors"]["strong_same_state"]
        different = entry["donors"]["different_event"]
        assert same["event"] == entry["current_event"] and same["state"] == entry["gt_state"]
        assert different["event"] != entry["current_event"] and different["state"] != entry["gt_state"]
    print_audit(manifest)
    print("JOINT_RESCUE_AUDIT_PASS")


def unit() -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from circuit_localization import replace_positions_hook
    from head_localization_l24 import pre_oproj_hook

    concat = torch.randn(1, 5, 4096)
    head = torch.randn(1, HEAD_DIM)
    patched = pre_oproj_hook(head, [HEAD], 4)(None, (concat,))[0]
    assert torch.equal(patched[0, 4, :HEAD_DIM], head[0])
    assert torch.equal(patched[0, 4, HEAD_DIM:], concat[0, 4, HEAD_DIM:])
    upstream_output = torch.randn(1, 5, 4096)
    donor_block = torch.randn(1, 4096)
    block_hook = replace_positions_hook([4], donor_block)
    block_only = block_hook(None, None, upstream_output)
    joint = block_hook(None, None, upstream_output + torch.randn_like(upstream_output))
    assert torch.equal(block_only[0, 4], donor_block[0])
    assert torch.equal(joint[0, 4], donor_block[0])
    print("JOINT_RESCUE_UNIT_PASS: Head 0 slice is correct; downstream full-token replacement occludes upstream state")


def run(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import circuit_localization
    from head_localization_l24 import pre_oproj_hook
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS, load_model_and_processor, render_inputs, to_device

    root = Path(args.out)
    manifest = json.loads((root / "joint_rescue_manifest.json").read_text())
    targets = manifest["targets"][args.shard_index::args.num_shards]
    print_audit(manifest)
    print(json.dumps({
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "target_prefixes": len(targets),
        "rows_expected": len(targets) * len(DONOR_TYPES) * len(CONDITIONS),
    }, indent=2))

    model, processor = load_model_and_processor(Path(args.model_dir))
    model.eval()
    device = next(model.parameters()).device
    blocks = circuit_localization.resolve_decoder_layers(model)
    oproj = blocks[HEAD_LAYER - 1].self_attn.o_proj
    if int(getattr(blocks[HEAD_LAYER - 1].self_attn, "num_heads", 32)) != 32:
        raise RuntimeError("unexpected L24 attention head count")
    if int(getattr(blocks[HEAD_LAYER - 1].self_attn, "head_dim", HEAD_DIM)) != HEAD_DIM:
        raise RuntimeError("unexpected L24 attention head dimension")

    def render_prefix(prefix: dict) -> tuple[dict, dict]:
        clip = sample_clip(
            Path(args.dataset) / f"{prefix['trajectory_id']}.mp4", 0, int(prefix["frame_end"])
        )
        inputs, fingerprint = render_inputs(
            processor,
            state_messages(clip, prefix["initial_state"], int(prefix["t"])),
            clip,
            REGIME,
        )
        return to_device(inputs, device), fingerprint

    def logits(inputs: dict) -> dict[str, float]:
        with torch.inference_mode():
            output = model(**inputs, logits_to_keep=1)
        values = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        return {state: float(values[token]) for state, token in POS_IDS.items()}

    def capture(prefix: dict) -> tuple[dict[str, float], dict, dict]:
        inputs, fingerprint = render_prefix(prefix)
        position = inputs["input_ids"].shape[1] - 1
        saved: dict[str, torch.Tensor] = {}

        def head_hook(_module, values):
            saved["head0"] = values[0][0, position, :HEAD_DIM].detach().float().cpu().clone()
            return values

        def block_hook(layer: int):
            def hook(_module, _values, output):
                hidden = circuit_localization.block_output_tensor(output)
                saved[f"l{layer}"] = hidden[0, position, :].detach().float().cpu().clone()
            return hook

        handles = [oproj.register_forward_pre_hook(head_hook)]
        handles += [blocks[layer - 1].register_forward_hook(block_hook(layer)) for layer in BLOCK_LAYERS]
        try:
            native = logits(inputs)
        finally:
            for handle in handles:
                handle.remove()
        return native, saved, fingerprint

    def patched_logits(inputs: dict, donor_activation: dict, head0: bool, block_layer: int | None) -> dict[str, float]:
        position = inputs["input_ids"].shape[1] - 1
        handles = []
        if head0:
            hook = pre_oproj_hook(donor_activation["head0"].reshape(1, HEAD_DIM), [HEAD], position)
            handles.append(oproj.register_forward_pre_hook(hook))
        if block_layer is not None:
            replacement = donor_activation[f"l{block_layer}"].reshape(1, -1)
            hook = circuit_localization.replace_positions_hook([position], replacement)
            handles.append(blocks[block_layer - 1].register_forward_hook(hook))
        try:
            return logits(inputs)
        finally:
            for handle in handles:
                handle.remove()

    donor_cache: dict[str, tuple[dict[str, float], dict, dict]] = {}
    records = []
    progress_path = root / "shards" / f"joint_rescue_shard_{args.shard_index}.progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = tqdm(targets, desc=f"joint shard {args.shard_index}", unit="target", dynamic_ncols=True)
    for target_index, entry in enumerate(iterator, start=1):
        target = {
            "trajectory_id": entry["target_trajectory"],
            "target_prefix": entry["target_prefix"],
            "t": entry["t"],
            "frame_end": entry["frame_end"],
            "initial_state": entry["initial_state"],
        }
        target_inputs, target_fp = render_prefix(target)
        position = target_inputs["input_ids"].shape[1] - 1
        target_saved: dict[str, torch.Tensor] = {}

        def target_head_hook(_module, values):
            target_saved["head0"] = values[0][0, position, :HEAD_DIM].detach().float().cpu().clone()
            return values

        def target_block_hook(layer: int):
            def hook(_module, _values, output):
                hidden = circuit_localization.block_output_tensor(output)
                target_saved[f"l{layer}"] = hidden[0, position, :].detach().float().cpu().clone()
            return hook

        handles = [oproj.register_forward_pre_hook(target_head_hook)]
        handles += [blocks[layer - 1].register_forward_hook(target_block_hook(layer)) for layer in BLOCK_LAYERS]
        try:
            baseline = logits(target_inputs)
        finally:
            for handle in handles:
                handle.remove()

        for donor_type in DONOR_TYPES:
            donor = entry["donors"][donor_type]
            donor_key = donor["target_prefix"]
            if donor_key not in donor_cache:
                donor_cache[donor_key] = capture(donor)
            donor_native, donor_activation, donor_fp = donor_cache[donor_key]
            computed = {
                "baseline": baseline,
                "head0_only": patched_logits(target_inputs, donor_activation, True, None),
                "l32_only": patched_logits(target_inputs, donor_activation, False, 32),
                "l36_only": patched_logits(target_inputs, donor_activation, False, 36),
            }
            # Full-token downstream replacement deterministically discards the
            # upstream final-token Head 0 intervention. Preserve the requested
            # joint conditions while avoiding two redundant model forwards.
            computed["head0_l32"] = computed["l32_only"].copy()
            computed["head0_l36"] = computed["l36_only"].copy()

            gt = entry["gt_state"]
            desired = gt if donor_type == "strong_same_state" else donor["state"]
            baseline_pred = max(STATES, key=baseline.get)
            baseline_gt_margin = baseline[gt] - max(baseline[state] for state in STATES if state != gt)
            baseline_cf_margin = baseline[desired] - baseline[gt] if desired != gt else np.nan
            for condition in CONDITIONS:
                patched = computed[condition]
                patched_pred = max(STATES, key=patched.get)
                patched_gt_margin = patched[gt] - max(patched[state] for state in STATES if state != gt)
                patched_cf_margin = patched[desired] - patched[gt] if desired != gt else np.nan
                delta = {state: patched[state] - baseline[state] for state in STATES}
                records.append({
                    "target_prefix": entry["target_prefix"],
                    "target_trajectory": entry["target_trajectory"],
                    "t": entry["t"],
                    "donor_type": donor_type,
                    "condition": condition,
                    "donor_prefix": donor_key,
                    "donor_trajectory": donor["trajectory_id"],
                    "prev_state": entry["prev_state"],
                    "target_event": entry["current_event"],
                    "donor_event": donor["event"],
                    "gt_state": gt,
                    "desired_state": desired,
                    "strength_group": entry["strength_group"],
                    "manifest_baseline_gt_margin": entry["baseline_gt_margin"],
                    "baseline_gt_margin": baseline_gt_margin,
                    "patched_gt_margin": patched_gt_margin,
                    "delta_gt_margin": patched_gt_margin - baseline_gt_margin,
                    "baseline_counterfactual_margin": baseline_cf_margin,
                    "patched_counterfactual_margin": patched_cf_margin,
                    "delta_counterfactual_margin": patched_cf_margin - baseline_cf_margin,
                    "baseline_pred": baseline_pred,
                    "patched_pred": patched_pred,
                    "baseline_correct": baseline_pred == gt,
                    "patched_correct": patched_pred == gt,
                    "wrong_to_correct": baseline_pred != gt and patched_pred == gt,
                    "correct_to_wrong": baseline_pred == gt and patched_pred != gt,
                    "flips_to_desired": baseline_pred != desired and patched_pred == desired,
                    "gt_state_specificity": delta[gt] - max(delta[state] for state in STATES if state != gt),
                    **{f"baseline_logit_{state}": baseline[state] for state in STATES},
                    **{f"patched_logit_{state}": patched[state] for state in STATES},
                    **{f"delta_logit_{state}": delta[state] for state in STATES},
                    "target_input_ids_sha256": target_fp["input_ids_sha256"],
                    "target_pixel_sha256": target_fp["pixel_values_videos_sha256"],
                    "target_video_grid_thw": json.dumps(target_fp["video_grid_thw"]),
                    "donor_input_ids_sha256": donor_fp["input_ids_sha256"],
                    "donor_pixel_sha256": donor_fp["pixel_values_videos_sha256"],
                    "donor_video_grid_thw": json.dumps(donor_fp["video_grid_thw"]),
                    "donor_native_gt_margin": donor_native[donor["state"]]
                    - max(donor_native[state] for state in STATES if state != donor["state"]),
                    "joint_logits_reused_from_block_only": condition.startswith("head0_l"),
                })
        progress_path.write_text(json.dumps({
            "shard": args.shard_index,
            "completed_targets": target_index,
            "total_targets": len(targets),
            "current_target": entry["target_prefix"],
        }) + "\n")

    output = root / "shards" / f"joint_rescue_shard_{args.shard_index}.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    progress_path.write_text(json.dumps({
        "shard": args.shard_index,
        "completed_targets": len(targets),
        "total_targets": len(targets),
        "status": "complete",
    }) + "\n")
    print(json.dumps({
        "status": "JOINT_RESCUE_SHARD_COMPLETE",
        "output": str(output),
        "targets": len(targets),
        "rows": len(records),
        "cached_donors": len(donor_cache),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="outputs/vetbench/head0_downstream_joint_rescue_v1")
    common.add_argument("--rescue-manifest", default="outputs/vetbench/head0_causal_rescue_v1/rescue_manifest.json")
    common.add_argument("--model-dir", default="models/Qwen3-VL-8B-Instruct")
    common.add_argument("--dataset", default="dataset/vetbench/cup")
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
