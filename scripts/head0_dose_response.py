#!/usr/bin/env python3
"""Frozen L24 Head 0 activation dose-response experiment.

Only ``run`` loads Qwen3-VL. The target/donor pairs are copied verbatim from
head0_causal_rescue_v1 and the intervention remains the final-token Head 0
slice at the L24 attention o_proj input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAYER = 24
HEAD = 0
HEAD_DIM = 128
SEED = 20260909
REGIME = "controlled_8fps"
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DONOR_TYPES = ("strong_same_state", "different_event")
SOURCE_DONORS = {
    "strong_same_state": "strong_matched_donor",
    "different_event": "different_event_matched_history",
}
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
    if source["patch"]["layer_one_based"] != LAYER or source["patch"]["head"] != HEAD:
        raise AssertionError("source rescue manifest does not use frozen L24 Head 0")
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
            "baseline_incorrect": entry["baseline_incorrect"],
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
        "alphas": list(ALPHAS),
        "donor_types": list(DONOR_TYPES),
        "interpolation": "h_alpha = h_target + alpha * (h_donor - h_target)",
        "patch": source["patch"],
        "regime": source["regime"],
        "dataset": source["dataset"],
        "target_split": source["target_split"],
        "donor_pool": source["donor_pool"],
        "independence_caveat": source["independence_caveat"],
        "weak_rule": source["weak_rule"],
        "weak_threshold": source["weak_threshold"],
        "baseline_incorrect_rule": source["baseline_incorrect_rule"],
        "primary_endpoints": {
            "strong_same_state": "GT margin = logit(GT) - max(logit(other states))",
            "different_event": "donor-state counterfactual margin = logit(donor state) - logit(target GT)",
        },
        "monotonicity_rule": (
            "Report target-level monotonic fraction and trajectory-cluster adjacent-alpha effects. "
            "A clear population dose response requires nonnegative adjacent-step mean effects with "
            "cluster CIs not materially below zero and a positive within-target dose slope."
        ),
        "decision_rules": {
            "categorical_improvement": "alpha=2 wrong-to-correct trajectory-cluster bootstrap CI lower bound > 0",
            "commitment_bottleneck_pattern": (
                "clear continuous same-state dose response and more than half of initially "
                "negative-margin targets remain below the GT decision boundary at alpha=2"
            ),
            "interpretation": "the bottleneck pattern is consistency evidence, not proof of a unique downstream cause",
        },
        "large_alpha_anomaly_metrics": [
            "centered_three_state_logit_l2",
            "new_unintended_prediction",
            "desired_state_specificity",
            "native_three_state_entropy",
        ],
        "counts": {
            "target_prefixes": len(targets),
            "target_trajectories": len({entry["target_trajectory"] for entry in targets}),
            "rows_expected": len(targets) * len(DONOR_TYPES) * len(ALPHAS),
            "weak": sum(entry["strength_group"] == "weak" for entry in targets),
            "strong": sum(entry["strength_group"] == "strong" for entry in targets),
            "baseline_margin_lt_zero": sum(entry["baseline_gt_margin"] < 0 for entry in targets),
        },
        "targets": targets,
    }


def print_audit(manifest: dict) -> None:
    print(json.dumps({
        "status": "DOSE_RESPONSE_MANIFEST_READY",
        "frozen_pairs": manifest["frozen_pairs"],
        "alphas": manifest["alphas"],
        "donor_types": manifest["donor_types"],
        "patch": manifest["patch"],
        "counts": manifest["counts"],
        "independence_caveat": manifest["independence_caveat"],
    }, indent=2))


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    (root / "dose_response_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print_audit(manifest)


def audit(args: argparse.Namespace) -> None:
    root = Path(args.out)
    manifest = json.loads((root / "dose_response_manifest.json").read_text())
    source_path = Path(manifest["source_rescue_manifest"])
    if file_sha256(source_path) != manifest["source_rescue_manifest_sha256"]:
        raise AssertionError("source rescue manifest changed after dose manifest freeze")
    source = json.loads(source_path.read_text())
    source_by_target = {entry["target_prefix"]: entry for entry in source["targets"]}
    assert tuple(manifest["alphas"]) == ALPHAS
    assert tuple(manifest["donor_types"]) == DONOR_TYPES
    assert manifest["patch"]["layer_one_based"] == LAYER
    assert manifest["patch"]["head"] == HEAD
    assert manifest["patch"]["token"] == "final prompt token"
    for entry in manifest["targets"]:
        original = source_by_target[entry["target_prefix"]]
        assert entry["donors"]["strong_same_state"] == original["donors"]["strong_matched_donor"]
        assert entry["donors"]["different_event"] == original["donors"]["different_event_matched_history"]
        same = entry["donors"]["strong_same_state"]
        different = entry["donors"]["different_event"]
        assert same["trajectory_id"] != entry["target_trajectory"]
        assert same["t"] == entry["t"] and same["prev_state"] == entry["prev_state"]
        assert same["event"] == entry["current_event"] and same["state"] == entry["gt_state"]
        assert different["trajectory_id"] != entry["target_trajectory"]
        assert different["t"] == entry["t"] and different["prev_state"] == entry["prev_state"]
        assert different["event"] != entry["current_event"] and different["state"] != entry["gt_state"]
    print_audit(manifest)
    print("DOSE_RESPONSE_AUDIT_PASS")


def unit() -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from head_localization_l24 import pre_oproj_hook
    target = torch.randn(HEAD_DIM)
    donor = torch.randn(HEAD_DIM)
    assert torch.equal(target + 0.0 * (donor - target), target)
    assert torch.allclose(target + 1.0 * (donor - target), donor)
    assert torch.allclose(target + 2.0 * (donor - target), 2.0 * donor - target)
    concat = torch.randn(1, 5, 4096)
    position = 4
    alpha = 0.5
    interpolated = target + alpha * (donor - target)
    patched = pre_oproj_hook(interpolated.reshape(1, HEAD_DIM), [HEAD], position)(None, (concat,))[0]
    assert torch.equal(patched[:, :position], concat[:, :position])
    assert torch.equal(patched[:, position, HEAD_DIM:], concat[:, position, HEAD_DIM:])
    assert torch.equal(patched[0, position, :HEAD_DIM], interpolated)
    print("DOSE_RESPONSE_UNIT_PASS: interpolation and frozen Head 0 slice")


def run(args: argparse.Namespace) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import circuit_localization
    from head_localization_l24 import pre_oproj_hook
    from run_state_rev_audit import state_messages
    from run_vetbench_screening import sample_clip
    from state_rev_input_pipeline import POS_IDS, load_model_and_processor, render_inputs, to_device

    root = Path(args.out)
    manifest = json.loads((root / "dose_response_manifest.json").read_text())
    targets = manifest["targets"][args.shard_index::args.num_shards]
    print_audit(manifest)
    print(json.dumps({
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "target_prefixes": len(targets),
        "target_trajectories": len({entry["target_trajectory"] for entry in targets}),
        "rows_expected": len(targets) * len(DONOR_TYPES) * len(ALPHAS),
    }, indent=2))

    model, processor = load_model_and_processor(Path(args.model_dir))
    model.eval()
    device = next(model.parameters()).device
    block = circuit_localization.resolve_decoder_layers(model)[LAYER - 1]
    oproj = block.self_attn.o_proj
    if int(getattr(block.self_attn, "num_heads", 32)) != 32 or int(getattr(block.self_attn, "head_dim", HEAD_DIM)) != HEAD_DIM:
        raise RuntimeError("unexpected Qwen3-VL L24 attention structure")

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

    def native_logits(inputs: dict) -> dict[str, float]:
        with torch.inference_mode():
            output = model(**inputs, logits_to_keep=1)
        logprobs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        return {state: float(logprobs[token]) for state, token in POS_IDS.items()}

    def capture(prefix: dict) -> tuple[dict[str, float], torch.Tensor, dict]:
        inputs, fingerprint = render_prefix(prefix)
        position = inputs["input_ids"].shape[1] - 1
        saved = {}
        def hook(_module, values):
            saved["head"] = values[0][0, position, :HEAD_DIM].detach().float().cpu().clone()
            return values
        handle = oproj.register_forward_pre_hook(hook)
        try:
            logits = native_logits(inputs)
        finally:
            handle.remove()
        return logits, saved["head"], fingerprint

    def patched_logits(inputs: dict, activation: torch.Tensor) -> dict[str, float]:
        position = inputs["input_ids"].shape[1] - 1
        hook = pre_oproj_hook(activation.reshape(1, HEAD_DIM), [HEAD], position)
        handle = oproj.register_forward_pre_hook(hook)
        try:
            return native_logits(inputs)
        finally:
            handle.remove()

    donor_cache: dict[str, tuple[dict[str, float], torch.Tensor, dict]] = {}
    records = []
    for target_index, entry in enumerate(targets, start=1):
        target = {
            "trajectory_id": entry["target_trajectory"],
            "target_prefix": entry["target_prefix"],
            "t": entry["t"],
            "frame_end": entry["frame_end"],
            "initial_state": entry["initial_state"],
        }
        target_inputs, target_fp = render_prefix(target)
        position = target_inputs["input_ids"].shape[1] - 1
        saved = {}
        def target_hook(_module, values):
            saved["head"] = values[0][0, position, :HEAD_DIM].detach().float().cpu().clone()
            return values
        handle = oproj.register_forward_pre_hook(target_hook)
        try:
            baseline = native_logits(target_inputs)
        finally:
            handle.remove()
        target_activation = saved["head"]
        baseline_pred = max(STATES, key=baseline.get)
        gt = entry["gt_state"]
        baseline_gt_margin = baseline[gt] - max(baseline[state] for state in STATES if state != gt)

        for donor_type in DONOR_TYPES:
            donor = entry["donors"][donor_type]
            donor_key = donor["target_prefix"]
            if donor_key not in donor_cache:
                donor_cache[donor_key] = capture(donor)
            donor_native, donor_activation, donor_fp = donor_cache[donor_key]
            desired = gt if donor_type == "strong_same_state" else donor["state"]
            third = next(state for state in STATES if state not in {gt, desired}) if desired != gt else ""
            baseline_desired_margin = (
                baseline[desired] - max(baseline[state] for state in STATES if state != desired)
                if desired == gt else baseline[desired] - baseline[gt]
            )
            direction = donor_activation - target_activation
            for alpha in ALPHAS:
                activation = target_activation + alpha * direction
                # alpha=0 is algebraically the clean target activation. Reuse
                # the clean logits for both donor curves instead of spending
                # two redundant model forwards per target.
                patched = baseline if alpha == 0.0 else patched_logits(target_inputs, activation)
                patched_pred = max(STATES, key=patched.get)
                patched_gt_margin = patched[gt] - max(patched[state] for state in STATES if state != gt)
                patched_desired_margin = (
                    patched[desired] - max(patched[state] for state in STATES if state != desired)
                    if desired == gt else patched[desired] - patched[gt]
                )
                delta = np.asarray([patched[state] - baseline[state] for state in STATES], dtype=float)
                centered = delta - delta.mean()
                three_prob = np.exp(np.asarray([patched[state] for state in STATES], dtype=float))
                three_prob /= three_prob.sum()
                desired_specificity = (
                    patched[desired] - baseline[desired]
                    - max(patched[state] - baseline[state] for state in STATES if state != desired)
                )
                records.append({
                    "target_prefix": entry["target_prefix"],
                    "target_trajectory": entry["target_trajectory"],
                    "t": entry["t"],
                    "donor_type": donor_type,
                    "donor_prefix": donor_key,
                    "donor_trajectory": donor["trajectory_id"],
                    "alpha": alpha,
                    "prev_state": entry["prev_state"],
                    "target_event": entry["current_event"],
                    "donor_event": donor["event"],
                    "gt_state": gt,
                    "desired_state": desired,
                    "third_state": third,
                    "strength_group": entry["strength_group"],
                    "manifest_baseline_gt_margin": entry["baseline_gt_margin"],
                    "baseline_gt_margin": baseline_gt_margin,
                    "patched_gt_margin": patched_gt_margin,
                    "delta_gt_margin": patched_gt_margin - baseline_gt_margin,
                    "baseline_desired_margin": baseline_desired_margin,
                    "patched_desired_margin": patched_desired_margin,
                    "delta_desired_margin": patched_desired_margin - baseline_desired_margin,
                    "desired_state_specificity": desired_specificity,
                    "baseline_pred": baseline_pred,
                    "patched_pred": patched_pred,
                    "baseline_correct": baseline_pred == gt,
                    "patched_correct": patched_pred == gt,
                    "wrong_to_correct": baseline_pred != gt and patched_pred == gt,
                    "correct_to_wrong": baseline_pred == gt and patched_pred != gt,
                    "flips_to_desired": baseline_pred != desired and patched_pred == desired,
                    "new_unintended_prediction": patched_pred not in {baseline_pred, desired},
                    "centered_three_state_logit_l2": float(np.linalg.norm(centered)),
                    "max_abs_centered_logit_change": float(np.abs(centered).max()),
                    "native_three_state_entropy": float(-(three_prob * np.log(three_prob + 1e-30)).sum()),
                    "target_activation_norm": float(target_activation.norm()),
                    "donor_direction_norm": float(direction.norm()),
                    "applied_activation_shift_norm": float((alpha * direction).norm()),
                    **{f"baseline_logit_{state}": baseline[state] for state in STATES},
                    **{f"patched_logit_{state}": patched[state] for state in STATES},
                    **{f"delta_logit_{state}": patched[state] - baseline[state] for state in STATES},
                    "target_input_ids_sha256": target_fp["input_ids_sha256"],
                    "target_pixel_sha256": target_fp["pixel_values_videos_sha256"],
                    "target_video_grid_thw": json.dumps(target_fp["video_grid_thw"]),
                    "donor_input_ids_sha256": donor_fp["input_ids_sha256"],
                    "donor_pixel_sha256": donor_fp["pixel_values_videos_sha256"],
                    "donor_video_grid_thw": json.dumps(donor_fp["video_grid_thw"]),
                    "donor_native_gt_margin": donor_native[donor["state"]]
                    - max(donor_native[state] for state in STATES if state != donor["state"]),
                })
        print(f"[{target_index}/{len(targets)}] {entry['target_prefix']} complete", flush=True)

    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    output = shard_dir / f"dose_response_shard_{args.shard_index}.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    print(json.dumps({
        "status": "DOSE_RESPONSE_SHARD_COMPLETE",
        "output": str(output),
        "target_prefixes": len(targets),
        "rows": len(records),
        "cached_donors": len(donor_cache),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="outputs/vetbench/head0_dose_response_v1")
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
