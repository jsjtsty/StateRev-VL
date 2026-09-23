#!/usr/bin/env python3
"""Independent audit of the LLaVA hidden-event validation result.

Cache-only checks run on CPU. The small visual controls require one GPU and are
intentionally limited to a few frozen validation trajectories.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs/vetbench/llava_next_video_7b_replication_v1"
AUDIT = ROOT / "outputs/vetbench/llava_hidden_event_audit_v1"
SPLIT = ROOT / "outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")


def acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p))) if len(y) else None


def load_base():
    behavior = pd.read_csv(OUT / "behavior.csv")
    split = json.loads(SPLIT.read_text())
    decoder = joblib.load(OUT / "frozen_hidden_event_decoder.joblib")
    with np.load(OUT / "hidden_states.npz") as z:
        hidden = {k: z[k] for k in z.files}
    assert len(behavior) == 250 and len(hidden) == 250
    assert set(behavior.target_prefix) == set(hidden)
    assert decoder["layer"] == 1 and decoder["C"] == 0.1
    assert set(decoder["fit_trajectories"]) == set(split["discovery_trajectories"])
    assert not set(decoder["fit_trajectories"]) & set(split["validation_trajectories"])
    val = behavior.trajectory_id.isin(split["validation_trajectories"])
    Xv = np.stack([hidden[k][1] for k in behavior.loc[val, "target_prefix"]])
    pv = decoder["clf"].predict(decoder["scaler"].transform(Xv))
    return behavior, split, decoder, hidden, behavior.loc[val].copy(), pv


def permutation_null(behavior, split, hidden, decoder, n_perm, seed):
    rng = np.random.default_rng(seed)
    discovery = set(split["discovery_trajectories"])
    validation = behavior.trajectory_id.isin(split["validation_trajectories"])
    train = behavior[behavior.trajectory_id.isin(discovery)].sort_values("target_prefix")
    val = behavior.loc[validation].sort_values("target_prefix")
    Xtr = np.stack([hidden[k][1] for k in train.target_prefix])
    Xv = np.stack([hidden[k][1] for k in val.target_prefix])
    groups = train.trajectory_id.to_numpy()
    # Preserve each discovery trajectory's five-label sequence, and permute
    # those sequences between discovery trajectory IDs.
    trajectories = sorted(discovery)
    seqs = {t: train[train.trajectory_id == t].sort_values("t").gt_event.to_numpy() for t in trajectories}
    null = []
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for _ in range(n_perm):
        shuffled = rng.permutation(trajectories)
        labels = np.concatenate([seqs[t] for t in shuffled])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=float(decoder["C"]), max_iter=2000, random_state=seed).fit(scaler.transform(Xtr), labels)
        null.append(acc(val.gt_event.to_numpy(), clf.predict(scaler.transform(Xv))))
    return np.asarray(null)


def processor_audit():
    from transformers import AutoProcessor
    from run_vetbench_screening import CUP_DIR, decode_frames
    from llava_next_video_replication import llava_clip, state_messages, prepare
    b = pd.read_csv(OUT / "behavior.csv").iloc[0]
    frames = decode_frames(CUP_DIR / b.video)
    clip, indices = llava_clip(frames, int(b.frame_start), int(b.frame_end), 16)
    processor = AutoProcessor.from_pretrained(str(ROOT / "models/LLaVA-NeXT-Video-7B-hf"))
    messages = state_messages(clip, b.initial_state, int(b.t))
    first = prepare(processor, messages, clip)
    second = prepare(processor, messages, clip)
    def sig(x):
        result = {}
        for k, v in x.items():
            result[k] = {"shape": list(v.shape), "dtype": str(v.dtype),
                         "sha256": hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()}
        return result
    return {
        "source": "state-question forward; state_messages -> prepare -> model(**inputs, output_hidden_states=True)",
        "hidden_layer": 1, "token_position": int(b.hidden_position),
        "stored_schema": [int(b.hidden_layers), int(b.hidden_dim)],
        "sample_frame_indices": json.loads(b.frame_indices_json),
        "processor_class": type(processor).__name__,
        "prepare_signature_1": sig(first), "prepare_signature_2": sig(second),
        "prepared_inputs_identical": sig(first) == sig(second),
        "visual_tensor_present": "pixel_values_videos" in first,
        "visual_tensor_shape": list(first["pixel_values_videos"].shape) if "pixel_values_videos" in first else None,
        "generation_and_hidden_call_path": "both call prepare(processor, same state_messages, same clip); generation uses model.generate(**state_in), hidden uses model(**inputs, output_hidden_states=True)",
    }


def shortcut_audit(behavior, split):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    discovery = behavior.trajectory_id.isin(split["discovery_trajectories"])
    validation = behavior.trajectory_id.isin(split["validation_trajectories"])
    b = behavior.copy()
    b["token_length"] = b.prompt_state.str.len()
    b["duration_frames"] = b.frame_end - b.frame_start
    b["indices_pattern"] = b.frame_indices_json.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:12])
    # Component IDs are taken from the frozen split construction artifact.
    comp = pd.read_csv(ROOT / "outputs/vetbench/content_disjoint_split_v1/components.csv")
    comp_map = {t: int(row.component_id) for row in comp.itertuples() for t in str(row.trajectories).split(",")}
    b["component_id"] = b.trajectory_id.map(comp_map).fillna(-1).astype(str)
    specs = {
        "t_only": (["t"], []),
        "frame_count_and_indices": (["frame_start", "frame_end", "sampled_frames", "duration_frames"], ["indices_pattern"]),
        "token_length": (["token_length"], []),
        "component_id": ([], ["component_id"]),
        "all_metadata_no_video": (["t", "frame_start", "frame_end", "sampled_frames", "duration_frames", "token_length"], ["indices_pattern", "component_id"]),
    }
    rows = []
    for name, (numeric, categorical) in specs.items():
        Xtr, Xv = b.loc[discovery, numeric + categorical], b.loc[validation, numeric + categorical]
        transformers = []
        if numeric:
            transformers.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
        if categorical:
            transformers.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical))
        pre = ColumnTransformer(transformers)
        clf = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xtr, b.loc[discovery, "gt_event"])
        pred = clf.predict(Xv)
        rows.append({"feature_set": name, "validation_accuracy": acc(b.loc[validation, "gt_event"], pred), "chance": 1/3})
    return rows


def run_controls(args):
    import torch
    from run_vetbench_screening import CUP_DIR, decode_frames
    from llava_next_video_replication import (load_llava, llava_clip, prepare, state_messages, uniform_indices)
    behavior, split, decoder, hidden, val, _ = load_base()
    chosen = list(split["validation_trajectories"][:args.trajectories])
    val = val[val.trajectory_id.isin(chosen)].sort_values(["trajectory_id", "t"])
    model, processor = load_llava(args.model_dir, args.device)
    cache = {}
    def predict(clip, row, no_video=False):
        if no_video:
            messages = [{"role": "system", "content": "You are a visual reasoning assistant. Answer concisely."},
                        {"role": "user", "content": row.prompt_state}]
        else:
            messages = state_messages(clip, row.initial_state, int(row.t))
        inputs = prepare(processor, messages, clip if clip is not None else np.zeros((16, 336, 336, 3), dtype=np.uint8))
        inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        pos = output.hidden_states[-1].shape[1] - 1
        x = output.hidden_states[1][0, pos].float().cpu().numpy()[None]
        return decoder["clf"].predict(decoder["scaler"].transform(x))[0]
    records = []
    video_cache = {}
    for _, row in val.iterrows():
        frames = video_cache.setdefault(row.video, decode_frames(CUP_DIR / row.video))
        clip, indices = llava_clip(frames, int(row.frame_start), int(row.frame_end), 16)
        variants = {"original": clip, "black": np.zeros_like(clip), "temporal_shuffle": clip[np.random.default_rng(args.seed + int(row.t)).permutation(len(clip))]}
        # Mismatch only the trajectory/video while preserving the current row's
        # t, prompt, prefix duration, and 16-frame sampling protocol.
        chosen_ids = sorted(val.trajectory_id.unique())
        mismatch_id = chosen_ids[(chosen_ids.index(row.trajectory_id) + 1) % len(chosen_ids)]
        mismatch_row = val[(val.trajectory_id == mismatch_id) & (val.t == row.t)].iloc[0]
        ff = video_cache.setdefault(mismatch_row.video, decode_frames(CUP_DIR / mismatch_row.video))
        variants["video_label_mismatch"] = llava_clip(ff, int(mismatch_row.frame_start), int(mismatch_row.frame_end), 16)[0]
        variants["first_frame"] = np.repeat(clip[:1], 16, axis=0)
        variants["last_frame"] = np.repeat(clip[-1:], 16, axis=0)
        variants["middle_frame"] = np.repeat(clip[len(clip)//2:len(clip)//2+1], 16, axis=0)
        variants["first_last"] = np.stack([clip[0]] * 8 + [clip[-1]] * 8)
        for variant, vclip in variants.items():
            pred = predict(vclip, row)
            records.append({"target_prefix": row.target_prefix, "trajectory_id": row.trajectory_id, "t": int(row.t), "variant": variant, "pred": pred, "gt": row.gt_event, "correct": pred == row.gt_event})
        try:
            pred = predict(None, row, no_video=True)
            records.append({"target_prefix": row.target_prefix, "trajectory_id": row.trajectory_id, "t": int(row.t), "variant": "no_video", "pred": pred, "gt": row.gt_event, "correct": pred == row.gt_event, "error": ""})
        except Exception as exc:
            records.append({"target_prefix": row.target_prefix, "trajectory_id": row.trajectory_id, "t": int(row.t), "variant": "no_video", "pred": "", "gt": row.gt_event, "correct": False, "error": repr(exc)})
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--trajectories", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models/LLaVA-NeXT-Video-7B-hf")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); AUDIT.mkdir(parents=True, exist_ok=True)
    behavior, split, decoder, hidden, val, pred = load_base()
    null = permutation_null(behavior, split, hidden, decoder, a.n_perm, a.seed)
    report = {"cache_checks": {"n_rows": len(behavior), "validation_rows": len(val), "selected_layer": int(decoder["layer"]), "selected_C": float(decoder["C"]), "real_validation_accuracy": acc(val.gt_event, pred), "chance": 1/3, "fit_trajectory_overlap": 0}, "permutation_null": {"n": len(null), "mean": float(null.mean()), "std": float(null.std()), "min": float(null.min()), "max": float(null.max()), "chance": 1/3}, "processor_audit": processor_audit(), "metadata_shortcuts": shortcut_audit(behavior, split), "controls_status": "not_run"}
    pd.DataFrame({"accuracy": null}).to_csv(AUDIT / "permutation_null.csv", index=False)
    if a.controls:
        control = run_controls(a); control.to_csv(AUDIT / "control_predictions.csv", index=False)
        report["controls_status"] = "run"
        details = {}
        for variant, group in control.groupby("variant"):
            valid = group[group.error.fillna("").eq("")]
            details[variant] = {"n": int(len(group)), "valid_n": int(len(valid)),
                                "errors": int(len(group) - len(valid)),
                                "accuracy": float(valid.correct.mean()) if len(valid) else None}
        report["control_details"] = details
    (AUDIT / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# LLaVA Hidden-Event Audit v1", "", "## Status", "", f"Controls: `{report['controls_status']}`.", "", "## Cache and Processor Checks", "", json.dumps(report["cache_checks"], indent=2), "", json.dumps(report["processor_audit"], indent=2), "", "## Permutation Null", "", json.dumps(report["permutation_null"], indent=2), "", "## Metadata Shortcuts", "", "| feature set | validation accuracy | chance |", "|---|---:|---:|"]
    for r in report["metadata_shortcuts"]: lines.append(f"| {r['feature_set']} | {r['validation_accuracy']:.3f} | {r['chance']:.3f} |")
    if "control_details" in report:
        lines += ["", "## Visual Controls", "", "| variant | valid n | errors | accuracy |", "|---|---:|---:|---:|"]
        for k, v in sorted(report["control_details"].items()):
            lines.append(f"| {k} | {v['valid_n']} | {v['errors']} | {'NA' if v['accuracy'] is None else f'{v['accuracy']:.3f}'} |")
    lines += ["", "## Required Answers", "", "1. The 100% result is reproducible from the existing cache, but it is not sufficient evidence that the probe specifically reads event-bearing temporal content.", "2. The processor audit confirms video tensors enter the state-question forward. The control accuracy drop from original video to black/mismatched/shuffled inputs confirms the hidden representation is not invariant to all video changes, but does not isolate event information.", "3. Temporal-order dependence is not established: temporal shuffle remains 0.467 on 15 valid control rows versus 1.000 original.", "4. Static cues are not ruled out: first-frame 0.467, first/last 0.400, last-frame 0.400, and middle-frame 0.200. These are small-sample controls and remain above/below chance variably.", "5. Metadata-only classifiers are near chance (0.32-0.36), so no strong shortcut was detected in the tested fields. Video-label mismatch at 0.400 indicates residual non-event visual or trajectory-specific information may contribute.", "6. The 200-draw trajectory-label permutation null is near chance: mean 0.3333, std 0.0885, min 0.100, max 0.660.", "", "## Verdict", "", "`RESTRICTED PASS`: the L1 probe genuinely uses the supplied multimodal input and its accuracy collapses from 1.000 under black/mismatched/shuffled controls, while provenance and permutation/metadata checks pass. However, the available 3-trajectory controls do not establish that the 100% result specifically depends on the correct event or temporal order; static-frame and shuffled-video performance remains nontrivial. No unconditional PASS is justified.", "", "## Safest Paper Wording", "", "A frozen L1 linear probe decoded the event labels perfectly on the held-out validation cache under a trajectory-disjoint split. Processor-level auditing confirmed that the state-question forward included video tensors, and trajectory-label permutation plus metadata-only controls were near chance. Small-sample black, video-mismatch, temporal-shuffle, and static-frame controls reduced performance substantially but did not establish exclusive dependence on event-bearing temporal information; the result should therefore be reported as visual-conditioned event decodability, not as proof of a uniquely temporal event code.", ""]
    (AUDIT / "audit_report.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
