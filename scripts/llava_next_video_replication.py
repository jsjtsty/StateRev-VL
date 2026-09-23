#!/usr/bin/env python3
"""LLaVA-NeXT-Video cross-model replication for StateRev-VL.

This is intentionally separate from the Qwen runners.  It reuses the frozen
content-disjoint split and VET-Bench trajectory definitions, but keeps all
LLaVA inputs, hidden states, probes, and recursive predictions in a new output
tree.  No full experiment is started unless ``run`` is explicitly requested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from content_disjoint_split import assert_content_disjoint  # noqa: E402
from run_state_rev_audit import (  # noqa: E402
    SYSTEM_PROMPT, event_messages, event_question_text, state_messages,
    state_question_text,
)
from run_vetbench_screening import (  # noqa: E402
    CUP_DIR, CUP_META, POSITION_NAMES,
    _frame_cache, decode_frames, derive_ground_truth, load_metadata,
    parse_single_swap_option, parse_tracking_option, prefix_frame_range,
)

# Avoid importing a non-existent alias above while retaining one canonical schema.
STATE_CLASSES = ("Left", "Middle", "Right")
EVENT_CLASSES = ("Left and Middle", "Middle and Right", "Left and Right")
STATE_INDEX = {x: i for i, x in enumerate(STATE_CLASSES)}
EVENT_INDEX = {x: i for i, x in enumerate(EVENT_CLASSES)}
OUT = ROOT / "outputs/vetbench/llava_next_video_7b_replication_v1"
SPLIT = ROOT / "outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"
FINGERPRINTS = ROOT / "outputs/vetbench/validity_gate_v1/input_fingerprints.csv"
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_DIR = ROOT / "models/LLaVA-NeXT-Video-7B-hf"
VIDEO_FPS = 30.0
LLAVA_FRAMES = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_check(split_path: Path, fingerprint_path: Path, hidden_path: Path | None = None) -> dict:
    split = json.loads(split_path.read_text())
    assert_content_disjoint(split, fingerprint_path, hidden_path if hidden_path and hidden_path.exists() else None)
    d, v = set(split["discovery_trajectories"]), set(split["validation_trajectories"])
    assert not d & v
    assert len(d) == 30 and len(v) == 20
    return split


def parse_state(text: str) -> str | None:
    return parse_tracking_option(text or "")


def parse_event(text: str) -> str | None:
    return parse_single_swap_option(text or "")


def uniform_indices(start: int, end: int, n: int = LLAVA_FRAMES) -> np.ndarray:
    if end <= start:
        raise ValueError(f"empty frame range [{start}, {end})")
    return np.unique(np.linspace(start, end - 1, min(n, end - start), dtype=np.int64))


def llava_clip(frames: np.ndarray, start: int, end: int, n: int = LLAVA_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    idx = uniform_indices(start, min(end, len(frames)), n)
    return frames[idx], idx


def fingerprint_frames(frames: np.ndarray, indices: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(frames).tobytes())
    return h.hexdigest()


def rows_from_split(split: dict, side: str, max_trajectories: int = 0) -> list[dict]:
    allowed = list(split[f"{side}_trajectories"])
    if max_trajectories:
        allowed = allowed[:max_trajectories]
    entries = {e["video"]: e for e in load_metadata()}
    rows = []
    for tr in allowed:
        video = f"{tr}.mp4"
        gt = derive_ground_truth(entries[video])
        for t in range(1, 6):
            start, end = prefix_frame_range(t)
            rows.append({
                "trajectory_id": tr, "t": t, "video": video,
                "frame_start": start, "frame_end": end,
                "initial_state": POSITION_NAMES[gt["initial_pos"]],
                "gt_prev_state": POSITION_NAMES[gt["states"][t - 1]],
                "gt_state": POSITION_NAMES[gt["states"][t]],
                "gt_event": {tuple(sorted(x)): f"{POSITION_NAMES[x[0]]} and {POSITION_NAMES[x[1]]}" for x in ((1, 2), (2, 3), (1, 3))}[tuple(sorted(gt["swaps"][t - 1]))],
            })
    return rows


def make_messages(kind: str, clip: np.ndarray, row: dict) -> list[dict]:
    if kind == "state":
        return state_messages(clip, row["initial_state"], row["t"])
    return event_messages(clip, row["t"])


def processor_inputs(processor, messages: list[dict], clip: np.ndarray) -> dict:
    # LLaVA-NeXT-Video's processor accepts the actual sampled ndarray through
    # the video content block.  Keeping this in one adapter makes the exact
    # prompt and frame provenance explicit in every shard row.
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )


def load_llava(model_dir: Path, device: str):
    import torch
    from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(model_dir if model_dir.exists() else MODEL_ID))
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        str(model_dir if model_dir.exists() else MODEL_ID), torch_dtype=torch.bfloat16,
        device_map={"": device}, low_cpu_mem_usage=True,
    ).eval()
    return model, processor


def prepare(processor, messages: list[dict], clip: np.ndarray) -> dict:
    # Some Transformers versions need the video supplied separately even when
    # the chat template rendered the <video> token.  The fallback covers both.
    try:
        out = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             return_dict=True, return_tensors="pt")
    except (TypeError, ValueError):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = processor(text=text, videos=clip, return_tensors="pt")
    if "pixel_values_videos" not in out and "video_values" not in out:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = processor(text=text, videos=clip, return_tensors="pt")
    return out


def move_inputs(inputs: dict, device: str) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def candidate_scores(model, processor, messages: list[dict], clip: np.ndarray,
                     candidates: tuple[str, ...], device: str) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    base = move_inputs(prepare(processor, messages, clip), device)
    prompt_len = base["input_ids"].shape[1]
    scores = {}
    for candidate in candidates:
        ids = processor.tokenizer.encode(" " + candidate, add_special_tokens=False, return_tensors="pt").to(device)
        full_ids = torch.cat([base["input_ids"], ids], dim=1)
        full = dict(base); full["input_ids"] = full_ids
        full["attention_mask"] = torch.ones_like(full_ids)
        if "mm_token_type_ids" in base:
            full["mm_token_type_ids"] = torch.cat([base["mm_token_type_ids"], torch.zeros_like(ids)], dim=1)
        with torch.inference_mode():
            out = model(**full, return_dict=True, logits_to_keep=ids.shape[1] + 1)
        lp = F.log_softmax(out.logits[0].float(), dim=-1)
        scores[candidate] = float(sum(lp[j, tok].item() for j, tok in enumerate(ids[0].tolist())) / ids.shape[1])
    candidate_items = list(scores.items())
    vals = np.exp(np.asarray([v for _, v in candidate_items]) - max(scores.values()))
    for (k, _), p in zip(candidate_items, vals / vals.sum()):
        scores[k + "__prob"] = float(p)
    return scores


def forward_hidden(model, processor, messages: list[dict], clip: np.ndarray, device: str) -> tuple[np.ndarray, int, int]:
    import torch
    inputs = move_inputs(prepare(processor, messages, clip), device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, return_dict=True, logits_to_keep=1)
    hs = out.hidden_states
    if not hs:
        raise RuntimeError("LLaVA forward did not return hidden_states")
    position = int(hs[-1].shape[1] - 1)
    arr = np.stack([x[0, position, :].float().cpu().numpy() for x in hs]).astype(np.float32)
    return arr, position, int(hs[-1].shape[-1])


def run_shard(args: argparse.Namespace) -> None:
    import torch
    import transformers
    hidden_path = OUT / "hidden_states.npz"
    split = split_check(args.split, args.fingerprints, hidden_path)
    rows = rows_from_split(split, args.side, args.max_trajectories)
    selected = rows[args.shard_index::args.num_shards]
    out = OUT / "shards" / args.side
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"behavior_shard_{args.shard_index}.csv"
    marker = csv_path.with_suffix(".complete.json")
    npz_path = out / f"hidden_shard_{args.shard_index}.npz"
    if csv_path.exists() or marker.exists() or npz_path.exists():
        raise FileExistsError(f"refusing to overwrite completed/in-progress shard {args.shard_index}")
    model, processor = load_llava(args.model_dir, args.device)
    records, hidden = [], {}
    entries = {e["video"]: e for e in load_metadata()}
    for i, row in enumerate(selected, 1):
        video_path = args.video_dir / row["video"]
        frames = decode_frames(video_path)
        clip, indices = llava_clip(frames, row["frame_start"], row["frame_end"], args.frames)
        fp = fingerprint_frames(clip, indices)
        state_msg = make_messages("state", clip, row)
        event_msg = make_messages("event", clip, row)
        state_scores = candidate_scores(model, processor, state_msg, clip, STATE_CLASSES, args.device)
        event_scores = candidate_scores(model, processor, event_msg, clip, EVENT_CLASSES, args.device)
        state_in = move_inputs(prepare(processor, state_msg, clip), args.device)
        with torch.inference_mode():
            generated_state = model.generate(**state_in, max_new_tokens=args.max_new_tokens, do_sample=False,
                                              pad_token_id=processor.tokenizer.pad_token_id)
        state_raw = processor.batch_decode(generated_state[:, state_in["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        event_in = move_inputs(prepare(processor, event_msg, clip), args.device)
        with torch.inference_mode():
            generated_event = model.generate(**event_in, max_new_tokens=args.max_new_tokens, do_sample=False,
                                             pad_token_id=processor.tokenizer.pad_token_id)
        event_raw = processor.batch_decode(generated_event[:, event_in["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        hs, pos, dim = forward_hidden(model, processor, state_msg, clip, args.device)
        key = f"{row['trajectory_id']}_t{row['t']}"
        hidden[key] = hs
        rec = {**row, "target_prefix": key, "state_raw": state_raw, "state_pred": parse_state(state_raw),
               "event_raw": event_raw, "event_pred": parse_event(event_raw), "state_correct": parse_state(state_raw) == row["gt_state"],
               "event_correct": parse_event(event_raw) == row["gt_event"], "state_scores_json": json.dumps(state_scores),
               "event_scores_json": json.dumps(event_scores), "frame_indices_json": json.dumps(indices.tolist()),
               "sampled_frames": len(indices), "sampling_strategy": "uniform_indices_over_original_prefix_window",
               "sample_fingerprint": fp, "hidden_position": pos, "hidden_layers": hs.shape[0], "hidden_dim": dim,
               "prompt_state": state_question_text(row["initial_state"], row["t"]),
               "prompt_event": event_question_text(row["t"]), "model_id": MODEL_ID,
               "model_class": type(model).__name__, "processor_class": type(processor).__name__,
               "transformers_version": transformers.__version__}
        records.append(rec)
        _frame_cache.pop(str(video_path), None)
        print(f"[{i}/{len(selected)}] {key} frames={len(indices)} hidden={hs.shape}")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **hidden)
    marker.write_text(json.dumps({"schema_version": 1, "side": args.side, "shard_index": args.shard_index,
                                  "num_shards": args.num_shards, "rows": len(records),
                                  "trajectory_ids": sorted({r["trajectory_id"] for r in records}),
                                  "target_prefixes": sorted(hidden), "hidden_shape": list(next(iter(hidden.values())).shape) if hidden else None,
                                  "csv_sha256": sha256(csv_path), "npz_sha256": sha256(npz_path),
                                  "model_id": MODEL_ID, "model_class": type(model).__name__,
                                  "processor_class": type(processor).__name__,
                                  "transformers_version": transformers.__version__,
                                  "frame_sampling": "uniform indices over original prefix window",
                                  "frames_per_clip": args.frames}, indent=2) + "\n")


def merge(args: argparse.Namespace) -> None:
    split = split_check(args.split, args.fingerprints, None)
    root = OUT / "shards"
    frames, hidden = [], {}
    expected = set(f"{t}_t{i}" for t in split["discovery_trajectories"] + split["validation_trajectories"] for i in range(1, 6))
    for side in ("discovery", "validation"):
        for i in range(args.num_shards):
            d = root / side; c = d / f"behavior_shard_{i}.csv"; n = d / f"hidden_shard_{i}.npz"; m = c.with_suffix(".complete.json")
            if not c.exists() or not n.exists() or not m.exists(): raise FileNotFoundError(f"incomplete {side} shard {i}")
            meta = json.loads(m.read_text()); assert sha256(c) == meta["csv_sha256"] and sha256(n) == meta["npz_sha256"]
            frames.append(pd.read_csv(c))
            with np.load(n) as z:
                for k in z.files:
                    if k in hidden: raise AssertionError(f"duplicate hidden key {k}")
                    hidden[k] = z[k]
    behavior = pd.concat(frames, ignore_index=True)
    assert set(behavior.target_prefix) == expected and set(hidden) == expected
    assert not behavior.target_prefix.duplicated().any()
    for side in ("discovery", "validation"):
        ids = set(split[f"{side}_trajectories"]); assert set(behavior.loc[behavior.trajectory_id.isin(ids), "trajectory_id"]) == ids
    d, v = set(split["discovery_trajectories"]), set(split["validation_trajectories"])
    assert not d & v
    seen = {}
    for key, value in hidden.items():
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        trajectory = key.rsplit("_t", 1)[0]
        partition = "discovery" if trajectory in d else "validation"
        for prior_digest, prior_key, prior_partition in seen.get(digest, []):
            if prior_partition != partition:
                raise AssertionError(f"exact hidden tensor overlap across split: {prior_key} / {key}")
        seen.setdefault(digest, []).append((digest, key, partition))
    pixel_fps = {}
    for rec in behavior.to_dict("records"):
        pixel_fps.setdefault(rec["sample_fingerprint"], set()).add(rec["trajectory_id"])
    if any(len(ids & d) and len(ids & v) for ids in pixel_fps.values()):
        raise AssertionError("exact sampled pixel fingerprint overlap")
    OUT.mkdir(parents=True, exist_ok=True)
    behavior.to_csv(OUT / "behavior.csv", index=False)
    np.savez_compressed(OUT / "hidden_states.npz", **hidden)
    (OUT / "merge_summary.json").write_text(json.dumps({"rows": len(behavior), "hidden_shape": list(next(iter(hidden.values())).shape),
        "trajectory_overlap": 0, "exact_pixel_fingerprint_overlap": 0, "exact_hidden_tensor_overlap": 0,
        "expected_prefixes": len(expected)}, indent=2) + "\n")
    print(f"MERGE_PASS rows={len(behavior)} hidden_shape={next(iter(hidden.values())).shape}")


def smoke(args: argparse.Namespace) -> None:
    split = split_check(args.split, args.fingerprints, None)
    rows = rows_from_split(split, "discovery", min(args.max_trajectories or 2, 2))
    assert len(rows) == min(args.max_trajectories or 2, 2) * 5
    for r in rows[:2]:
        assert parse_state("The answer is Left") == "Left"
        assert parse_event("(C) Left and Right") == "Left and Right"
        assert len(uniform_indices(r["frame_start"], r["frame_end"], args.frames)) <= args.frames
    print(json.dumps({"status": "PASS", "trajectories": sorted({r["trajectory_id"] for r in rows}),
                      "rows": len(rows), "frame_strategy": f"uniform {args.frames} frames over original [frame_start, frame_end)",
                      "split": {"discovery": 30, "validation": 20}, "parser": "PASS", "hidden_schema": "deferred until model run"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("smoke", "run", "merge"))
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--video-dir", type=Path, default=CUP_DIR)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--fingerprints", type=Path, default=FINGERPRINTS)
    p.add_argument("--side", choices=("discovery", "validation"), default="discovery")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-trajectories", type=int, default=0)
    p.add_argument("--frames", type=int, default=LLAVA_FRAMES)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.command == "smoke": smoke(args)
    elif args.command == "run": run_shard(args)
    else: merge(args)


if __name__ == "__main__":
    main()
