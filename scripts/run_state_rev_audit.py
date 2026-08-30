"""StateRev-VL behavioral audit on VET-Bench cup (shell game) videos.

Three experiments, all 50 cup videos (5 swaps each):

  A. one_swap_audit     per (video, swap k): isolated single-swap clip prediction
                        vs. full-sequence video final-state prediction.
                        Statistics include P(video_correct | isolated_swap_correct)
                        and a per-swap-type confusion matrix.
  B. prefix_state       per (video, t=0..5): state query on the prefix ending right
                        after swap t completes.  Per-step accuracy, first failure
                        step, and whether wrong answers equal S0 or S_{t-1}.
  C. in_context_event   per (video, t=1..5): "which two positions did the LAST
                        swap exchange" asked inside the full prefix context
                        (no cropped swap clip), joined with the same-step state
                        prediction to flag event-correct/state-wrong samples.

Incremental extension of run_vetbench_screening.py: reuses its ground-truth
derivation, frame-window definitions, video decoding/sampling, strict
Left/Middle/Right parsers, and the vLLM batch runner.  Video sampling keeps the
current strategy (all frames of the window are passed to the processor, which
resamples at --sample-fps using the fps=30 metadata; no fixed 32-frame grid).

Deterministic inference: greedy decoding (vLLM temperature=0.0 /
transformers do_sample=False), fixed seeds, cudnn deterministic.  Long CoT is
off by default; Qwen3-VL-8B-Instruct's chat template does not support
enable_thinking, so --thinking on fails fast instead of silently ignoring it.

Full-sequence window: prefix_frame_range(5) = [0, 357) covers the initial
reveal and all five swaps.  Frames 357..371 are a static post-shuffle state
(inspected visually; these videos contain no final reveal), so [0, 357) leaks
no answer and matches the previous screening round's n=5 window.

Run from the project root, physical GPU 3 only:
  CUDA_VISIBLE_DEVICES=3 python scripts/run_state_rev_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from run_behavior_screening import run_inference as transformers_inference
from run_vetbench_screening import (
    CUP_DIR,
    CUP_META,
    OUT_DIR,
    POSITION_NAMES,
    SYSTEM_PROMPT,
    SWAP_OPTION_TEXT,
    SWAP_PAIRS,
    VLLMRunner,
    _frame_cache,
    derive_ground_truth,
    load_metadata,
    parse_single_swap_option,
    parse_tracking_option,
    prefix_frame_range,
    sample_clip,
    sampled_frame_count,
    single_swap_messages,
    swap_frame_range,
    video_processor_kwargs,
)


AUDIT_OUT_DIR = OUT_DIR / "audit"

SWAP_TYPE_ORDER = [SWAP_OPTION_TEXT[p] for p in SWAP_PAIRS]


# --------------------------------------------------------------------------
# Prompts (new for this audit; the isolated single-swap prompt is reused as-is)
# --------------------------------------------------------------------------

def state_question_text(initial_name: str, n_swaps: int) -> str:
    """Ask for the CURRENT POSITION of the cup containing the ball.

    The phrasing pins "position" (Left/Middle/Right place on the table), not
    cup identity, to avoid identity/position ambiguity.
    """
    if n_swaps == 0:
        swap_clause = "No swap has happened yet."
    else:
        swap_clause = f"{n_swaps} swap(s) of the cups have happened in this video."
    return (
        "Three identical cups are at the fixed positions Left, Middle and Right, "
        f"with a ball under one of them. The ball starts under the cup that is at the "
        f"{initial_name} position. {swap_clause} The ball is under the cup that is "
        "currently at one of the three positions. Which position is the cup that "
        "currently contains the ball at? "
        '(A) Left (B) Middle (C) Right. Answer with the option text, e.g. "Left".'
    )


def event_question_text(n_swaps: int) -> str:
    """Ask which two positions the last (just-finished) swap exchanged, in context."""
    return (
        "Three identical cups are at the fixed positions Left, Middle and Right. "
        f"{n_swaps} swap(s) have happened in this video. Focus on the swap that just "
        "happened, i.e. the last (most recent) swap shown in this video. "
        "Which two positions were swapped in that last swap? "
        '(A) Left and Middle (B) Middle and Right (C) Left and Right. '
        'Answer with the option text, e.g. "Left and Right".'
    )


def state_messages(clip: np.ndarray, initial_name: str, n_swaps: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": clip},
            {"type": "text", "text": state_question_text(initial_name, n_swaps)},
        ]},
    ]


def event_messages(clip: np.ndarray, n_swaps: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": clip},
            {"type": "text", "text": event_question_text(n_swaps)},
        ]},
    ]


# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------

def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_thinking_support(processor, thinking: bool) -> None:
    """Fail fast if thinking was requested but the template ignores it."""
    if not thinking:
        return
    probe = [{"role": "user", "content": "Answer 1+1."}]
    off = processor.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    on = processor.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    if on == off:
        raise SystemExit(
            "--thinking on was requested, but this model's chat template produces the same "
            "prompt for enable_thinking=True and False (Qwen3-VL-8B-Instruct does not support "
            "thinking mode). Run with --thinking off."
        )


def load_transformers_vl_model(model_dir: Path):
    """Transformers fallback loader for Qwen3-VL (the shared loader in
    run_behavior_screening targets the Qwen3.5 model class instead)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    print("Loading model (transformers backend)...")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, processor


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def confusion_matrix(rows: list[dict], gt_key: str, pred_key: str) -> dict[str, Any]:
    """Confusion matrix over swap types: rows = GT type, cols = predicted type."""
    cols = SWAP_TYPE_ORDER + ["unparsed"]
    matrix = {g: {c: 0 for c in cols} for g in SWAP_TYPE_ORDER}
    for r in rows:
        matrix[r[gt_key]][r[pred_key] or "unparsed"] += 1
    return {"gt_rows": SWAP_TYPE_ORDER, "pred_cols": cols, "matrix": matrix}


def by_swap_type(rows: list[dict], gt_key: str, pred_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for text in SWAP_TYPE_ORDER:
        subset = [r for r in rows if r[gt_key] == text]
        out[text] = {
            "n": len(subset),
            "accuracy": (sum(r[pred_key] == text for r in subset) / len(subset)) if subset else None,
            "prediction_distribution": dict(Counter(r[pred_key] or "unparsed" for r in subset)),
        }
    return out


def accuracy(rows: list[dict], key: str) -> float | None:
    return (sum(r[key] for r in rows) / len(rows)) if rows else None


def build_one_swap_summary(rows: list[dict]) -> dict[str, Any]:
    by_video: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by_video.setdefault(r["video"], {"iso_total": 0, "iso_correct": 0, "video_correct": r["video_correct"]})
        d["iso_total"] += 1
        d["iso_correct"] += int(r["isolated_swap_correct"])

    # NOTE: the row-level P(video_correct | isolated_swap_correct) used to be computed
    # here (250 rows). It is INVALID and has been removed: the full-sequence video
    # prediction is one result per video, repeated across its 5 isolated-swap rows, so
    # the rows are not independent samples of the conditional. Use the video-level
    # statistic below instead. See build_behavior_audit.py / behavior_audit_summary.json.

    all_iso_videos = {v for v, d in by_video.items() if d["iso_total"] == 5 and d["iso_correct"] == 5}
    all_iso_video_ok = {v for v in all_iso_videos if by_video[v]["video_correct"]}

    by_swap_index: dict[str, dict[str, Any]] = {}
    for k in range(1, 6):
        subset = [r for r in rows if r["swap_index"] == k]
        by_swap_index[str(k)] = {
            "n": len(subset),
            "isolated_accuracy": accuracy(subset, "isolated_swap_correct"),
            "video_accuracy": accuracy(subset, "video_correct"),
        }

    return {
        "n_rows": len(rows),
        "n_videos": len(by_video),
        "isolated_swap_accuracy": accuracy(rows, "isolated_swap_correct"),
        "video_accuracy": accuracy(rows, "video_correct"),
        "p_video_correct_given_isolated_correct": {
            "status": "DEPRECATED_INVALID_STATISTIC",
            "reason": ("row-level conditional removed: the full-sequence video prediction is "
                       "one result per video repeated across its 5 isolated-swap rows, so the "
                       "250 rows are not independent samples of the conditional"),
            "previously_reported_value": 0.39805825242718446,
        },
        "p_video_correct_given_all_isolated_correct": {
            "definition": "video-level: among videos whose 5 isolated swap "
                          "predictions are all correct, fraction with video_correct=true",
            "n_videos_all_isolated_correct": len(all_iso_videos),
            "n_video_correct_among_them": len(all_iso_video_ok),
            "prob": (len(all_iso_video_ok) / len(all_iso_videos)) if all_iso_videos else None,
        },
        "by_swap_index": by_swap_index,
        "by_swap_type": by_swap_type(rows, "gt_swap", "isolated_swap_prediction"),
        "confusion_matrix": confusion_matrix(rows, "gt_swap", "isolated_swap_prediction"),
        "unparsed": {
            "isolated_swap": sum(r["isolated_swap_prediction"] is None for r in rows),
            "video": sum(r["video_prediction"] is None for r in rows),
        },
    }


def build_prefix_state_summary(state_rows: list[dict], trajectories: list[dict]) -> dict[str, Any]:
    per_step: dict[str, dict[str, Any]] = {}
    for t in range(0, 6):
        subset = [r for r in state_rows if r["t"] == t]
        per_step[str(t)] = {
            "n": len(subset),
            "accuracy": accuracy(subset, "state_correct"),
            "unparsed": sum(r["state_prediction"] is None for r in subset),
        }

    wrong = [r for r in state_rows if not r["state_correct"]]
    wrong_s0 = [r for r in wrong if r["equals_s0"]]
    wrong_prev = [r for r in wrong if r["equals_prev_state"]]
    per_step_wrong: dict[str, dict[str, Any]] = {}
    for t in range(0, 6):
        subset = [r for r in wrong if r["t"] == t]
        per_step_wrong[str(t)] = {
            "n_wrong": len(subset),
            "equals_s0": sum(1 for r in subset if r["equals_s0"]),
            "equals_prev_state": sum(1 for r in subset if r["equals_prev_state"]),
        }

    first_failure = Counter(("none" if tr["first_failure_step"] is None else str(tr["first_failure_step"]))
                            for tr in trajectories)

    return {
        "n_rows": len(state_rows),
        "n_trajectories": len(trajectories),
        "overall_accuracy": accuracy(state_rows, "state_correct"),
        "per_step_accuracy": per_step,
        "trajectories_all_steps_correct": sum(tr["first_failure_step"] is None for tr in trajectories),
        "first_failure_step_distribution": dict(sorted(first_failure.items(),
                                                        key=lambda kv: (kv[0] == "none", kv[0]))),
        "wrong_answer_analysis": {
            "n_wrong": len(wrong),
            "equals_s0": {
                "n": len(wrong_s0),
                "rate_of_wrong": (len(wrong_s0) / len(wrong)) if wrong else None,
            },
            "equals_prev_state": {
                "n": len(wrong_prev),
                "rate_of_wrong": (len(wrong_prev) / len(wrong)) if wrong else None,
                "note": "S_{t-1} does not exist for t=0; those rows are excluded from the denominator-free count",
            },
            "per_step": per_step_wrong,
        },
        "unparsed": sum(r["state_prediction"] is None for r in state_rows),
    }


def build_event_summary(event_rows: list[dict]) -> dict[str, Any]:
    per_step: dict[str, dict[str, Any]] = {}
    for t in range(1, 6):
        subset = [r for r in event_rows if r["t"] == t]
        per_step[str(t)] = {
            "n": len(subset),
            "accuracy": accuracy(subset, "event_correct"),
            "unparsed": sum(r["event_prediction"] is None for r in subset),
        }

    joint = Counter()
    flagged: list[str] = []
    for r in event_rows:
        key = ("correct" if r["event_correct"] else "wrong") + "_event_" + \
              ("correct" if r["state_correct"] else "wrong") + "_state"
        joint[key] += 1
        if r["event_correct_state_wrong"]:
            flagged.append(r["sample_id"])
    n = len(event_rows)
    event_correct = sum(r["event_correct"] for r in event_rows)

    return {
        "n_rows": n,
        "overall_accuracy": accuracy(event_rows, "event_correct"),
        "per_step_accuracy": per_step,
        "by_swap_type": by_swap_type(event_rows, "event_gt", "event_prediction"),
        "confusion_matrix": confusion_matrix(event_rows, "event_gt", "event_prediction"),
        "joint_with_state_prediction": {
            "n_rows": n,
            "event_correct_state_correct": joint["correct_event_correct_state"],
            "event_correct_state_wrong": {
                "n": joint["correct_event_wrong_state"],
                "rate_of_event_correct": (joint["correct_event_wrong_state"] / event_correct)
                if event_correct else None,
                "sample_ids": flagged,
            },
            "event_wrong_state_correct": joint["wrong_event_correct_state"],
            "both_wrong": joint["wrong_event_wrong_state"],
        },
        "unparsed": sum(r["event_prediction"] is None for r in event_rows),
    }


# --------------------------------------------------------------------------
# Per-video job planning
# --------------------------------------------------------------------------

def plan_jobs(video_path: Path, gt: dict, initial_name: str, sample_fps: float) -> list[dict]:
    """All 15 inference calls for one video (5 isolated swaps + 6 prefix states + 5 events).

    The t=5 state call (full-sequence window [0, 357)) is shared between the
    1-swap audit's video prediction and the prefix-state experiment.
    """
    jobs: list[dict] = []
    for k in range(1, 6):
        start, end = swap_frame_range(k)
        clip = sample_clip(video_path, start, end)
        jobs.append({
            "job_id": ("swap", k),
            "clip": clip,
            "messages": single_swap_messages(clip, k),
            "vk": video_processor_kwargs(clip, sample_fps),
            "frame_range": [start, end],
            "clip_frames": len(clip),
            "sampled_frames": sampled_frame_count(clip, sample_fps),
        })
    for t in range(0, 6):
        start, end = prefix_frame_range(t)
        clip = sample_clip(video_path, start, end)
        jobs.append({
            "job_id": ("state", t),
            "clip": clip,
            "messages": state_messages(clip, initial_name, t),
            "vk": video_processor_kwargs(clip, sample_fps),
            "frame_range": [start, end],
            "clip_frames": len(clip),
            "sampled_frames": sampled_frame_count(clip, sample_fps),
        })
    for t in range(1, 6):
        start, end = prefix_frame_range(t)
        clip = sample_clip(video_path, start, end)
        jobs.append({
            "job_id": ("event", t),
            "clip": clip,
            "messages": event_messages(clip, t),
            "vk": video_processor_kwargs(clip, sample_fps),
            "frame_range": [start, end],
            "clip_frames": len(clip),
            "sampled_frames": sampled_frame_count(clip, sample_fps),
        })
    return jobs


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([cell(row.get(name)) for name in fieldnames])


ONE_SWAP_CSV_COLUMNS = [
    "video", "sample_id", "swap_index", "initial_state",
    "gt_swap", "gt_swap_positions",
    "isolated_swap_raw", "isolated_swap_prediction", "isolated_swap_correct",
    "gt_final_state", "video_raw", "video_prediction", "video_correct",
    "swap_frame_range", "swap_clip_frames", "swap_sampled_frames",
    "video_sampled_frames", "sample_fps",
]

STATE_CSV_COLUMNS = [
    "video", "sample_id", "t", "gt_state",
    "state_raw", "state_prediction", "state_correct",
    "equals_s0", "equals_prev_state",
    "frame_range", "clip_frames", "sampled_frames", "sample_fps",
]

EVENT_CSV_COLUMNS = [
    "video", "sample_id", "t",
    "event_gt", "event_gt_positions",
    "event_raw", "event_prediction", "event_correct",
    "state_prediction", "state_correct", "event_correct_state_wrong",
    "frame_range", "clip_frames", "sampled_frames", "sample_fps",
]


def trajectory_columns() -> list[str]:
    cols = ["video", "sample_id", "initial_state", "swap_sequence"]
    for t in range(0, 6):
        cols += [f"S{t}_gt", f"S{t}_pred", f"S{t}_correct"]
    cols += ["n_wrong", "first_failure_step", "wrong_equals_s0", "wrong_equals_prev"]
    for t in range(1, 6):
        cols += [f"E{t}_gt", f"E{t}_pred", f"E{t}_correct"]
    cols += ["n_event_wrong", "event_correct_state_wrong_steps"]
    return cols


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral audit on VET-Bench cup videos: 1-swap audit, prefix-state "
                    "tracking, and in-context event recognition."
    )
    parser.add_argument("--meta", type=Path, default=CUP_META)
    parser.add_argument("--video-dir", type=Path, default=CUP_DIR)
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Number of videos to process (0 = all 50).")
    parser.add_argument("--sample-fps", type=float, default=8.0,
                        help="Processor frame-sampling rate (current strategy, unchanged).")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Budget for structured short answers (CoT is off).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", choices=["on", "off"], default="off",
                        help="Long CoT; off by default. Qwen3-VL-8B-Instruct does not "
                             "support thinking mode, so 'on' exits with an error.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=AUDIT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print the prompt plan without loading the model.")
    args = parser.parse_args()

    entries = load_metadata()
    if args.max_videos:
        entries = entries[: args.max_videos]
    gt_by_video = {e["video"]: derive_ground_truth(e) for e in entries}
    n_swaps_each = {len(gt["swaps"]) for gt in gt_by_video.values()}
    if n_swaps_each != {5}:
        raise SystemExit(f"Expected 5 swaps per video, found {n_swaps_each}.")
    full_window = prefix_frame_range(5)
    print(f"Loaded {len(entries)} cup videos; GT derived and validated. "
          f"Full-sequence window: {list(full_window)}")

    if args.dry_run:
        for entry in entries[:2]:
            gt = gt_by_video[entry["video"]]
            video_path = args.video_dir / entry["video"]
            jobs = plan_jobs(video_path, gt, POSITION_NAMES[gt["initial_pos"]], args.sample_fps)
            for job in jobs:
                kind, idx = job["job_id"]
                text = next(c["text"] for c in job["messages"][1]["content"] if c["type"] == "text")
                print(f"\n--- {kind} t={idx} {entry['video']} frames={job['frame_range']} "
                      f"clip={job['clip_frames']} sampled={job['sampled_frames']}")
                print(f"[system] {SYSTEM_PROMPT}")
                print(f"[user] {text}")
        _frame_cache.clear()
        print(f"\nDry run OK: {len(entries) * 15} inference calls "
              f"({len(entries)} videos x 15 jobs each) would be generated.")
        return

    setup_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "vllm":
        runner = VLLMRunner(args.model_dir, args.parallel, args.gpu_memory_utilization)
        check_thinking_support(runner.processor, args.thinking == "on")
        model = runner
    else:
        model, processor = load_transformers_vl_model(args.model_dir)
        check_thinking_support(processor, args.thinking == "on")
    thinking = args.thinking == "on"

    one_swap_path = args.out_dir / "one_swap_audit.jsonl"
    state_path = args.out_dir / "prefix_state.jsonl"
    event_path = args.out_dir / "in_context_event.jsonl"
    traj_path = args.out_dir / "trajectories.jsonl"
    one_swap_file = open(one_swap_path, "w", encoding="utf-8")
    state_file = open(state_path, "w", encoding="utf-8")
    event_file = open(event_path, "w", encoding="utf-8")
    traj_file = open(traj_path, "w", encoding="utf-8")

    one_swap_rows: list[dict] = []
    state_rows: list[dict] = []
    event_rows: list[dict] = []
    trajectories: list[dict] = []
    t_start = time.time()

    try:
        for video_idx, entry in enumerate(entries, start=1):
            video_name = entry["video"]
            video_path = args.video_dir / video_name
            stem = video_name.rsplit(".", 1)[0]
            gt = gt_by_video[video_name]
            initial_name = POSITION_NAMES[gt["initial_pos"]]
            final_name = POSITION_NAMES[gt["final_pos"]]
            states = [POSITION_NAMES[s] for s in gt["states"]]

            jobs = plan_jobs(video_path, gt, initial_name, args.sample_fps)
            if isinstance(model, VLLMRunner):
                raws = model.infer_batch(
                    [j["messages"] for j in jobs], args.max_new_tokens, thinking,
                    [j["vk"] for j in jobs])
            else:
                raws = [
                    transformers_inference(model, processor, j["messages"], args.device,
                                           args.max_new_tokens, thinking, j["vk"])
                    for j in jobs
                ]
            raw_by_id = {j["job_id"]: raw for j, raw in zip(jobs, raws)}
            job_by_id = {j["job_id"]: j for j in jobs}

            # ---- A. one-swap audit rows (isolated swap k + shared full-sequence state) ----
            video_raw = raw_by_id[("state", 5)]
            video_prediction = parse_tracking_option(video_raw)
            for k in range(1, 6):
                swap_raw = raw_by_id[("swap", k)]
                gt_swap = gt["swaps"][k - 1]
                gt_swap_text = SWAP_OPTION_TEXT[gt_swap]
                iso_pred = parse_single_swap_option(swap_raw)
                job = job_by_id[("swap", k)]
                one_swap_rows.append({
                    "video": video_name,
                    "sample_id": f"{stem}_swap{k}",
                    "swap_index": k,
                    "initial_state": initial_name,
                    "gt_swap": gt_swap_text,
                    "gt_swap_positions": list(gt_swap),
                    "isolated_swap_raw": swap_raw,
                    "isolated_swap_prediction": iso_pred,
                    "isolated_swap_correct": iso_pred == gt_swap_text,
                    "gt_final_state": final_name,
                    "video_raw": video_raw,
                    "video_prediction": video_prediction,
                    "video_correct": video_prediction == final_name,
                    "swap_frame_range": job["frame_range"],
                    "swap_clip_frames": job["clip_frames"],
                    "swap_sampled_frames": job["sampled_frames"],
                    "video_sampled_frames": job_by_id[("state", 5)]["sampled_frames"],
                    "sample_fps": args.sample_fps,
                })
                one_swap_file.write(json.dumps(one_swap_rows[-1], ensure_ascii=False) + "\n")
                print(f"  [A] {video_name} swap{k}: gt={gt_swap_text} "
                      f"iso={iso_pred} ok={one_swap_rows[-1]['isolated_swap_correct']} | "
                      f"video final: gt={final_name} pred={video_prediction} "
                      f"ok={video_prediction == final_name}")

            # ---- B. prefix-state rows (t=0..5) ----
            state_preds: list[str | None] = []
            for t in range(0, 6):
                state_raw = raw_by_id[("state", t)]
                state_pred = parse_tracking_option(state_raw)
                state_preds.append(state_pred)
                state_correct = state_pred == states[t]
                wrong = not state_correct
                row = {
                    "video": video_name,
                    "sample_id": f"{stem}_state_t{t}",
                    "t": t,
                    "gt_state": states[t],
                    "state_raw": state_raw,
                    "state_prediction": state_pred,
                    "state_correct": state_correct,
                    "equals_s0": (state_pred == states[0]) if wrong else None,
                    "equals_prev_state": (t >= 1 and state_pred == states[t - 1]) if wrong else None,
                    "frame_range": job_by_id[("state", t)]["frame_range"],
                    "clip_frames": job_by_id[("state", t)]["clip_frames"],
                    "sampled_frames": job_by_id[("state", t)]["sampled_frames"],
                    "sample_fps": args.sample_fps,
                }
                state_rows.append(row)
                state_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  [B] {video_name} t={t}: gt={states[t]} pred={state_pred} ok={state_correct}")

            # ---- C. in-context event rows (t=1..5), joined with same-step state ----
            event_preds: list[str | None] = []
            for t in range(1, 6):
                event_raw = raw_by_id[("event", t)]
                gt_pair = gt["swaps"][t - 1]
                gt_pair_text = SWAP_OPTION_TEXT[gt_pair]
                event_pred = parse_single_swap_option(event_raw)
                event_preds.append(event_pred)
                state_pred = state_preds[t]
                event_correct = event_pred == gt_pair_text
                state_correct = state_pred == states[t]
                flagged = event_correct and not state_correct
                row = {
                    "video": video_name,
                    "sample_id": f"{stem}_event_t{t}",
                    "t": t,
                    "event_gt": gt_pair_text,
                    "event_gt_positions": list(gt_pair),
                    "event_raw": event_raw,
                    "event_prediction": event_pred,
                    "event_correct": event_correct,
                    "state_prediction": state_pred,
                    "state_correct": state_correct,
                    "event_correct_state_wrong": flagged,
                    "frame_range": job_by_id[("event", t)]["frame_range"],
                    "clip_frames": job_by_id[("event", t)]["clip_frames"],
                    "sampled_frames": job_by_id[("event", t)]["sampled_frames"],
                    "sample_fps": args.sample_fps,
                }
                event_rows.append(row)
                event_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  [C] {video_name} t={t}: gt={gt_pair_text} pred={event_pred} "
                      f"ok={event_correct} | state ok={state_correct}"
                      + ("  ** event-right/state-wrong **" if flagged else ""))

            # ---- wide per-trajectory record ----
            first_failure = next((t for t in range(6) if state_preds[t] != states[t]), None)
            wrong_steps = [t for t in range(6) if state_preds[t] != states[t]]
            flagged_steps = [t for t in range(1, 6)
                             if event_preds[t - 1] == SWAP_OPTION_TEXT[gt["swaps"][t - 1]]
                             and state_preds[t] != states[t]]
            traj: dict[str, Any] = {
                "video": video_name,
                "sample_id": stem,
                "initial_state": initial_name,
                "swap_sequence": [list(s) for s in gt["swaps"]],
            }
            for t in range(0, 6):
                traj[f"S{t}_gt"] = states[t]
                traj[f"S{t}_pred"] = state_preds[t]
                traj[f"S{t}_correct"] = state_preds[t] == states[t]
            traj["n_wrong"] = len(wrong_steps)
            traj["first_failure_step"] = first_failure
            traj["wrong_equals_s0"] = sum(
                1 for t in wrong_steps if state_preds[t] == states[0])
            traj["wrong_equals_prev"] = sum(
                1 for t in wrong_steps if t >= 1 and state_preds[t] == states[t - 1])
            for t in range(1, 6):
                traj[f"E{t}_gt"] = SWAP_OPTION_TEXT[gt["swaps"][t - 1]]
                traj[f"E{t}_pred"] = event_preds[t - 1]
                traj[f"E{t}_correct"] = event_preds[t - 1] == SWAP_OPTION_TEXT[gt["swaps"][t - 1]]
            traj["n_event_wrong"] = sum(
                1 for t in range(1, 6)
                if event_preds[t - 1] != SWAP_OPTION_TEXT[gt["swaps"][t - 1]])
            traj["event_correct_state_wrong_steps"] = flagged_steps
            trajectories.append(traj)
            traj_file.write(json.dumps(traj, ensure_ascii=False) + "\n")

            print(f"[video {video_idx}/{len(entries)}] {video_name} done")
            del jobs, raws
            _frame_cache.pop(str(video_path), None)
    finally:
        one_swap_file.close()
        state_file.close()
        event_file.close()
        traj_file.close()

    elapsed = time.time() - t_start

    # ---- CSV exports ----
    write_csv(args.out_dir / "one_swap_audit.csv", one_swap_rows, ONE_SWAP_CSV_COLUMNS)
    write_csv(args.out_dir / "prefix_state.csv", state_rows, STATE_CSV_COLUMNS)
    write_csv(args.out_dir / "in_context_event.csv", event_rows, EVENT_CSV_COLUMNS)
    write_csv(args.out_dir / "trajectories.csv", trajectories, trajectory_columns())

    # ---- summary + config ----
    summary = {
        "one_swap_audit": build_one_swap_summary(one_swap_rows),
        "prefix_state": build_prefix_state_summary(state_rows, trajectories),
        "in_context_event": build_event_summary(event_rows),
    }
    config = {
        "script": "scripts/run_state_rev_audit.py",
        "model_dir": str(args.model_dir),
        "backend": args.backend,
        "seed": args.seed,
        "thinking": args.thinking,
        "sample_fps": args.sample_fps,
        "max_new_tokens": args.max_new_tokens,
        "parallel": args.parallel if args.backend == "vllm" else None,
        "gpu_memory_utilization": args.gpu_memory_utilization if args.backend == "vllm" else None,
        "n_videos": len(entries),
        "video_meta": str(args.meta),
        "full_sequence_window": list(full_window),
        "sampling_strategy": ("all frames of each window are passed to the processor, which "
                              "resamples uniformly at --sample-fps using video_metadata fps=30 "
                              "(current strategy; no fixed frame grid)"),
        "decoding": "greedy (vLLM temperature=0.0 / transformers do_sample=False); cudnn deterministic",
        "shared_call": ("one_swap_audit.video_prediction reuses the prefix_state t=5 call "
                        "(identical full-sequence clip and state prompt)"),
        "prompts": {
            "system": SYSTEM_PROMPT,
            "state_question_t0_example": state_question_text("Left", 0),
            "state_question_t5_example": state_question_text("Left", 5),
            "event_question_t1_example": event_question_text(1),
            "isolated_swap": (
                "The video clip shows one swap in a shell game with three cups "
                "(positions: Left, Middle, Right). "
                "Which two positions were swapped in this clip? "
                '(A) Left and Middle (B) Middle and Right (C) Left and Right. '
                'Answer with the option text, e.g. "Left and Right".'
            ),
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "audit_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults dir: {args.out_dir}")
    print(f"  one_swap_audit.jsonl/.csv  prefix_state.jsonl/.csv  "
          f"in_context_event.jsonl/.csv  trajectories.jsonl/.csv")
    print(f"  audit_summary.json  audit_config.json")


if __name__ == "__main__":
    main()
