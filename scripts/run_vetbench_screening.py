from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np

from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from run_behavior_screening import (
    load_model_and_processor as load_transformers_model,
    run_inference as transformers_inference,
    score_candidates,
)


CUP_DIR = Path("dataset/vetbench/cup")
CUP_META = CUP_DIR / "cup.json"
OUT_DIR = DEFAULT_OUTPUT_DIR / "vetbench"

POSITION_NAMES = {1: "Left", 2: "Middle", 3: "Right"}
POSITION_INDEX = {name: idx for idx, name in POSITION_NAMES.items()}
TRACKING_OPTIONS = ("Left", "Middle", "Right")
SWAP_PAIRS = ((1, 2), (2, 3), (1, 3))  # option A / B / C
SWAP_OPTION_TEXT = {
    pair: f"{POSITION_NAMES[pair[0]]} and {POSITION_NAMES[pair[1]]}" for pair in SWAP_PAIRS
}

# Official export timing from the VET-Bench generator (cup.html):
# reveal lift 600ms, hold 800ms, lower 500ms -> swaps start at 1900ms;
# each swap cycle is 2000ms (swaps_per_second=0.5, interval=0).
SHUFFLE_START_MS = 1900
SWAP_CYCLE_MS = 2000
FPS = 30

SYSTEM_PROMPT = (
    "You are a visual reasoning assistant. Answer concisely with the requested "
    "option text only, e.g. \"Left and Right\" or \"Left\". No explanations."
)

_frame_cache: dict[str, np.ndarray] = {}


def load_metadata() -> list[dict]:
    with open(CUP_META, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_ground_truth(entry: dict) -> dict:
    """Derive per-step ground truth from official metadata (validated against final GT).

    - swap k: the two positions (1-based) whose cups differ between the arrangement
      before and after swap k;
    - ball cup: cup label sitting at the official final position in the last arrangement;
    - initial ball position: that cup's position in the initial arrangement ("123");
    - step states: ball position after 0..5 swaps.
    """
    initial = entry["initial"]
    intermediate = entry["intermediate"]
    final_pos = entry["ground_truth"]["cup"]

    swaps: list[tuple[int, int]] = []
    prev = initial
    for nxt in intermediate:
        diff = [i + 1 for i in range(len(prev)) if prev[i] != nxt[i]]
        if len(diff) != 2:
            raise ValueError(f"Unexpected arrangement transition {prev} -> {nxt} in {entry['video']}")
        swaps.append((int(diff[0]), int(diff[1])))
        prev = nxt

    ball_cup = intermediate[-1][final_pos - 1]
    initial_pos = initial.index(ball_cup) + 1

    states = [initial_pos]
    pos = initial_pos
    prev = initial
    for nxt in intermediate:
        diff = [i + 1 for i in range(len(prev)) if prev[i] != nxt[i]]
        if pos in diff:
            pos = diff[0] if pos == diff[1] else diff[1]
        states.append(pos)
        prev = nxt

    if states[-1] != final_pos:
        raise ValueError(f"Derived final state does not match official GT in {entry['video']}")
    return {"swaps": swaps, "initial_pos": initial_pos, "states": states, "final_pos": final_pos}


def swap_frame_range(swap_index: int) -> tuple[int, int]:
    """Frame range [start, end) of the k-th swap (1-based) in the 30fps video."""
    start = SHUFFLE_START_MS / 1000 * FPS + (swap_index - 1) * SWAP_CYCLE_MS / 1000 * FPS
    end = start + SWAP_CYCLE_MS / 1000 * FPS
    return int(start), int(end)


def prefix_frame_range(n_swaps: int) -> tuple[int, int]:
    if n_swaps == 0:
        return 0, int(SHUFFLE_START_MS / 1000 * FPS)
    _, end = swap_frame_range(n_swaps)
    return 0, end


def decode_frames(video_path: Path) -> np.ndarray:
    if str(video_path) not in _frame_cache:
        container = av.open(str(video_path))
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        container.close()
        _frame_cache[str(video_path)] = np.stack(frames)
    return _frame_cache[str(video_path)]


def sample_clip(video_path: Path, start: int, end: int) -> np.ndarray:
    """Return every frame in [start, end) of the video (the processor samples at --sample-fps)."""
    frames = decode_frames(video_path)
    end = min(end, len(frames))
    start = min(max(start, 0), max(end - 1, 0))
    return frames[start:end]


def video_processor_kwargs(clip: np.ndarray, sample_fps: float) -> dict:
    """Tell the processor the clip's true fps/frame count and ask it to sample at ``sample_fps``."""
    return {
        "fps": float(sample_fps),
        "video_metadata": {"total_num_frames": len(clip), "fps": float(FPS)},
    }


def sampled_frame_count(clip: np.ndarray, sample_fps: float) -> int:
    """Number of frames the processor will sample from this clip at ``sample_fps``."""
    return max(1, int(round(len(clip) / FPS * sample_fps)))


def single_swap_messages(clip: np.ndarray, swap_index: int) -> list[dict]:
    """Single-swap tracking diagnostic: given one swap clip, which two positions swapped.

    The cups are visually identical, so identifying the swap already requires
    spatiotemporal entity tracking; this is not a low-level perception check.
    """
    text = (
        "The video clip shows one swap in a shell game with three cups "
        "(positions: Left, Middle, Right). "
        "Which two positions were swapped in this clip? "
        "(A) Left and Middle (B) Middle and Right (C) Left and Right. "
        "Answer with the option text, e.g. \"Left and Right\"."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "video", "video": clip}, {"type": "text", "text": text}]},
    ]


def tracking_messages(
    clip: np.ndarray | None,
    initial_name: str,
    n_swaps: int,
    swap_descriptions: list[str] | None = None,
) -> list[dict]:
    if swap_descriptions is not None:
        parts = [
            f"The ball starts under the cup at the {initial_name} position.",
            f"Exactly {n_swaps} swap(s) occur.",
        ]
        if n_swaps == 0:
            parts.append("No cups move or exchange positions before the question is asked.")
        parts.extend(swap_descriptions)
        parts.append(
            "Which cup contains the ball at the end? "
            "(A) Left (B) Middle (C) Right. Answer with the option text, e.g. \"Left\"."
        )
        text = " ".join(parts)
        content: list[dict] = [{"type": "text", "text": text}]
    else:
        text = (
            f"The ball starts under the cup at the {initial_name} position. The video shows {n_swaps} swap(s) "
            "of the three cups. Which cup contains the ball at the end? "
            "(A) Left (B) Middle (C) Right. Answer with the option text, e.g. \"Left\"."
        )
        content = [{"type": "video", "video": clip}, {"type": "text", "text": text}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _post_think(text: str) -> str:
    """Drop any <think>...</think> block; only the text after it is the answer."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


class VLLMRunner:
    """vLLM multimodal runner. Requests are submitted in batches by ``infer_batch``."""

    def __init__(self, model_dir: Path, max_num_seqs: int, gpu_memory_utilization: float):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=str(model_dir), max_num_seqs=max_num_seqs,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=32768,
                       trust_remote_code=True)
        self.SamplingParams = SamplingParams
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def validate_thinking(self) -> None:
        probe = [{"role": "user", "content": "Answer 1+1."}]
        off = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        on = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if on == off:
            raise RuntimeError(
                "--thinking on was requested, but this model's chat template produces "
                "the same prompt for enable_thinking=True and False. Thinking is not supported/enabled."
            )
        print("Thinking template check passed: on/off prompts differ.")

    def infer_batch(self, messages_batch: list[list[dict]], max_tokens: int, thinking: bool,
                    processor_kwargs_batch: list[dict | None] | None = None) -> list[str]:
        prompts = []
        for i, messages in enumerate(messages_batch):
            kwargs = processor_kwargs_batch[i] if processor_kwargs_batch else None
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking, processor_kwargs=kwargs)
            mm = {}
            for message in messages:
                content = message.get("content", [])
                if isinstance(content, str):
                    continue
                for item in content:
                    if item.get("type") == "video":
                        video = item["video"]
                        # vLLM's Qwen VL video parser requires metadata when
                        # frames are supplied as an ndarray rather than a path.
                        if kwargs and "video_metadata" in kwargs:
                            metadata = dict(kwargs["video_metadata"])
                            metadata.setdefault("fps", float(FPS))
                            metadata.setdefault("total_num_frames", len(video))
                            metadata.setdefault("frames_indices", np.arange(len(video), dtype=np.int64))
                            mm["video"] = (video, metadata)
                        else:
                            mm["video"] = (video, {"fps": float(FPS),
                                                   "total_num_frames": len(video),
                                                   "frames_indices": np.arange(len(video), dtype=np.int64)})
            prompts.append({"prompt": prompt, "multi_modal_data": mm} if mm else prompt)
        params = self.SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

    def infer(self, messages: list[dict], max_tokens: int, thinking: bool,
              processor_kwargs: dict | None = None) -> str:
        return self.infer_batch([messages], max_tokens, thinking, [processor_kwargs])[0]


def parse_single_swap_option(text: str) -> str | None:
    text = _post_think(text or "")
    for pair in SWAP_PAIRS:
        option_text = SWAP_OPTION_TEXT[pair]
        if re.search(rf"\b{re.escape(option_text)}\b", text, re.IGNORECASE):
            return option_text
    words = [w for w in ("Left", "Middle", "Right") if re.search(rf"\b{w}\b", text, re.IGNORECASE)]
    if len(words) == 2:
        pair = tuple(sorted(POSITION_INDEX[w] for w in words))
        if pair in SWAP_OPTION_TEXT:
            return SWAP_OPTION_TEXT[pair]
    letter = re.search(r"\b([ABC])\b", text)
    if letter:
        return SWAP_OPTION_TEXT[SWAP_PAIRS[ord(letter.group(1).upper()) - ord("A")]]
    return None


def parse_tracking_option(text: str) -> str | None:
    text = _post_think(text or "")
    for option in TRACKING_OPTIONS:
        if re.search(rf"\b{option}\b", text, re.IGNORECASE):
            return option
    letter = re.search(r"\b([ABC])\b", text)
    if letter:
        return TRACKING_OPTIONS[ord(letter.group(1).upper()) - ord("A")]
    return None


def thinking_evidence(text: str) -> dict[str, Any]:
    """Record whether thinking mode actually produced a non-empty reasoning block."""
    raw = text or ""
    if "</think>" not in raw:
        return {"think_block_present": False, "think_tokens": 0}
    before = raw.split("</think>", 1)[0]
    before = before.split("<think>", 1)[-1]
    return {"think_block_present": True, "think_tokens": len(before.split())}


def swap_descriptions(swaps: list[tuple[int, int]]) -> list[str]:
    return [
        f"Swap {i + 1}: the cups currently occupying the {POSITION_NAMES[a]} and "
        f"{POSITION_NAMES[b]} positions exchange positions."
        for i, (a, b) in enumerate(swaps)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VET-Bench cup (shell game) screening: single-swap tracking diagnostic and state tracking."
    )
    parser.add_argument("--meta", type=Path, default=CUP_META)
    parser.add_argument("--video-dir", type=Path, default=CUP_DIR)
    parser.add_argument("--max-videos", type=int, default=10,
                        help="Number of videos to process (0 = all 50). Pilot default: 10.")
    parser.add_argument("--tracking-input", choices=["video", "text", "both"], default="both",
                        help="video = visual swap sequence; text = semantically equivalent text-action control.")
    parser.add_argument("--scoring-mode", choices=["forced_choice", "free", "both"], default="forced_choice")
    parser.add_argument("--thinking", choices=["on", "off"], default="off")
    parser.add_argument("--sample-fps", type=float, default=8.0,
                        help="Fixed frame sampling rate for video inputs (frames per second of clip). "
                             "Keeps temporal resolution consistent across different n_swaps.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-think-tokens", type=int, default=32768)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm",
                        help="Inference backend; vLLM batches concurrent multimodal requests.")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Maximum concurrent sequences reserved by vLLM.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = load_metadata()[: args.max_videos] if args.max_videos else load_metadata()
    gt_by_video = {entry["video"]: derive_ground_truth(entry) for entry in entries}
    print(f"Loaded {len(entries)} cup videos (ground truth derived and validated).")

    tracking_modes = ["video", "text"] if args.tracking_input == "both" else [args.tracking_input]
    scoring_modes = []
    if args.scoring_mode in ("forced_choice", "both"):
        scoring_modes.append("forced_choice")
    if args.scoring_mode in ("free", "both"):
        scoring_modes.append("free")
    if args.backend == "vllm" and "forced_choice" in scoring_modes:
        # vLLM generation does not expose the multimodal continuation logits used
        # by the Transformers scorer; keep the experiment runnable and explicit.
        scoring_modes = [m for m in scoring_modes if m != "forced_choice"]
        if "free" not in scoring_modes:
            scoring_modes.append("free")
        print("vLLM backend: forced_choice replaced by free generation (no logits scoring).")

    single_swap_plan = [
        (entry, k)
        for entry in entries
        for k in range(1, len(gt_by_video[entry["video"]]["swaps"]) + 1)
    ]
    tracking_plan = [
        (entry, n_swaps, mode)
        for entry in entries
        for n_swaps in range(len(gt_by_video[entry["video"]]["states"]))
        for mode in tracking_modes
    ]
    print(f"Plans: single_swap_tracking={len(single_swap_plan)} tracking={len(tracking_plan)} "
          f"per condition; scoring modes: {scoring_modes}")

    if args.dry_run:
        for entry, k in single_swap_plan[:2]:
            gt = gt_by_video[entry["video"]]
            start, end = swap_frame_range(k)
            clip = sample_clip(args.video_dir / entry["video"], start, end)
            msgs = single_swap_messages(clip, k)
            print(f"\n--- single_swap_tracking {entry['video']} swap{k} gt={gt['swaps'][k-1]} "
                  f"frames=[{start},{end}) sampled={sampled_frame_count(clip, args.sample_fps)}")
            for message in msgs:
                content = message["content"]
                if isinstance(content, str):
                    print(f"[{message['role']}] {content}")
                else:
                    kinds = [c.get("type") for c in content]
                    text = next((c.get("text") for c in content if c.get("type") == "text"), "")
                    print(f"[{message['role']}] content_types={kinds} window_frames={clip.shape[0]}")
                    print(f"[{message['role']}] text={text}")
        for entry, n_swaps, mode in tracking_plan[:2]:
            gt = gt_by_video[entry["video"]]
            if mode == "video":
                start, end = prefix_frame_range(n_swaps)
                clip = sample_clip(args.video_dir / entry["video"], start, end)
                sampled = sampled_frame_count(clip, args.sample_fps)
            else:
                clip = None
                sampled = None
            msgs = tracking_messages(
                clip,
                POSITION_NAMES[gt["initial_pos"]],
                n_swaps,
                swap_descriptions(gt["swaps"][:n_swaps]) if mode == "text" else None,
            )
            print(f"\n--- tracking {entry['video']} n_swaps={n_swaps} mode={mode} "
                  f"gt={POSITION_NAMES[gt['states'][n_swaps]]} sampled_frames={sampled}")
            for message in msgs:
                content = message["content"]
                if isinstance(content, str):
                    print(f"[{message['role']}] {content}")
                else:
                    kinds = [c.get("type") for c in content]
                    text = next((c.get("text") for c in content if c.get("type") == "text"), "")
                    print(f"[{message['role']}] content_types={kinds}")
                    print(f"[{message['role']}] text={text}")
        print("\nDry run OK.")
        return

    if args.backend == "vllm":
        runner = VLLMRunner(args.model_dir, args.parallel, args.gpu_memory_utilization)
        if args.thinking == "on":
            runner.validate_thinking()
        model, processor = runner, runner.processor
    else:
        model, processor = load_transformers_model(args.model_dir)
    thinking = args.thinking == "on"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    single_swap_path = args.out_dir / "results_single_swap_tracking.jsonl"
    tracking_path = args.out_dir / "results_tracking.jsonl"
    single_swap_file = open(single_swap_path, "w", encoding="utf-8")
    tracking_file = open(tracking_path, "w", encoding="utf-8")

    single_swap_rows: list[dict] = []
    tracking_rows: list[dict] = []

    try:
        for video_idx, entry in enumerate(entries, start=1):
            video_path = args.video_dir / entry["video"]
            gt = gt_by_video[entry["video"]]

            # ---- single-swap tracking diagnostic: one clip per swap ----
            single_swap_by_mode: dict[str, dict[int, dict]] = {mode: {} for mode in scoring_modes}
            for k in range(1, len(gt["swaps"]) + 1):
                start, end = swap_frame_range(k)
                clip = sample_clip(video_path, start, end)
                messages = single_swap_messages(clip, k)
                vk = video_processor_kwargs(clip, args.sample_fps)
                sampled = sampled_frame_count(clip, args.sample_fps)
                gt_option = SWAP_OPTION_TEXT[gt["swaps"][k - 1]]
                for mode in scoring_modes:
                    if mode == "forced_choice" and args.backend == "transformers":
                        candidates = tuple(SWAP_OPTION_TEXT[p] for p in SWAP_PAIRS)
                        scores = score_candidates(model, processor, messages, candidates, args.device, processor_kwargs=vk)
                        predicted = max(candidates, key=lambda c: scores[c])
                        margin = scores[gt_option] - max(scores[c] for c in candidates if c != gt_option)
                        row = {
                            "scoring": "forced_choice",
                            "candidate_scores": scores,
                            "score_margin": margin,
                            "predicted": predicted,
                        }
                    else:
                        max_tokens = args.max_think_tokens if thinking else args.max_new_tokens
                        raw = (model.infer(messages, max_tokens, thinking, vk)
                               if isinstance(model, VLLMRunner) else
                               transformers_inference(model, processor, messages, args.device, max_tokens, thinking, processor_kwargs=vk))
                        predicted = parse_single_swap_option(raw)
                        row = {"scoring": "free", "raw_output": raw, "predicted": predicted,
                               "thinking_requested": thinking, **thinking_evidence(raw)}
                    single_swap_rows.append({
                        "task": "single_swap_tracking",
                        "video": entry["video"],
                        "sample_id": f"{entry['video'].rsplit('.', 1)[0]}_swap{k}",
                        "swap_index": k,
                        "swap_ground_truth": list(gt["swaps"][k - 1]),
                        "swap_ground_truth_text": gt_option,
                        "frame_range": [start, end],
                        "sample_fps": args.sample_fps,
                        "sampled_frames": sampled,
                        "correct": row["predicted"] == gt_option,
                        **row,
                    })
                    single_swap_by_mode[mode][k] = {
                        "swap_index": k,
                        "swap_ground_truth_text": gt_option,
                        "predicted": row["predicted"],
                        "correct": row["predicted"] == gt_option,
                    }
                    single_swap_file.write(json.dumps(single_swap_rows[-1], ensure_ascii=False) + "\n")
                    print(f"  single_swap {entry['video']} swap{k} [{mode}]: "
                          f"gt={gt_option} pred={single_swap_by_mode[mode][k]['predicted']} "
                          f"ok={single_swap_by_mode[mode][k]['correct']} sampled={sampled}")

            # ---- tracking: prefixes of 0..5 swaps ----
            for n_swaps in range(len(gt["states"])):
                for mode in tracking_modes:
                    if mode == "video":
                        start, end = prefix_frame_range(n_swaps)
                        clip = sample_clip(video_path, start, end)
                        vk = video_processor_kwargs(clip, args.sample_fps)
                        sampled = sampled_frame_count(clip, args.sample_fps)
                    else:
                        start, end = None, None
                        clip = None
                        vk = None
                        sampled = None
                    messages = tracking_messages(
                        clip,
                        POSITION_NAMES[gt["initial_pos"]],
                        n_swaps,
                        swap_descriptions(gt["swaps"][:n_swaps]) if mode == "text" else None,
                    )
                    gt_final = POSITION_NAMES[gt["states"][n_swaps]]
                    for scoring in scoring_modes:
                        if scoring == "forced_choice" and args.backend == "transformers":
                            scores = score_candidates(model, processor, messages, TRACKING_OPTIONS, args.device, processor_kwargs=vk)
                            predicted = max(TRACKING_OPTIONS, key=lambda c: scores[c])
                            margin = scores[gt_final] - max(scores[c] for c in TRACKING_OPTIONS if c != gt_final)
                            row = {"scoring": "forced_choice", "candidate_scores": scores,
                                   "score_margin": margin, "predicted": predicted}
                        else:
                            max_tokens = args.max_think_tokens if thinking else args.max_new_tokens
                            raw = (model.infer(messages, max_tokens, thinking, vk)
                                   if isinstance(model, VLLMRunner) else
                                   transformers_inference(model, processor, messages, args.device, max_tokens, thinking, processor_kwargs=vk))
                            predicted = parse_tracking_option(raw)
                            row = {"scoring": "free", "raw_output": raw, "predicted": predicted,
                                   "thinking_requested": thinking, **thinking_evidence(raw)}
                        prefix_single_swap = [single_swap_by_mode[scoring][k] for k in range(1, n_swaps + 1)]
                        tracking_rows.append({
                            "task": "tracking",
                            "video": entry["video"],
                            "sample_id": f"{entry['video'].rsplit('.', 1)[0]}_track_n{n_swaps}",
                            "input_mode": mode,
                            "n_swaps": n_swaps,
                            "initial_state": POSITION_NAMES[gt["initial_pos"]],
                            "swap_sequence": [list(s) for s in gt["swaps"]],
                            "step_ground_truth_states": [POSITION_NAMES[s] for s in gt["states"]],
                            "final_ground_truth": gt_final,
                            "final_prediction": predicted,
                            "final_correct": predicted == gt_final,
                            "frame_range": [start, end] if start is not None else None,
                            "sample_fps": args.sample_fps if mode == "video" else None,
                            "sampled_frames": sampled,
                            "single_swap_predictions": prefix_single_swap,
                            "single_swap_all_correct": bool(prefix_single_swap) and all(
                                p["correct"] for p in prefix_single_swap
                            ),
                            **row,
                        })
                        tracking_file.write(json.dumps(tracking_rows[-1], ensure_ascii=False) + "\n")
                        print(f"  tracking {entry['video']} n={n_swaps} mode={mode} "
                              f"gt={gt_final} pred={predicted} ok={predicted == gt_final}")
            print(f"[video {video_idx}/{len(entries)}] {entry['video']} done")
            _frame_cache.pop(str(video_path), None)
    finally:
        single_swap_file.close()
        tracking_file.close()

    summary = build_summary(single_swap_rows, tracking_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSingle-swap tracking results: {single_swap_path}")
    print(f"Tracking results: {tracking_path}")
    print(f"Summary: {args.out_dir / 'summary.json'}")


def build_summary(single_swap_rows: list[dict], tracking_rows: list[dict]) -> dict:
    summary: dict[str, Any] = {}

    # ---- single-swap tracking diagnostic ----
    ss_rows = [r for r in single_swap_rows if r["scoring"] == "forced_choice"]
    if not ss_rows:
        ss_rows = [r for r in single_swap_rows if r["scoring"] == "free"]
    summary["single_swap_tracking"] = {
        "n": len(ss_rows),
        "accuracy": (sum(r["correct"] for r in ss_rows) / len(ss_rows)) if ss_rows else None,
    }
    per_swap = {}
    for k, rows in sorted((k, [r for r in ss_rows if r["swap_index"] == k]) for k in range(1, 6)):
        per_swap[str(k)] = {
            "n": len(rows),
            "accuracy": (sum(r["correct"] for r in rows) / len(rows)) if rows else None,
        }
    summary["single_swap_tracking"]["by_swap_index"] = per_swap
    per_pair = {}
    for pair in SWAP_PAIRS:
        text = SWAP_OPTION_TEXT[pair]
        rows = [r for r in ss_rows if r["swap_ground_truth_text"] == text]
        per_pair[text] = {
            "n": len(rows),
            "accuracy": (sum(r["correct"] for r in rows) / len(rows)) if rows else None,
        }
    summary["single_swap_tracking"]["by_swap_type"] = per_pair

    # Adjacent (Left-Middle / Middle-Right) vs non-adjacent (Left-Right) swaps.
    # Reported without assuming any pattern (e.g. 0% on non-adjacent) is a code bug.
    adjacent_pairs = {(1, 2), (2, 3)}
    adjacent_rows = [r for r in ss_rows if tuple(r["swap_ground_truth"]) in adjacent_pairs]
    non_adjacent_rows = [r for r in ss_rows if tuple(r["swap_ground_truth"]) not in adjacent_pairs]
    summary["single_swap_tracking"]["by_adjacency"] = {
        "adjacent": {
            "n": len(adjacent_rows),
            "accuracy": (sum(r["correct"] for r in adjacent_rows) / len(adjacent_rows)) if adjacent_rows else None,
        },
        "non_adjacent": {
            "n": len(non_adjacent_rows),
            "accuracy": (sum(r["correct"] for r in non_adjacent_rows) / len(non_adjacent_rows)) if non_adjacent_rows else None,
        },
    }

    # ---- tracking ----
    track_rows = [r for r in tracking_rows if r["scoring"] == "forced_choice"]
    if not track_rows:
        track_rows = [r for r in tracking_rows if r["scoring"] == "free"]
    by_mode: dict[str, dict] = {}
    for mode in ("video", "text"):
        rows = [r for r in track_rows if r["input_mode"] == mode]
        by_n = {}
        for n_swaps in sorted({r["n_swaps"] for r in rows}):
            subset = [r for r in rows if r["n_swaps"] == n_swaps]
            by_n[str(n_swaps)] = {
                "n": len(subset),
                "accuracy": (sum(r["final_correct"] for r in subset) / len(subset)) if subset else None,
            }
        by_mode[mode] = {
            "n": len(rows),
            "accuracy": (sum(r["final_correct"] for r in rows) / len(rows)) if rows else None,
            "by_n_swaps": by_n,
        }
    summary["tracking"] = by_mode

    # ---- trajectory analysis (full 5-swap, video mode) ----
    full = [r for r in track_rows if r["input_mode"] == "video" and r["n_swaps"] == 5]
    ss_correct = [r for r in full if r["single_swap_all_correct"]]
    ss_correct_final_wrong = [r for r in ss_correct if not r["final_correct"]]
    summary["trajectory_analysis"] = {
        "n_trajectories": len(full),
        "n_single_swap_all_correct_trajectories": len(ss_correct),
        "single_swap_all_correct_trajectory_rate": (len(ss_correct) / len(full)) if full else None,
        "final_accuracy_on_single_swap_all_correct": (
            (sum(r["final_correct"] for r in ss_correct) / len(ss_correct))
            if ss_correct else None
        ),
        "final_accuracy_all": (sum(r["final_correct"] for r in full) / len(full)) if full else None,
        "single_swap_all_correct_but_final_wrong": {
            "n": len(ss_correct_final_wrong),
            "rate_of_single_swap_all_correct": (
                len(ss_correct_final_wrong) / len(ss_correct)
            ) if ss_correct else None,
            "rate_of_all_trajectories": (len(ss_correct_final_wrong) / len(full)) if full else None,
            "sample_ids": [r["sample_id"] for r in ss_correct_final_wrong],
        },
    }
    return summary


if __name__ == "__main__":
    main()
