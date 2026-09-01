"""Experiment 4: build a temporal intervention manifest (no model, no GPU).

Goal
----
Experiments 1-3 operate on the model's hidden states and on the text prompt.
This experiment prepares the *video-side* interventions: for every canonical
sample we pin down, frame-by-frame, WHERE in the video each temporal segment
lives, and EXACTLY which source frames the processor samples inside each
segment. Future runs can then (a) drop/alter frames of the current swap
segment to test whether the stale answer comes from the swap evidence itself,
or (b) keep the pre-swap frames and drop the post-swap tail to test whether
the model relies on the post-event visual state.

Frame timing (derived from the VET-Bench generator constants, NOT from
visual inspection):

  SHUFFLE_START_MS = 1900   -> first swap starts at frame 57 (30 fps)
  SWAP_CYCLE_MS    = 2000   -> each swap occupies 60 frames
  interval_ms      = 0      -> swaps are back-to-back (cup.json)

  swap k (1-based) occupies frames [57+(k-1)*60, 57+k*60)
  video total_frames = 372 (all 50 cup videos)

For a sample at step t (the model saw clips [0, 57+t*60)):

  pre_current_event     [0,            57+(t-1)*60)
  current_event         [57+(t-1)*60,  57+t*60)
  post_event_immediate  (gap between current and next event)
                        empty interval for t<5 (swaps back-to-back, 0
                        frames); [357, 372) for t=5: a real 15-frame
                        post-tail
  next_swap             [57+t*60,      57+(t+1)*60)   for t<5 only: the next
                        event, which any extended clip hits immediately
                        (informational; NOT a post-event window)
  post_event_extended   [57+t*60,      372)           for t<5 this contains
                                                       the LATER swaps (it is
                                                       metadata, not a clean
                                                       post-event window)

Sampling - TWO different pipelines actually ran (critical!):

  NOMINAL (intended, 8 fps): what the probe runs used. The probe script
  deliberately avoids the `enable_thinking` kwarg so the processor receives
  fps=8 + video_metadata (fps=30):
      num_frames = int(clip_len/30*8) clipped to [4,768,clip_len]
      indices    = round(linspace(0, clip_len-1, num_frames))
      -> 31 / 47 / 63 / 79 / 95 frames for t=1..5

  ACTUAL (behavior audit + rescue baseline): run_inference passes
  enable_thinking to apply_chat_template, which makes the base class DROP
  processor_kwargs (see the NOTE in run_hidden_state_probe.py). The
  processor then falls back to its defaults (fps=2, no metadata -> fps=24):
      num_frames = int(clip_len/24*2) clipped to [4,768,clip_len]
      indices    = round(linspace(0, clip_len-1, num_frames))
      -> 9 / 14 / 19 / 24 / 29 frames for t=1..5
  The behavior CSV column "sampled_frames" (31/47/63/79/95) is the NOMINAL
  formula, NOT what the audit actually sampled.

The manifest records BOTH index sets (per segment) for the baseline clip
[0, frame_end) and the extended clip [0, 372). A future patching run MUST
rebuild its clip to match the pipeline it wants to reproduce: ACTUAL set to
stay comparable with the behavior/rescue data, NOMINAL set to match the
hidden-state probe data. Extending the clip changes num_frames, so every
index shifts - use the set matching the run's own clip length.

Validation performed at build time (hard failures -> exit 1):
  * every boundary is an integer, 0 <= pre < event <= clip_end <= 372
  * segments are contiguous (pre end == event start, etc.)
  * sampled indices strictly increasing, within [0, clip_len)
  * frame_end of the behavior row == 57 + t*60 (schedule cross-check)
  * baseline num_sampled == sampled_frames recorded in the behavior CSV
  * cup.json total_frames == 372 and fps == 30 for every referenced video
  * video file exists on disk

Output
------
  temporal_intervention_manifest.csv  one row per (sample) with all
                                      boundaries, frame counts, sampled
                                      index lists (semicolon-joined) and
                                      validation flags.

Usage
-----
  python scripts/build_temporal_intervention_manifest.py
  python scripts/build_temporal_intervention_manifest.py --include-rest
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_OUT_DIR = Path("outputs/vetbench/stale_origin_analysis_v1")
SHUFFLE_START_MS = 1900
SWAP_CYCLE_MS = 2000
FPS = 30
FIRST_SWAP_START = int(SHUFFLE_START_MS / 1000 * FPS)      # 57
SWAP_LEN = int(SWAP_CYCLE_MS / 1000 * FPS)                  # 60
EXPECTED_TOTAL_FRAMES = 372
SAMPLE_FPS = 8.0
MIN_FRAMES, MAX_FRAMES = 4, 768  # Qwen3-VL video processor defaults


def swap_range(t: int) -> tuple[int, int]:
    """Frame range [start, end) of swap t (1-based)."""
    return FIRST_SWAP_START + (t - 1) * SWAP_LEN, FIRST_SWAP_START + t * SWAP_LEN


def processor_sample_indices(clip_len: int, metadata_fps: float,
                             target_fps: float) -> list[int]:
    """Exact frame-index selection of Qwen3VLVideoProcessor.sample_frames."""
    num_frames = int(clip_len / metadata_fps * target_fps)
    num_frames = min(max(num_frames, MIN_FRAMES), MAX_FRAMES, clip_len)
    if num_frames == 1:
        return [0]
    import numpy as np
    return [int(x) for x in
            np.linspace(0, clip_len - 1, num_frames).round().astype(int)]


def idx_in(indices: list[int], a: int, b: int) -> list[int]:
    return [i for i in indices if a <= i < b]


def load_samples(behavior_csv: Path) -> list[dict]:
    """stale 26 + clean_maintenance 23 + other canonical failures (17).

    Grouping uses the SAME clean_revision/clean_maintenance columns as the
    aligned behavior build (and run_revision_rescue.py), so the counts are
    identical to every other experiment. Optionally adds same-step (t>=2)
    rest controls via --include-rest."""
    rows = list(csv.DictReader(open(behavior_csv)))

    def T(x):
        return str(x).strip().lower() == "true"

    def group_of(r: dict) -> str | None:
        if not T(r["clean_revision"]) and not T(r["clean_maintenance"]):
            return "rest" if int(r["t"]) in (2, 3, 4, 5) else None
        if T(r["clean_maintenance"]):
            return "maintenance"
        if T(r["state_correct"]):
            return "success"
        if r["state_pred"] == r["gt_prev_state"]:
            return "stale"
        return "other_failure"

    out = []
    for r in rows:
        g = group_of(r)
        if g in ("stale", "maintenance", "other_failure"):
            r["group"] = g
            out.append(r)
    return out


def build(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cup = {c["video"]: c for c in json.load(open(args.cup_json))}
    samples = load_samples(Path(args.behavior_csv))
    if args.include_rest:
        rows_all = list(csv.DictReader(open(args.behavior_csv)))
        have = {(r["trajectory_id"], r["t"]) for r in samples}
        for r in rows_all:
            key = (r["trajectory_id"], r["t"])
            if key not in have and int(r["t"]) >= 2:
                r["group"] = "rest"
                samples.append(r)

    manifest = []
    errors: list[str] = []
    for r in samples:
        t = int(r["t"])
        video = r["video"]
        meta = cup.get(video)
        row = {"trajectory_id": r["trajectory_id"], "t": t, "group": r["group"],
               "video": video,
               "initial_state": r["initial_state"],
               "gt_prev_state": r["gt_prev_state"], "gt_state": r["gt_state"],
               "baseline_state_pred": r["state_pred"]}
        if meta is None:
            errors.append(f"{video}: not in cup.json")
            continue
        total = int(meta["total_frames"])
        if total != EXPECTED_TOTAL_FRAMES or int(meta["fps"]) != FPS:
            errors.append(f"{video}: total_frames={total} fps={meta['fps']}")
        if not Path(r["prefix_path"]).exists():
            errors.append(f"{video}: file missing at {r['prefix_path']}")

        frame_end = int(r["frame_end"])
        ok_end = (frame_end == FIRST_SWAP_START + t * SWAP_LEN)
        row["frame_end"] = frame_end
        row["frame_end_matches_schedule"] = ok_end
        if not ok_end:
            errors.append(f"{r['trajectory_id']}_t{t}: frame_end={frame_end} "
                          f"!= {FIRST_SWAP_START + t * SWAP_LEN}")

        pre_a, ev_a = 0, FIRST_SWAP_START + (t - 1) * SWAP_LEN
        ev_b = FIRST_SWAP_START + t * SWAP_LEN
        if t < 5:
            post_imm_a, post_imm_b = ev_b, ev_b  # empty gap
            next_swap = (ev_b, FIRST_SWAP_START + (t + 1) * SWAP_LEN)
        else:
            post_imm_a, post_imm_b = ev_b, total  # real 15-frame tail
            next_swap = None
        post_ext_a, post_ext_b = ev_b, total

        bounds = {"pre": (pre_a, ev_a), "event": (ev_a, ev_b),
                  "post_imm": (post_imm_a, post_imm_b),
                  "post_ext": (post_ext_a, post_ext_b)}
        if next_swap is not None:
            bounds["next_swap"] = next_swap
        ok_bounds = (
            all(isinstance(a, int) and isinstance(b, int)
                for a, b in bounds.values())
            and 0 <= pre_a < ev_a <= ev_b <= total
            and post_imm_a <= post_imm_b <= total
            and post_ext_b == total
            and ev_b == frame_end  # current event ends where the clip ends
        )
        row["bounds_ok"] = ok_bounds
        if not ok_bounds:
            errors.append(f"{r['trajectory_id']}_t{t}: bad boundaries "
                          f"{bounds}")

        # ACTUAL = the behavior audit / rescue baseline's effective sampling
        # (run_inference's enable_thinking kwarg drops processor_kwargs ->
        # processor defaults: target fps=2, metadata fps default 24).
        # NOMINAL = the intended 8 fps sampling used by the hidden-state
        # probe runs (and by the behavior CSV's sampled_frames column).
        base_actual = processor_sample_indices(frame_end, 24, 2.0)
        base_nominal = processor_sample_indices(frame_end, FPS, SAMPLE_FPS)
        row.update({
            "baseline_clip_start": 0, "baseline_clip_end": frame_end,
            "baseline_actual_num_sampled": len(base_actual),
            "baseline_nominal_num_sampled": len(base_nominal),
            "baseline_actual_sampled_idx": ";".join(map(str, base_actual)),
            "baseline_nominal_sampled_idx": ";".join(map(str, base_nominal)),
            "baseline_nominal_matches_behavior_csv":
                len(base_nominal) == int(r["sampled_frames"]),
        })
        if len(base_nominal) != int(r["sampled_frames"]):
            errors.append(
                f"{r['trajectory_id']}_t{t}: nominal {len(base_nominal)} "
                f"sampled frames != {r['sampled_frames']} in behavior CSV")

        # extended clip = [0, total) : reference for future post-event runs
        ext_actual = processor_sample_indices(total, 24, 2.0)
        ext_nominal = processor_sample_indices(total, FPS, SAMPLE_FPS)
        row.update({
            "extended_clip_start": 0, "extended_clip_end": total,
            "extended_actual_num_sampled": len(ext_actual),
            "extended_nominal_num_sampled": len(ext_nominal),
            "extended_actual_sampled_idx": ";".join(map(str, ext_actual)),
            "extended_nominal_sampled_idx": ";".join(map(str, ext_nominal)),
            "extended_index_shift_note":
                "extending the clip changes num_frames, so every index "
                "shifts vs the baseline set; future runs MUST use the set "
                "matching their own clip length",
        })

        ok_sorted = all(
            all(x < y for x, y in zip(seq, seq[1:])) for seq in
            (base_actual, base_nominal, ext_actual, ext_nominal))
        row["sampled_indices_strictly_increasing"] = ok_sorted
        if not ok_sorted:
            errors.append(f"{r['trajectory_id']}_t{t}: non-increasing idx")

        for name, (a, b) in bounds.items():
            row[f"{name}_start"] = a
            row[f"{name}_end"] = b
            row[f"{name}_frames"] = max(b - a, 0)
            for tag, idx in (("baseline_actual", base_actual),
                             ("baseline_nominal", base_nominal),
                             ("extended_actual", ext_actual),
                             ("extended_nominal", ext_nominal)):
                row[f"{tag}_{name}_sampled_idx"] = ";".join(
                    map(str, idx_in(idx, a, b)))
                row[f"{tag}_{name}_n_sampled"] = len(idx_in(idx, a, b))

        row["total_frames"] = total
        row["fps"] = FPS
        row["sample_fps_nominal"] = SAMPLE_FPS
        row["sample_fps_actual"] = 2.0
        row["metadata_fps_actual"] = 24.0
        row["visual_token_map_note"] = (
            "per-frame visual token count is set by the processor's "
            "smart_resize (temporal_patch/merge) at runtime; this manifest "
            "records source-frame indices only. A future patching run must "
            "rebuild the clip from these indices and re-encode with the "
            "same processor to obtain the visual token boundaries.")
        manifest.append(row)

    out_csv = out_dir / "temporal_intervention_manifest.csv"
    if manifest:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)
        print(f"Wrote {out_csv} ({len(manifest)} rows)")

    summary = {
        "mode": "build",
        "n_rows": len(manifest),
        "groups": {g: sum(1 for m in manifest if m["group"] == g)
                   for g in sorted({m["group"] for m in manifest})},
        "schedule": {
            "first_swap_start_frame": FIRST_SWAP_START,
            "swap_len_frames": SWAP_LEN,
            "swap_t_range": [f"[{FIRST_SWAP_START + (t-1)*SWAP_LEN}, "
                             f"{FIRST_SWAP_START + t*SWAP_LEN})" for t in range(1, 6)],
            "total_frames": EXPECTED_TOTAL_FRAMES,
            "fps": FPS,
            "note": "swaps are back-to-back (interval_ms=0 in cup.json); "
                    "post_event_immediate is empty for t<5 and is the real "
                    "15-frame tail [357,372) only for t=5",
        },
        "sampling_formula": (
            "both sets use Qwen3VLVideoProcessor.sample_frames exactly: "
            "num_frames = int(clip_len/metadata_fps*target_fps) clipped to "
            "[4,768,clip_len]; indices = round(linspace(0, clip_len-1, "
            "num_frames)). ACTUAL (behavior audit / rescue baseline): "
            "metadata_fps=24 (default), target_fps=2 (processor default) "
            "-> 9/14/19/24/29 frames at t=1..5. NOMINAL (intended, used by "
            "probe runs): metadata_fps=30, target_fps=8 -> 31/47/63/79/95. "
            "Reason for the discrepancy: run_inference passes "
            "enable_thinking to apply_chat_template, which makes the base "
            "class drop processor_kwargs (fps + video_metadata) - see the "
            "NOTE in run_hidden_state_probe.py."),
        "n_errors": len(errors),
        "errors": errors,
        "validation": "bounds integer/contiguous/within-range; sampled idx "
                      "strictly increasing and in-range; frame_end == "
                      "57+t*60; baseline num_sampled == behavior CSV; "
                      "cup.json total_frames==372 & fps==30; files exist",
    }
    out_json = out_dir / "temporal_intervention_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"groups: {summary['groups']}")
    print("VALIDATION " + ("PASS" if not errors else f"FAIL ({len(errors)})"))
    for e in errors[:20]:
        print("  ERROR:", e)
    if errors:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--cup-json", default="dataset/vetbench/cup/cup.json")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--include-rest", action="store_true",
                    help="add same-step (t>=2) rest controls as a group")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
