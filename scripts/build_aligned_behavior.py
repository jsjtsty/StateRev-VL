"""Build the Transformers-aligned behavior manifest for the composition analysis.

The old behavioral audit (``outputs/vetbench/behavior_audit_v2/``) was generated
with vLLM. The hidden states and native logits come from transformers, so all
mechanism analysis must use behavior produced by the SAME backend. This script
rebuilds the mechanism manifest (identical columns and flag definitions as
``scripts/build_behavior_audit.py``) from the transformers regeneration
(``outputs/vetbench/transformers_behavior_v1/``) and compares it with the old
vLLM manifest.

Outputs (in ``outputs/vetbench/composition_analysis_v1/``):
  transformers_behavior.csv      250-row manifest, same columns as
                                 behavior_audit_v2/mechanism_candidates.csv
  canonical_stale_failure.csv    aligned canonical revision rows (51-column
                                 format of canonical_revision_candidates.csv)
  aligned_behavior_summary.json  counts, per-step accuracy, vLLM-vs-transformers
                                 agreement, canonical overlap

Usage (from project root):
  python scripts/build_aligned_behavior.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_vetbench_screening import CUP_DIR  # noqa: F401  (kept for path checks)

ROOT = Path(__file__).resolve().parent.parent
BEHAVIOR_DIR = ROOT / "outputs" / "vetbench" / "behavior_audit_v2"
TRANSFORMERS_DIR = ROOT / "outputs" / "vetbench" / "transformers_behavior_v1"
OUT_DIR = ROOT / "outputs" / "vetbench" / "composition_analysis_v1"

MANIFEST_CSV_COLUMNS = [
    "trajectory_id", "t", "video", "prefix_path", "frame_start", "frame_end",
    "clip_frames", "sampled_frames", "sample_fps", "initial_state", "n_swaps_shown",
    "gt_event", "event_pred", "event_correct",
    "gt_prev_state", "gt_state", "prev_state_pred", "state_pred", "state_correct",
    "joint_class", "is_transition", "prev_state_correct",
    "clean_revision", "clean_maintenance",
]
CANONICAL_CSV_COLUMNS = [
    "trajectory_id", "t", "video", "gt_event", "event_pred",
    "gt_prev_state", "prev_state_pred", "gt_state", "state_pred",
    "state_correct", "stale_failure", "prefix_path",
]


def safe_eq(a: Any, b: Any) -> bool:
    """Equality that never treats two unparsed (None) predictions as 'equal'."""
    return a is not None and a == b


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in rows:
            w.writerow([cell(r.get(c)) for c in fieldnames])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformers-dir", type=Path, default=TRANSFORMERS_DIR)
    ap.add_argument("--behavior-dir", type=Path, default=BEHAVIOR_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    state_rows = load_jsonl(args.transformers_dir / "prefix_state.jsonl")
    event_rows = load_jsonl(args.transformers_dir / "in_context_event.jsonl")
    assert len(state_rows) == 300, len(state_rows)
    assert len(event_rows) == 250, len(event_rows)

    state_by: dict[str, dict[int, dict]] = {}
    for r in state_rows:
        state_by.setdefault(r["video"], {})[int(r["t"])] = r
    assert all(len(v) == 6 for v in state_by.values())
    assert set(state_by) == {r["video"] for r in event_rows}

    # ---- old vLLM manifest (GT + row geometry + old predictions) -------------
    old = list(csv.DictReader(open(args.behavior_dir / "mechanism_candidates.csv",
                                   encoding="utf-8")))
    assert len(old) == 250
    T = lambda r, k: r[k] == "true"
    old_by = {(r["video"], int(r["t"])): r for r in old}

    manifest: list[dict] = []
    for er in event_rows:
        video, t = er["video"], int(er["t"])
        o = old_by[(video, t)]
        sr_t = state_by[video][t]
        sr_tm1 = state_by[video][t - 1]
        m: dict[str, Any] = {
            "trajectory_id": video.rsplit(".", 1)[0],
            "t": t,
            "video": video,
            "prefix_path": o["prefix_path"],
            "frame_start": o["frame_start"],
            "frame_end": o["frame_end"],
            "clip_frames": sr_t["clip_frames"],
            "sampled_frames": sr_t["sampled_frames"],
            "sample_fps": sr_t["sample_fps"],
            "initial_state": o["initial_state"],
            "n_swaps_shown": t,
            "gt_event": er["event_gt"],
            "event_pred": er["event_prediction"],
            "event_correct": bool(er["event_correct"]),
            "gt_prev_state": sr_tm1["gt_state"],
            "gt_state": sr_t["gt_state"],
            "prev_state_pred": sr_tm1["state_prediction"],
            "state_pred": sr_t["state_prediction"],
            "state_correct": bool(sr_t["state_correct"]),
        }
        m["joint_class"] = (
            f"event_{'correct' if m['event_correct'] else 'wrong'}"
            f"_state_{'correct' if m['state_correct'] else 'wrong'}")
        # ---- identical flag definitions as build_behavior_audit.py -----------
        m["is_transition"] = bool(m["gt_state"] != m["gt_prev_state"])
        m["prev_state_correct"] = bool(safe_eq(m["prev_state_pred"], m["gt_prev_state"]))
        m["clean_revision"] = bool(m["is_transition"] and m["event_correct"]
                                   and m["prev_state_correct"])
        m["clean_maintenance"] = bool((not m["is_transition"]) and m["event_correct"]
                                      and m["prev_state_correct"])
        m["_vllm_state_pred"] = o["state_pred"]
        m["_vllm_event_pred"] = o["event_pred"]
        m["_vllm_state_correct"] = T(o, "state_correct")
        m["_vllm_event_correct"] = T(o, "event_correct")
        m["_vllm_clean_revision"] = T(o, "clean_revision")
        manifest.append(m)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_cols = [c for c in MANIFEST_CSV_COLUMNS]
    write_csv(args.out_dir / "transformers_behavior.csv",
              [{k: v for k, v in m.items() if not k.startswith("_vllm")}
               for m in manifest], manifest_cols)

    # ---- canonical revision rows (same definition as v2) --------------------
    canonical = []
    for m in manifest:
        if not m["clean_revision"]:
            continue
        stale = (not m["state_correct"]) and m["state_pred"] == m["gt_prev_state"]
        canonical.append({
            "trajectory_id": m["trajectory_id"], "t": m["t"], "video": m["video"],
            "gt_event": m["gt_event"], "event_pred": m["event_pred"],
            "gt_prev_state": m["gt_prev_state"],
            "prev_state_pred": m["prev_state_pred"],
            "gt_state": m["gt_state"], "state_pred": m["state_pred"],
            "state_correct": m["state_correct"], "stale_failure": bool(stale),
            "prefix_path": m["prefix_path"],
        })
    write_csv(args.out_dir / "canonical_stale_failure.csv", canonical,
              CANONICAL_CSV_COLUMNS)

    # ---- summary ------------------------------------------------------------
    n = len(manifest)
    n_rev = sum(m["clean_revision"] for m in manifest)
    n_succ = sum(m["clean_revision"] and m["state_correct"] for m in manifest)
    n_fail = n_rev - n_succ
    n_stale = sum(c["stale_failure"] for c in canonical)
    n_maint = sum(m["clean_maintenance"] for m in manifest)

    def acc(rows_: list[dict], key: str) -> float:
        return (sum(1 for r in rows_ if r[key]) / len(rows_) if rows_ else None)

    # t=0 state rows (initial-state question) are not in the event-driven
    # manifest; report them separately
    t0_rows = [r for r in state_rows if int(r["t"]) == 0]
    per_step_state = {str(t): acc([m for m in manifest if m["t"] == t], "state_correct")
                      for t in range(1, 6)}
    per_step_state["0_initial_state_question"] = (
        sum(1 for r in t0_rows if r["state_correct"]) / len(t0_rows))
    per_step_event = {str(t): acc([m for m in manifest if m["t"] == t], "event_correct")
                      for t in range(1, 6)}

    # vLLM vs transformers agreement
    agree_state = sum(m["state_pred"] == m["_vllm_state_pred"] for m in manifest)
    agree_event = sum(m["event_pred"] == m["_vllm_event_pred"] for m in manifest)
    agree_state_t = {str(t): sum(1 for m in manifest if m["t"] == t
                                 and m["state_pred"] == m["_vllm_state_pred"])
                     for t in range(1, 6)}
    # canonical overlap
    old_can = {(r["trajectory_id"], int(r["t"]))
               for r in old if T(r, "clean_revision")}
    new_can = {(m["trajectory_id"], m["t"]) for m in manifest
               if m["clean_revision"]}
    old_stale = {(r["trajectory_id"], int(r["t"])) for r in old
                 if T(r, "clean_revision") and not T(r, "state_correct")
                 and r["state_pred"] == r["gt_prev_state"]}
    new_stale = {(c["trajectory_id"], int(c["t"])) for c in canonical
                 if c["stale_failure"]}
    flip = [f"{a}_{b}" for (a, b) in sorted(old_can ^ new_can)]

    summary = {
        "backend": "transformers (Qwen3VLForConditionalGeneration, bf16, "
                   "greedy do_sample=False, max_new_tokens=32, sample_fps=8.0, "
                   "identical prompt/processor/chat-template as the hidden-state "
                   "probe)",
        "n_rows": n,
        "state_correct": sum(m["state_correct"] for m in manifest),
        "event_correct": sum(m["event_correct"] for m in manifest),
        "prev_state_correct": sum(m["prev_state_correct"] for m in manifest),
        "is_transition": sum(m["is_transition"] for m in manifest),
        "joint_class_counts": dict(Counter(m["joint_class"] for m in manifest)),
        "clean_revision": {"n": n_rev, "success": n_succ, "failure": n_fail,
                           "stale": n_stale},
        "clean_maintenance": n_maint,
        "per_step_state_acc": per_step_state,
        "per_step_event_acc": per_step_event,
        "state_acc": acc(manifest, "state_correct"),
        "event_acc": acc(manifest, "event_correct"),
        "vllm_vs_transformers": {
            "state_pred_agree": agree_state,
            "state_pred_agree_rate": agree_state / n,
            "state_pred_agree_per_t": agree_state_t,
            "event_pred_agree": agree_event,
            "event_pred_agree_rate": agree_event / n,
            "state_correct_flip": sum(
                m["state_correct"] != m["_vllm_state_correct"] for m in manifest),
            "event_correct_flip": sum(
                m["event_correct"] != m["_vllm_event_correct"] for m in manifest),
            "unparsed_state": sum(m["state_pred"] is None for m in manifest),
            "unparsed_event": sum(m["event_pred"] is None for m in manifest),
        },
        "canonical": {
            "n_old_vllm": len(old_can),
            "n_new_transformers": len(new_can),
            "overlap": len(old_can & new_can),
            "diff": flip,
            "stale_old": len(old_stale),
            "stale_new": len(new_stale),
            "stale_overlap": len(old_stale & new_stale),
        },
        "old_vllm_counts": {
            "state_correct": sum(T(r, "state_correct") for r in old),
            "event_correct": sum(T(r, "event_correct") for r in old),
            "clean_revision": sum(T(r, "clean_revision") for r in old),
            "clean_maintenance": sum(T(r, "clean_maintenance") for r in old),
        },
        "definitions": {
            "prev_state_correct": "state_pred_{t-1} == GT_{t-1} (t-1 row from the "
                                  "same backend; t=1 uses the t=0 state row)",
            "clean_revision": "true transition AND event_correct_t AND "
                              "prev_state_correct",
            "clean_revision_success": "clean revision with state_correct_t",
            "clean_revision_failure": "clean revision with not state_correct_t",
            "canonical_stale_failure": "clean revision failure with "
                                       "state_pred_t == GT_{t-1}",
            "clean_maintenance": "non-transition AND event_correct_t AND "
                                 "prev_state_correct",
        },
    }
    (args.out_dir / "aligned_behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {args.out_dir}/transformers_behavior.csv ({n} rows), "
          f"canonical_stale_failure.csv ({len(canonical)} rows), "
          f"aligned_behavior_summary.json")
    print(f"state_correct={summary['state_correct']}/{n}  "
          f"event_correct={summary['event_correct']}/{n}")
    print(f"clean_revision={n_rev} (success {n_succ}, failure {n_fail}, "
          f"stale {n_stale})  clean_maintenance={n_maint}")
    print(f"vLLM vs transformers: state agree {agree_state}/{n} "
          f"({agree_state/n:.3f}), event agree {agree_event}/{n} "
          f"({agree_event/n:.3f})")
    print(f"canonical: old {len(old_can)} -> new {len(new_can)} "
          f"(overlap {len(old_can & new_can)}); stale {len(old_stale)} -> "
          f"{len(new_stale)}")


if __name__ == "__main__":
    main()
