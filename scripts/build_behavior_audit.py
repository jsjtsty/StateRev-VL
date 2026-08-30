"""Build the corrected StateRev-VL behavior audit (v2) from the existing 50-video run.

Offline (no GPU, no new inference). Reads the existing audit outputs
(outputs/vetbench/audit/*.jsonl) plus the VET-Bench cup metadata, and produces:

  temporal_event_audit.csv    per (trajectory, t=1..5): event error temporal
                              classification (E_{t-1} / any earlier event / never
                              appeared in prefix / unparsed) with all matching
                              history step indices
  state_dynamics_audit.csv    per (trajectory, t) on true GT transitions
                              (S_t != S_{t-1}): stale-ground-truth-state /
                              prediction-inertia / other-wrong classification
  mechanism_candidates.csv    joint event/state manifest for every prefix step
                              t=1..5 (incl. transition / previous-state-correct /
                              clean-revision / clean-maintenance flags); the
                              event_correct+state_wrong subset is the input for
                              downstream hidden-state analysis
  canonical_revision_candidates.csv  clean revision rows (true transition +
                              event correct + previous state correct) with a
                              non-transition clean maintenance control
  behavior_audit_summary.json corrected + new statistics, incl. the deprecation
                              of the previous round's invalid row-level
                              P(video_correct | isolated_swap_correct)

Findings carried over from the data audit of this round:
- The previous one_swap_audit attached the 5-swap final-video prediction to all 5
  isolated-swap rows of each video; the row-level conditional built on those 250
  rows is invalid and is reported as deprecated (no replacement value).
- The official VET-Bench dataset contains NO standalone 1-swap tracking videos:
  cup.json has 50 videos, all swap_count=5; the upstream repo manifest
  (dataset/vetbench/.hfd/manifest, tiedong/vetbench @ 258185944dba3df09145bb41f0721127e4e19575)
  lists only 50 cup + 50 card videos, all 5-swap. Therefore no official
  one_swap_audit.csv is produced (not fabricated).

Run from the project root:
  python scripts/build_behavior_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR
from run_vetbench_screening import (
    CUP_DIR,
    CUP_META,
    POSITION_NAMES,
    SWAP_OPTION_TEXT,
    SWAP_PAIRS,
    derive_ground_truth,
    load_metadata,
)

AUDIT_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "audit"
OUT_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "behavior_audit_v2"

SWAP_TYPE_ORDER = [SWAP_OPTION_TEXT[p] for p in SWAP_PAIRS]
JOINT_CLASSES = (
    "event_correct_state_correct",
    "event_correct_state_wrong",
    "event_wrong_state_correct",
    "event_wrong_state_wrong",
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def safe_eq(a: Any, b: Any) -> bool:
    """Equality that never treats two unparsed (None) predictions as 'equal'."""
    return a is not None and a == b


def confusion_matrix(rows: list[dict], gt_key: str, pred_key: str) -> dict[str, Any]:
    cols = SWAP_TYPE_ORDER + ["unparsed"]
    matrix = {g: {c: 0 for c in cols} for g in SWAP_TYPE_ORDER}
    for r in rows:
        matrix[r[gt_key]][r[pred_key] or "unparsed"] += 1
    return {"gt_rows": SWAP_TYPE_ORDER, "pred_cols": cols, "matrix": matrix}


# --------------------------------------------------------------------------
# 2. Temporal event audit
# --------------------------------------------------------------------------

def build_temporal_event(event_rows: list[dict], gt_by_video: dict[str, dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for r in event_rows:
        video = r["video"]
        t = r["t"]
        gt_events = [SWAP_OPTION_TEXT[s] for s in gt_by_video[video]["swaps"]]  # E1..E5
        pred = r["event_prediction"]
        history = gt_events[: t - 1]  # E1..E_{t-1}
        match_steps = [j + 1 for j, e in enumerate(history) if pred is not None and e == pred]
        if not r["event_correct"]:
            if pred is None:
                error_class = "unparsed"
            elif t >= 2 and pred == gt_events[t - 2]:
                error_class = "equals_E_t_minus_1"
            elif match_steps:
                error_class = "equals_any_earlier_event"
            else:
                error_class = "never_appeared_in_prefix"
        else:
            error_class = ""
        row: dict[str, Any] = {
            "trajectory_id": video.rsplit(".", 1)[0],
            "video": video,
            "t": t,
            "gt_event": gt_events[t - 1],
            "event_pred": pred,
            "event_correct": r["event_correct"],
            "error_class": error_class,
            "history_match_steps": match_steps,
            "gt_repeated_in_prefix": gt_events[t - 1] in history,
            # "eligible" = parsed wrong prediction at t>=2 where the previous event
            # type DIFFERS from the current GT. Only there is a match with E_{t-1}
            # non-trivial: if E_{t-1} == E_t a wrong prediction cannot equal both, so
            # such rows trivially cannot hit the previous event.
            "prev_eligible": bool(pred is not None and not r["event_correct"]
                                  and t >= 2 and gt_events[t - 2] != gt_events[t - 1]),
        }
        for j in range(1, 6):
            row[f"E{j}_gt"] = gt_events[j - 1] if j <= t else None
        rows.append(row)

    wrong = [r for r in rows if not r["event_correct"]]
    parsed_wrong = [r for r in wrong if r["error_class"] != "unparsed"]
    hit_history = [r for r in wrong if r["error_class"] in ("equals_E_t_minus_1", "equals_any_earlier_event")]
    hit_prev = [r for r in wrong if r["error_class"] == "equals_E_t_minus_1"]
    # eligibility-aware previous-event statistics (only 3 event types exist, so a
    # previous/history hit needs a composition-adjusted baseline to mean anything)
    parsed_wrong_t2 = [r for r in parsed_wrong if r["t"] >= 2]
    eligible = [r for r in parsed_wrong_t2 if r["prev_eligible"]]
    hit_prev_eligible = [r for r in eligible if r["error_class"] == "equals_E_t_minus_1"]

    by_step: dict[str, dict[str, Any]] = {}
    for t in range(1, 6):
        subset = [r for r in rows if r["t"] == t]
        w_subset = [r for r in subset if not r["event_correct"]]
        prev_subset = [r for r in w_subset if r["error_class"] == "equals_E_t_minus_1"]
        by_step[str(t)] = {
            "n": len(subset),
            "accuracy": (sum(r["event_correct"] for r in subset) / len(subset)) if subset else None,
            "n_wrong": len(w_subset),
            "n_equals_prev_event": len(prev_subset),
            "recent_event_selection_failure_rate": (len(prev_subset) / len(w_subset)) if w_subset else None,
        }

    class_counts = Counter(r["error_class"] for r in wrong)
    class_counts_by_step = {
        str(t): dict(Counter(r["error_class"] for r in rows if r["t"] == t and not r["event_correct"]))
        for t in range(1, 6)
    }

    summary = {
        "n_rows": len(rows),
        "n_trajectories": len({r["video"] for r in rows}),
        "overall_accuracy": (sum(r["event_correct"] for r in rows) / len(rows)) if rows else None,
        "per_step_accuracy": {
            str(t): {
                "n": by_step[str(t)]["n"],
                "accuracy": by_step[str(t)]["accuracy"],
                "unparsed": sum(r["event_pred"] is None for r in rows if r["t"] == t),
            }
            for t in range(1, 6)
        },
        "error_class_counts": dict(class_counts),
        "error_class_counts_by_step": class_counts_by_step,
        "errors_hitting_history_event": {
            "definition": "wrong predictions whose type appeared at some earlier step E_j (j < t)",
            "n_wrong": len(wrong),
            "n_hitting_any_history": len(hit_history),
            "rate_of_wrong": (len(hit_history) / len(wrong)) if wrong else None,
            "n_parsed_wrong": len(parsed_wrong),
            "rate_of_parsed_wrong": (len(hit_history) / len(parsed_wrong)) if parsed_wrong else None,
        },
        "recent_event_selection_failure": {
            "definition": "wrong predictions equal to E_{t-1} (the immediately previous swap)",
            "n": len(hit_prev),
            "rate_of_wrong": (len(hit_prev) / len(wrong)) if wrong else None,
            "by_step": by_step,
            "caveat": ("with only 3 event types, matching E_{t-1} is not by itself "
                       "temporal-selection evidence - see eligibility_aware_previous_event "
                       "and permutation_null for the composition-adjusted baseline"),
        },
        "eligibility_aware_previous_event": {
            "note": ("'eligible' = parsed wrong prediction at t>=2 with E_{t-1} != E_t, where "
                     "matching the previous event is non-trivial (a uniform choice among the "
                     "2 non-GT types would hit at 0.5). Rows where E_{t-1} == E_t can never "
                     "be hit by a wrong prediction and are excluded. Observed rates are "
                     "reported against the within-trajectory event-order permutation null "
                     "below; no interpretation is made here"),
            "n_parsed_wrong_t_ge_2": len(parsed_wrong_t2),
            "n_eligible": len(eligible),
            "n_prev_event_repeats_gt": len(parsed_wrong_t2) - len(eligible),
            "n_hit_prev_among_eligible": len(hit_prev_eligible),
            "hit_prev_rate_among_eligible": (len(hit_prev_eligible) / len(eligible)) if eligible else None,
            "hit_prev_rate_among_parsed_wrong_t_ge_2": (len(hit_prev) / len(parsed_wrong_t2)) if parsed_wrong_t2 else None,
        },
        "permutation_null": temporal_permutation_null(rows, n_perms=1000, seed=0),
        "temporal_confusion_matrix": confusion_matrix(rows, "gt_event", "event_pred"),
        "temporal_confusion_matrix_by_step": {
            str(t): confusion_matrix([r for r in rows if r["t"] == t], "gt_event", "event_pred")
            for t in range(1, 6)
        },
    }
    return rows, summary


def temporal_permutation_null(rows: list[dict], n_perms: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Null distribution for previous-event / history-hit rates under shuffled event order.

    Each permutation shuffles the 5 GT event types WITHIN each trajectory (preserving
    the per-trajectory type composition and all model predictions) and recomputes the
    rates as if the shuffled order were the true event sequence. This breaks the
    temporal order of events while keeping type composition fixed: the null
    distribution is what the observed rates look like when the model's errors carry no
    information about event order. Distributions are reported only - no conclusion is
    drawn in this module.
    """
    per_video: dict[str, dict[int, Any]] = {}
    seqs: dict[str, list[str]] = {}
    for r in rows:
        per_video.setdefault(r["video"], {})[r["t"]] = r["event_pred"]
        if r["video"] not in seqs:
            seqs[r["video"]] = [r[f"E{j}_gt"] for j in range(1, 6)]

    dist: dict[str, list[float]] = {
        "prev_hit_rate_wrong_t_ge_2": [],
        "prev_hit_rate_eligible": [],
        "history_hit_rate_wrong": [],
    }
    rng = random.Random(seed)
    for _ in range(n_perms):
        n_wrong = n_wrong_t2 = n_elig = 0
        hit_prev = hit_prev_elig = hit_hist = 0
        for v, preds in per_video.items():
            perm = list(seqs[v])
            rng.shuffle(perm)
            for t in range(1, 6):
                pred = preds[t]
                if pred is None:
                    continue
                gt_t = perm[t - 1]
                if pred == gt_t:
                    continue
                n_wrong += 1
                if pred in perm[: t - 1]:
                    hit_hist += 1
                if t >= 2:
                    n_wrong_t2 += 1
                    elig = perm[t - 2] != gt_t
                    if elig:
                        n_elig += 1
                    if pred == perm[t - 2]:
                        hit_prev += 1
                        if elig:
                            hit_prev_elig += 1
        dist["prev_hit_rate_wrong_t_ge_2"].append(hit_prev / n_wrong_t2 if n_wrong_t2 else 0.0)
        dist["prev_hit_rate_eligible"].append(hit_prev_elig / n_elig if n_elig else 0.0)
        dist["history_hit_rate_wrong"].append(hit_hist / n_wrong if n_wrong else 0.0)

    def _rate(numerator: int, denominator: int):
        return numerator / denominator if denominator else None

    observed = {
        "prev_hit_rate_wrong_t_ge_2": _rate(
            sum(1 for r in rows if r["error_class"] == "equals_E_t_minus_1"),
            sum(1 for r in rows if r["event_pred"] is not None and not r["event_correct"] and r["t"] >= 2)),
        "prev_hit_rate_eligible": _rate(
            sum(1 for r in rows if r["prev_eligible"] and r["error_class"] == "equals_E_t_minus_1"),
            sum(1 for r in rows if r["prev_eligible"])),
        "history_hit_rate_wrong": _rate(
            sum(1 for r in rows if r["error_class"] in ("equals_E_t_minus_1", "equals_any_earlier_event")),
            sum(1 for r in rows if r["event_pred"] is not None and not r["event_correct"])),
    }

    out: dict[str, Any] = {
        "procedure": ("for each of the %d permutations, shuffle the 5 GT event types within "
                      "every trajectory (type composition and predictions unchanged) and "
                      "recompute the rates as if the shuffled order were ground truth; breaks "
                      "event order, preserves per-trajectory type composition" % n_perms),
        "n_permutations": n_perms,
        "seed": seed,
        "note": "observed vs null reported only; no interpretation in this module",
    }
    for name, values in dist.items():
        arr = np.asarray(values)
        out[name] = {
            "observed": observed[name],
            "null_mean": float(arr.mean()),
            "null_std": float(arr.std()),
            "null_p05": float(np.percentile(arr, 5)),
            "null_p95": float(np.percentile(arr, 95)),
            "null_max": float(arr.max()),
            "observed_percentile": float((arr <= observed[name]).mean()) if observed[name] is not None else None,
            "one_sided_p_ge": float((arr >= observed[name]).mean()) if observed[name] is not None else None,
            "distribution": [float(x) for x in arr],
        }
    return out


# --------------------------------------------------------------------------
# 3. State inertia / recovery audit
# --------------------------------------------------------------------------

def build_state_dynamics(
    state_rows: list[dict], traj_rows: list[dict], gt_by_video: dict[str, dict]
) -> tuple[list[dict], list[dict], dict]:
    by_video: dict[str, dict[int, dict]] = {}
    for r in state_rows:
        by_video.setdefault(r["video"], {})[r["t"]] = r

    dyn_rows: list[dict] = []
    n_steps_total = 0
    for video, steps in by_video.items():
        gt = gt_by_video[video]
        for t in range(1, 6):
            n_steps_total += 1
            prev, cur = steps.get(t - 1), steps.get(t)
            if prev is None or cur is None:
                continue
            gt_prev = prev["gt_state"]
            gt_cur = cur["gt_state"]
            if gt_cur == gt_prev:
                continue  # not a true state transition
            pred_prev = prev["state_prediction"]
            pred_cur = cur["state_prediction"]
            correct = cur["state_correct"]
            eq_gt_prev = safe_eq(pred_cur, gt_prev)
            eq_pred_prev = safe_eq(pred_cur, pred_prev)
            if correct:
                error_class = ""
            elif eq_gt_prev:
                error_class = "stale_ground_truth_state"
            elif eq_pred_prev:
                error_class = "prediction_inertia"
            else:
                error_class = "other_wrong"
            dyn_rows.append({
                "trajectory_id": video.rsplit(".", 1)[0],
                "video": video,
                "t": t,
                "gt_prev_state": gt_prev,
                "gt_state": gt_cur,
                "prev_state_pred": pred_prev,
                "state_pred": pred_cur,
                "state_correct": correct,
                "pred_eq_gt_prev": eq_gt_prev,
                "pred_eq_pred_prev": eq_pred_prev,
                "error_class": error_class,
            })

    # trajectory-level dynamics from the wide trajectory rows
    traj_dyn: list[dict] = []
    for tr in traj_rows:
        correct_seq = [bool(tr[f"S{t}_correct"]) for t in range(6)]
        trans = Counter()
        for t in range(1, 6):
            a, b = correct_seq[t - 1], correct_seq[t]
            trans["correct_to_correct" if a and b else
                  "wrong_to_wrong" if not a and not b else
                  "correct_to_wrong" if a else "wrong_to_correct"] += 1
        first_failure = next((t for t in range(6) if not correct_seq[t]), None)
        first_recovery = next((t for t in range(1, 6) if not correct_seq[t - 1] and correct_seq[t]), None)
        failed = first_failure is not None
        recovered = first_recovery is not None
        pattern = ("all_correct" if not failed else
                   "failed_with_recovery" if recovered else "failed_no_recovery")
        traj_dyn.append({
            "trajectory_id": tr["sample_id"],
            "video": tr["video"],
            "n_correct_to_wrong": trans["correct_to_wrong"],
            "n_wrong_to_correct": trans["wrong_to_correct"],
            "n_wrong_to_wrong": trans["wrong_to_wrong"],
            "first_failure_step": first_failure,
            "first_recovery_step": first_recovery,
            "recovered_after_failure": bool(failed and recovered),
            "pattern": pattern,
        })

    wrong = [r for r in dyn_rows if not r["state_correct"]]
    class_counts = Counter(r["error_class"] for r in wrong)
    n_failed = sum(d["first_failure_step"] is not None for d in traj_dyn)
    n_recovered = sum(d["recovered_after_failure"] for d in traj_dyn)

    summary = {
        "n_steps_total": n_steps_total,
        "n_transition_steps": len(dyn_rows),
        "n_non_transition_steps_skipped": n_steps_total - len(dyn_rows),
        "transition_steps_accuracy": (sum(r["state_correct"] for r in dyn_rows) / len(dyn_rows))
        if dyn_rows else None,
        "stale_ground_truth_state_rate": (sum(r["pred_eq_gt_prev"] for r in dyn_rows) / len(dyn_rows))
        if dyn_rows else None,
        "prediction_inertia_rate": (sum(r["pred_eq_pred_prev"] for r in dyn_rows) / len(dyn_rows))
        if dyn_rows else None,
        "error_class_counts": dict(class_counts),
        "error_class_counts_note": "mutually exclusive; stale GT takes priority over inertia",
        "trajectory_dynamics": {
            "n_trajectories": len(traj_dyn),
            "n_correct_to_wrong": sum(d["n_correct_to_wrong"] for d in traj_dyn),
            "n_wrong_to_correct": sum(d["n_wrong_to_correct"] for d in traj_dyn),
            "n_wrong_to_wrong": sum(d["n_wrong_to_wrong"] for d in traj_dyn),
            "first_failure_step_distribution": dict(sorted(
                ((str(k), v) for k, v in
                 Counter(("none" if d["first_failure_step"] is None else d["first_failure_step"])
                         for d in traj_dyn).items()),
                key=lambda kv: (kv[0] == "none", kv[0]))),
            "first_recovery_step_distribution": dict(sorted(
                ((str(k), v) for k, v in
                 Counter(("none" if d["first_recovery_step"] is None else d["first_recovery_step"])
                         for d in traj_dyn).items()),
                key=lambda kv: (kv[0] == "none", kv[0]))),
            "pattern_counts": dict(Counter(d["pattern"] for d in traj_dyn)),
            "n_failed": n_failed,
            "n_recovered_after_failure": n_recovered,
            "recovery_rate": (n_recovered / n_failed) if n_failed else None,
            "recovery_rate_definition": "trajectories with >=1 wrong->correct transition / "
                                        "trajectories with >=1 failure (a failure is not a permanent loss)",
        },
    }
    return dyn_rows, traj_dyn, summary


# --------------------------------------------------------------------------
# 4. Joint mechanism candidates (manifest for downstream hidden-state analysis)
# --------------------------------------------------------------------------

def build_mechanism_manifest(
    event_rows: list[dict], state_rows: list[dict], gt_by_video: dict[str, dict],
    video_dir: Path,
) -> tuple[list[dict], dict]:
    state_by: dict[str, dict[int, dict]] = {}
    for r in state_rows:
        state_by.setdefault(r["video"], {})[r["t"]] = r

    manifest: list[dict] = []
    for r in event_rows:
        video = r["video"]
        t = r["t"]
        gt = gt_by_video[video]
        prev_state_row = state_by[video][t - 1]
        cur_state_row = state_by[video][t]
        joint = (f"event_{'correct' if r['event_correct'] else 'wrong'}"
                 f"_state_{'correct' if r['state_correct'] else 'wrong'}")
        m: dict[str, Any] = {
            "trajectory_id": video.rsplit(".", 1)[0],
            "t": t,
            "video": video,
            "prefix_path": str(video_dir / video),
            "frame_start": r["frame_range"][0],
            "frame_end": r["frame_range"][1],
            "clip_frames": r["clip_frames"],
            "sampled_frames": r["sampled_frames"],
            "sample_fps": r["sample_fps"],
            "initial_state": POSITION_NAMES[gt["initial_pos"]],
            "n_swaps_shown": t,
            "gt_event": r["event_gt"],
            "event_pred": r["event_prediction"],
            "event_correct": r["event_correct"],
            "gt_prev_state": prev_state_row["gt_state"],
            "gt_state": cur_state_row["gt_state"],
            "prev_state_pred": prev_state_row["state_prediction"],
            "state_pred": cur_state_row["state_prediction"],
            "state_correct": r["state_correct"],
            "joint_class": joint,
        }
        m["is_transition"] = bool(m["gt_state"] != m["gt_prev_state"])
        m["prev_state_correct"] = bool(safe_eq(m["prev_state_pred"], m["gt_prev_state"]))
        # canonical revision: the model actually saw the swap (event correct) and knew
        # the state before it (previous prediction correct) on a TRUE transition -
        # the cleanest sample for "did the model update its state after the swap".
        m["clean_revision"] = bool(m["is_transition"] and m["event_correct"] and m["prev_state_correct"])
        # clean maintenance control: same filters but no true transition (GT state
        # unchanged), i.e. the model should simply maintain its correct state.
        m["clean_maintenance"] = bool((not m["is_transition"]) and m["event_correct"]
                                      and m["prev_state_correct"])
        manifest.append(m)

    counts = Counter(r["joint_class"] for r in manifest)
    flagged = [r for r in manifest if r["joint_class"] == "event_correct_state_wrong"]
    summary = {
        "n_rows": len(manifest),
        "joint_class_counts": {c: counts.get(c, 0) for c in JOINT_CLASSES},
        "event_correct_state_wrong_manifest": {
            "n": len(flagged),
            "rate_of_event_correct": (len(flagged) / sum(1 for r in manifest if r["event_correct"]))
            if any(r["event_correct"] for r in manifest) else None,
            "n_trajectories": len({r["trajectory_id"] for r in flagged}),
            "manifest_columns": MANIFEST_CSV_COLUMNS,
        },
    }
    return manifest, summary


# --------------------------------------------------------------------------
# 4b. Canonical revision audit (clean state-update samples)
# --------------------------------------------------------------------------

def build_canonical_revision(manifest: list[dict]) -> tuple[list[dict], dict]:
    """Canonical revision rows: true GT transitions where the model both saw the
    event correctly (event_correct_t) and knew the previous state
    (state_pred_{t-1} == GT_{t-1}). On these rows the only thing left to do is
    UPDATE the state after the swap.

    Control: clean maintenance rows - same two filters but GT_t == GT_{t-1} (the
    swap did not move the ball), where the model should merely maintain its state.
    Both current-state accuracies are reported for comparison.
    """
    rev_rows: list[dict] = []
    for r in manifest:
        if not r["clean_revision"]:
            continue
        success = bool(r["state_correct"])
        rev_rows.append({
            "trajectory_id": r["trajectory_id"],
            "t": r["t"],
            "video": r["video"],
            "gt_event": r["gt_event"],
            "event_pred": r["event_pred"],
            "gt_prev_state": r["gt_prev_state"],
            "prev_state_pred": r["prev_state_pred"],
            "gt_state": r["gt_state"],
            "state_pred": r["state_pred"],
            "state_correct": success,
            "stale_failure": (not success) and safe_eq(r["state_pred"], r["gt_prev_state"]),
            "prefix_path": r["prefix_path"],
        })

    n_rev = len(rev_rows)
    n_success = sum(bool(r["state_correct"]) for r in rev_rows)
    n_failure = n_rev - n_success
    n_stale = sum(bool(r["stale_failure"]) for r in rev_rows)

    maint = [r for r in manifest if r["clean_maintenance"]]
    n_maint = len(maint)
    n_maint_correct = sum(bool(r["state_correct"]) for r in maint)

    transition = [r for r in manifest if r["is_transition"]]
    rev_success_rate = (n_success / n_rev) if n_rev else None
    maint_rate = (n_maint_correct / n_maint) if n_maint else None

    summary = {
        "definition": {
            "clean_revision": "true transition (GT_t != GT_{t-1}) AND event_correct_t "
                              "AND state_pred_{t-1} == GT_{t-1}",
            "clean_revision_success": "clean revision with pred_t == GT_t",
            "clean_revision_failure": "clean revision with pred_t != GT_t",
            "canonical_stale_failure": "clean revision failure with pred_t == pred_{t-1} == GT_{t-1}",
            "clean_maintenance_control": "non-transition (GT_t == GT_{t-1}) AND event_correct_t "
                                         "AND state_pred_{t-1} == GT_{t-1}",
        },
        "n_transition_rows": len(transition),
        "n_transition_event_correct": sum(1 for r in transition if r["event_correct"]),
        "clean_revision_n": n_rev,
        "clean_revision_success": {"n": n_success, "rate": rev_success_rate},
        "clean_revision_failure": {"n": n_failure,
                                   "rate": (n_failure / n_rev) if n_rev else None},
        "canonical_stale_failure": {
            "n": n_stale,
            "rate_of_clean_revision": (n_stale / n_rev) if n_rev else None,
            "rate_of_clean_revision_failure": (n_stale / n_failure) if n_failure else None,
        },
        "clean_maintenance_control": {
            "n": n_maint,
            "n_current_state_correct": n_maint_correct,
            "current_state_accuracy": maint_rate,
        },
        "revision_vs_maintenance": {
            "clean_revision_success_rate": rev_success_rate,
            "clean_maintenance_accuracy": maint_rate,
            "difference": (rev_success_rate - maint_rate)
                          if (rev_success_rate is not None and maint_rate is not None) else None,
        },
    }
    return rev_rows, summary


# --------------------------------------------------------------------------
# 1. One-swap correction record
# --------------------------------------------------------------------------

ONE_SWAP_OFFICIAL_RECORD = {
    "status": "official_1swap_videos_not_found",
    "finding": ("No standalone official 1-swap tracking videos exist in VET-Bench; the "
                "previous one_swap_audit rows attached the 5-swap final-video prediction to "
                "each of the 5 isolated-swap clips of the same video"),
    "evidence": [
        "dataset/vetbench/cup/cup.json: 50 videos, every entry game_settings.swap_count=5, "
        "372 frames, 5 intermediate arrangements",
        "dataset/vetbench/README.md: Cup Game = 50 videos x 5 swaps; Card Game = 50 videos x 5 swaps",
        "dataset/vetbench/.hfd/manifest (upstream tiedong/vetbench @ sha 258185944dba3df09145bb41f0721127e4e19575, "
        "lastModified 2026-04-02): 108 files total = 50 cup + 50 card mp4s + metadata only; "
        "no 1-swap subset exists upstream",
    ],
    "deprecated_statistic": {
        "name": "one_swap_audit.p_video_correct_given_isolated_correct (row-level)",
        "previously_reported_value": 0.39805825242718446,
        "reason": ("the full-sequence video prediction is a single result per video, repeated "
                   "across its 5 isolated-swap rows; the 250 rows are not independent samples "
                   "of the conditional, so the row-level probability is invalid"),
        "action": ("removed from build_one_swap_summary in run_state_rev_audit.py; "
                   "outputs/vetbench/audit/audit_summary.json left untouched as the original run record"),
    },
    "retained_valid_statistic": {
        "name": "one_swap_audit.p_video_correct_given_all_isolated_correct (video-level)",
        "value": 0.34782608695652173,
        "n_videos": 23,
    },
    "one_swap_audit_csv_produced": False,
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

TEMPORAL_CSV_COLUMNS = [
    "trajectory_id", "video", "t", "gt_event", "event_pred", "event_correct", "error_class",
    "E1_gt", "E2_gt", "E3_gt", "E4_gt", "E5_gt",
    "history_match_steps", "gt_repeated_in_prefix", "prev_eligible",
]

DYNAMICS_CSV_COLUMNS = [
    "trajectory_id", "video", "t",
    "gt_prev_state", "gt_state", "prev_state_pred", "state_pred", "state_correct",
    "pred_eq_gt_prev", "pred_eq_pred_prev", "error_class",
]

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the corrected behavior audit v2 (temporal event, state dynamics, "
                    "mechanism manifest) from the existing 50-video audit outputs.")
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--meta", type=Path, default=CUP_META)
    parser.add_argument("--video-dir", type=Path, default=CUP_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    entries = load_metadata()
    gt_by_video = {e["video"]: derive_ground_truth(e) for e in entries}
    state_rows = load_jsonl(args.audit_dir / "prefix_state.jsonl")
    event_rows = load_jsonl(args.audit_dir / "in_context_event.jsonl")
    traj_rows = load_jsonl(args.audit_dir / "trajectories.jsonl")
    print(f"Loaded audit outputs: state={len(state_rows)} event={len(event_rows)} "
          f"trajectories={len(traj_rows)} (GT derived and validated for {len(entries)} videos)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    temporal_rows, temporal_summary = build_temporal_event(event_rows, gt_by_video)
    dyn_rows, traj_dyn, dyn_summary = build_state_dynamics(state_rows, traj_rows, gt_by_video)
    manifest_rows, manifest_summary = build_mechanism_manifest(
        event_rows, state_rows, gt_by_video, args.video_dir)
    canonical_rows, canonical_summary = build_canonical_revision(manifest_rows)

    write_csv(args.out_dir / "temporal_event_audit.csv", temporal_rows, TEMPORAL_CSV_COLUMNS)
    write_csv(args.out_dir / "state_dynamics_audit.csv", dyn_rows, DYNAMICS_CSV_COLUMNS)
    write_csv(args.out_dir / "mechanism_candidates.csv", manifest_rows, MANIFEST_CSV_COLUMNS)
    write_csv(args.out_dir / "canonical_revision_candidates.csv", canonical_rows, CANONICAL_CSV_COLUMNS)
    # trajectory dynamics are per-trajectory; keep them in the summary + a small JSONL
    with open(args.out_dir / "trajectory_dynamics.jsonl", "w", encoding="utf-8") as f:
        for d in traj_dyn:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    summary = {
        "one_swap_official_audit": ONE_SWAP_OFFICIAL_RECORD,
        "temporal_event": temporal_summary,
        "state_dynamics": dyn_summary,
        "mechanism_candidates": manifest_summary,
        "canonical_revision": canonical_summary,
        "inputs": {
            "audit_dir": str(args.audit_dir),
            "cup_meta": str(args.meta),
            "model": "Qwen3-VL-8B-Instruct (results from the previous deterministic audit run)",
            "files": {
                "prefix_state.jsonl": len(state_rows),
                "in_context_event.jsonl": len(event_rows),
                "trajectories.jsonl": len(traj_rows),
            },
        },
        "outputs": {
            "temporal_event_audit.csv": len(temporal_rows),
            "state_dynamics_audit.csv": len(dyn_rows),
            "mechanism_candidates.csv": len(manifest_rows),
            "canonical_revision_candidates.csv": len(canonical_rows),
            "trajectory_dynamics.jsonl": len(traj_dyn),
        },
    }
    (args.out_dir / "behavior_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nbehavior_audit_summary.json written:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs in {args.out_dir}: temporal_event_audit.csv "
          f"({len(temporal_rows)} rows), state_dynamics_audit.csv ({len(dyn_rows)} rows), "
          f"mechanism_candidates.csv ({len(manifest_rows)} rows), trajectory_dynamics.jsonl "
          f"({len(traj_dyn)} rows), behavior_audit_summary.json")


if __name__ == "__main__":
    main()
