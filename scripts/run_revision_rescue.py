"""Experiment 2 — Paired prior/event rescue intervention (fixed factorial).

Question
--------
The stale answer is S_{t-1}: the pre-event state. Which bottleneck produces
it?

  1. prior-state accessibility  -> give the prior explicitly (prior_only)
  2. event bottleneck           -> give the event explicitly (event_only)
  3. composition / readout      -> give both (prior_and_event)

This is a BEHAVIORAL fixed-factorial intervention on the original
Transformers-aligned state question. Wording is PRE-FIXED (no per-sample
prompt tuning), the final question is unchanged, video sampling is unchanged,
and the GT current state is NEVER stated as the ball's position.

Conditions (same sample, same clip, greedy deterministic decoding)
------------------------------------------------------------------
  A baseline          original state-question prompt, nothing added
  B prior_only        + "Immediately before the latest swap, the ball was
                      under the cup currently at the {GT_PREV} position."
  C event_only        + "In the latest swap, the cups currently at the {A}
                      and {B} positions exchanged places."
  D prior_and_event   both sentences (prior first, then event)

The two sentences are inserted at a FIXED position: immediately before the
final question sentence ("The ball is under the cup that is currently at one
of the three positions. ..."), which itself is left byte-identical.

Design caveat (documented, by construction of the user-specified wording):
the event operand names the two swapped positions {A, B}. For clean-revision
rows the ball moved, so S_t is necessarily one of {A, B} and its POSITION
NAME appears in the event sentence. The sentence states the SWAP, not the
ball's location (it never mentions the ball). The "no GT current state"
safety check therefore asserts: (i) the added text never states the ball's
position as GT S_t, and (ii) the event sentence contains no "ball" clause.

Samples
-------
  primary: 26 canonical stale failures (Transformers-aligned mask:
           clean_revision AND not state_correct AND state_pred == gt_prev)
  controls: clean maintenance (23); other aligned revision failures (17);
           same-step (t=2..5) non-canonical "rest" control
  (a same-step matched-success control interface is reserved but NOT forced,
   since all 17 successes sit at t=1 where the prior is verbatim in the
   prompt - see the t=1 caveat in the retention analysis)

Modes
-----
  --mode build   (default, this round): render all prompts, run all safety
                 checks, write manifest + skeleton results. NO model.
  --mode run     (future): load the model, run greedy inference + forced-
                 choice margins for all conditions, write results/summary.

Outputs (default outputs/vetbench/stale_origin_analysis_v1/):
  build: revision_rescue_manifest.jsonl, revision_rescue_build_summary.json,
         revision_rescue_results.csv (skeleton)
  run:   revision_rescue_results.csv (filled), revision_rescue_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR
from run_state_rev_audit import (
    SYSTEM_PROMPT,
    state_question_text,
)

ROOT = Path(__file__).resolve().parent.parent
BEHAVIOR_CSV = (DEFAULT_OUTPUT_DIR / "vetbench" / "composition_analysis_v1"
                / "transformers_behavior.csv")
OUT_DIR_DEFAULT = DEFAULT_OUTPUT_DIR / "vetbench" / "stale_origin_analysis_v1"

CONDITIONS = ("baseline", "prior_only", "event_only", "prior_and_event")

# ---- PRE-FIXED wording (no per-sample optimization) ------------------------
PRIOR_TMPL = ("Immediately before the latest swap, the ball was under the "
              "cup currently at the {PREV} position.")
EVENT_TMPL = ("In the latest swap, the cups currently at the {A} and {B} "
              "positions exchanged places.")
# fixed insertion point: immediately before this final-question sentence
INSERT_ANCHOR = ("The ball is under the cup that is currently at one of the "
                 "three positions.")

# generation settings replicated from the aligned Transformers run
# (outputs/vetbench/transformers_behavior_v1/audit_config.json)
GEN_MAX_NEW_TOKENS = 32
GEN_THINKING = False
SAMPLE_FPS = 8.0


def T(x: str) -> bool:
    return str(x).strip().lower() == "true"


def render_prompts(row: dict) -> dict[str, str]:
    """Render the 4 conditions for one behavior row. Deterministic."""
    baseline = state_question_text(row["initial_state"],
                                   int(row["n_swaps_shown"]))
    a, b = row["gt_event"].split(" and ")
    prior = PRIOR_TMPL.format(PREV=row["gt_prev_state"])
    event = EVENT_TMPL.format(A=a, B=b)

    def insert(sentences: str) -> str:
        assert baseline.count(INSERT_ANCHOR) == 1, "anchor not unique"
        return baseline.replace(INSERT_ANCHOR,
                                sentences + " " + INSERT_ANCHOR)

    return {
        "baseline": baseline,
        "prior_only": insert(prior),
        "event_only": insert(event),
        "prior_and_event": insert(prior + " " + event),
        "_prior_sentence": prior,
        "_event_sentence": event,
    }


def validate_row(row: dict, pr: dict[str, str]) -> dict[str, bool]:
    """All safety checks for one rendered sample. All must pass."""
    gt_state, gt_prev = row["gt_state"], row["gt_prev_state"]
    checks: dict[str, bool] = {}

    # 1. baseline byte-identical to the original state question
    checks["baseline_matches_reference"] = (
        pr["baseline"]
        == state_question_text(row["initial_state"],
                               int(row["n_swaps_shown"])))
    # 2. final question byte-identical in every condition
    q0 = pr["baseline"].split(INSERT_ANCHOR, 1)[1]
    checks["question_unchanged"] = all(
        pr[c].split(INSERT_ANCHOR, 1)[1] == q0
        for c in CONDITIONS if pr[c] != pr["baseline"])
    # 3. added delta is exactly the fixed sentence(s)
    head0, _ = pr["baseline"].split(INSERT_ANCHOR, 1)
    for c, expect in (("prior_only", pr["_prior_sentence"]),
                      ("event_only", pr["_event_sentence"]),
                      ("prior_and_event",
                       pr["_prior_sentence"] + " " + pr["_event_sentence"])):
        head1, _ = pr[c].split(INSERT_ANCHOR, 1)
        checks[f"delta_exact_{c}"] = (head1 == head0 + expect + " ")
    # 4. the added text never states the ball's position as GT current state.
    # NOTE: for rows where S_{t-1} == S_t (maintenance / non-transition
    # rest rows) the prior sentence necessarily names the current position -
    # that is an unavoidable identity of the no-change control, flagged via
    # prior_equals_current and excluded from this check (the check applies
    # to all transition rows, incl. all 26 stale).
    prior_equals_current = (gt_prev == gt_state)
    for c in CONDITIONS[1:]:
        if c == "prior_only" and prior_equals_current:
            checks[f"no_gt_current_claim_{c}"] = True
            continue
        head1, _ = pr[c].split(INSERT_ANCHOR, 1)
        delta = head1[len(head0):]
        # for prior_and_event on prior==current rows, strip the prior
        # sentence (its claim is the flagged identity) before checking
        if c == "prior_and_event" and prior_equals_current:
            delta = delta.replace(pr["_prior_sentence"], "")
        pat = re.compile(rf"ball[^.]*at the {gt_state} position",
                         re.IGNORECASE)
        checks[f"no_gt_current_claim_{c}"] = pat.search(delta) is None
    checks["_info_prior_equals_current"] = prior_equals_current
    # 5. event sentence states the swap, never the ball
    checks["event_no_ball_clause"] = "ball" not in pr["_event_sentence"]
    # 6. event operands match GT event (ordered)
    a, b = row["gt_event"].split(" and ")
    checks["event_operands_match_gt"] = (
        pr["_event_sentence"] == EVENT_TMPL.format(A=a, B=b))
    # 7. prior sentence states exactly GT previous state
    checks["prior_states_gt_prev"] = (
        pr["_prior_sentence"] == PRIOR_TMPL.format(PREV=gt_prev))
    # 8. for revision rows the prior differs from the current state
    if T(row["clean_revision"]):
        checks["prev_differs_current"] = gt_prev != gt_state
    return checks


def load_samples(behavior_csv: Path, controls: set[str]) -> list[dict]:
    rows = list(csv.DictReader(open(behavior_csv, newline="")))
    samples: list[dict] = []

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

    for r in rows:
        g = group_of(r)
        if g in ("stale",) or (g in controls):
            s = dict(r)
            s["group"] = g
            samples.append(s)
    return samples


def write_manifest(samples: list[dict], out_dir: Path,
                   build_summary: dict) -> int:
    n = 0
    with open(out_dir / "revision_rescue_manifest.jsonl", "w") as f:
        for s in samples:
            pr = render_prompts(s)
            checks = validate_row(s, pr)
            head0, _ = pr["baseline"].split(INSERT_ANCHOR, 1)
            for c in CONDITIONS:
                head1, _ = pr[c].split(INSERT_ANCHOR, 1)
                rec = {
                    "sample_id": f"{s['trajectory_id']}_t{s['t']}",
                    "trajectory_id": s["trajectory_id"], "t": s["t"],
                    "group": s["group"], "condition": c,
                    "prompt_text": pr[c],
                    "delta_text": ("" if c == "baseline"
                                   else head1[len(head0):].strip()),
                    "video": s["video"], "prefix_path": s["prefix_path"],
                    "frame_start": int(s["frame_start"]),
                    "frame_end": int(s["frame_end"]),
                    "clip_frames": int(s["clip_frames"]),
                    "sampled_frames": int(s["sampled_frames"]),
                    "sample_fps": float(s["sample_fps"]),
                    "initial_state": s["initial_state"],
                    "n_swaps_shown": int(s["n_swaps_shown"]),
                    "gt_prev_state": s["gt_prev_state"],
                    "gt_state": s["gt_state"], "gt_event": s["gt_event"],
                    "native_state_pred": s["state_pred"],
                    "native_state_correct": T(s["state_correct"]),
                    "native_event_correct": T(s["event_correct"]),
                    "validation": checks,
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def write_results_skeleton(samples: list[dict], out_dir: Path) -> None:
    cols = ["sample_id", "trajectory_id", "t", "group", "condition",
            "pred_state", "state_correct",
            "margin_gt_state_minus_gt_prev", "baseline_reproduced"]
    with open(out_dir / "revision_rescue_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in samples:
            for c in CONDITIONS:
                w.writerow([f"{s['trajectory_id']}_t{s['t']}",
                            s["trajectory_id"], s["t"], s["group"], c,
                            "", "", "", ""])


def build(args: argparse.Namespace) -> dict:
    samples = load_samples(args.behavior, set(args.controls.split(",")))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_checks: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    per_sample: list[dict] = []
    for s in samples:
        pr = render_prompts(s)
        checks = validate_row(s, pr)
        sid = f"{s['trajectory_id']}_t{s['t']}"
        pass_checks = {k: v for k, v in checks.items()
                       if not k.startswith("_info")}
        per_sample.append({"sample_id": sid, "group": s["group"],
                           "all_pass": all(pass_checks.values()),
                           "failed": [k for k, v in pass_checks.items()
                                      if not v]})
        for k, v in pass_checks.items():
            all_checks[k].append((sid, v))

    n_rows = write_manifest(samples, out_dir, None)
    write_results_skeleton(samples, out_dir)

    by_group: dict[str, int] = defaultdict(int)
    for s in samples:
        by_group[s["group"]] += 1
    summary = {
        "mode": "build",
        "config": {
            "behavior": str(args.behavior),
            "controls": sorted(args.controls.split(",")),
            "conditions": list(CONDITIONS),
            "wording": {"prior": PRIOR_TMPL, "event": EVENT_TMPL,
                        "insert_before": INSERT_ANCHOR},
            "generation": {"max_new_tokens": GEN_MAX_NEW_TOKENS,
                           "thinking": GEN_THINKING,
                           "sample_fps": SAMPLE_FPS,
                           "decoding": "greedy (do_sample=False)"},
        },
        "samples": {g: n for g, n in sorted(by_group.items())},
        "n_manifest_rows": n_rows,
        "checks": {k: {"n_pass": sum(1 for _, v in vs if v),
                       "n_total": len(vs),
                       "failed": [sid for sid, v in vs if not v]}
                   for k, vs in sorted(all_checks.items())},
        "all_samples_pass": all(p["all_pass"] for p in per_sample),
        "design_caveat_event_operand": (
            "the event sentence names the two swapped positions; for "
            "clean-revision rows the ball moved, so the position NAME of the "
            "GT current state is one of the two named positions. The "
            "sentence states the swap, never the ball's location (no "
            "'ball' clause); see validation checks event_no_ball_clause and "
            "no_gt_current_claim_*."),
        "n_prior_equals_current_rows": sum(
            1 for s in samples if s["gt_prev_state"] == s["gt_state"]),
        "caveat_prior_equals_current": (
            "on no-change rows (maintenance and non-transition rest "
            "controls) S_{t-1} == S_t by definition, so the fixed prior "
            "sentence necessarily names the current position. The strict "
            "no-GT-current claim check therefore applies to all transition "
            "rows (all 26 stale + other failures + successes); on "
            "no-change rows the identity is flagged via the manifest field "
            "_info_prior_equals_current."),
        "t1_caveat": (
            "t=1 priors are the verbatim initial state already present in "
            "the prompt; all 17 successes sit at t=1, so no matched-success "
            "control is forced (interface reserved)."),
    }
    with open(out_dir / "revision_rescue_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'revision_rescue_manifest.jsonl'} ({n_rows} rows)")
    print(f"Wrote {out_dir / 'revision_rescue_results.csv'} (skeleton)")
    print(f"Wrote {out_dir / 'revision_rescue_build_summary.json'}")
    for g, n in sorted(by_group.items()):
        print(f"  group {g}: {n}")
    print(f"ALL CHECKS PASS: {summary['all_samples_pass']}")
    for k, v in summary["checks"].items():
        if v["n_pass"] != v["n_total"]:
            print(f"  FAILED {k}: {v}")
    return summary


# ---------------------------------------------------------------------------
# Future run path (NOT executed this round; requires GPU + model)
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    from run_behavior_screening import (
        load_model_and_processor,
        run_inference,
        score_candidates,
    )
    from run_vetbench_screening import (
        parse_tracking_option,
        sample_clip,
        video_processor_kwargs,
    )

    model_dir = Path(args.model_dir)
    model, processor = load_model_and_processor(model_dir)
    device = "cuda:0"
    manifest = [json.loads(l) for l in
                open(args.out_dir / "revision_rescue_manifest.jsonl")]
    results = []
    for i, rec in enumerate(manifest):
        clip = sample_clip(Path(rec["prefix_path"]), rec["frame_start"],
                           rec["frame_end"])
        pkw = video_processor_kwargs(clip, rec["sample_fps"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video", "video": clip},
                {"type": "text", "text": rec["prompt_text"]},
            ]},
        ]
        raw = run_inference(model, processor, messages, device,
                            GEN_MAX_NEW_TOKENS, GEN_THINKING,
                            processor_kwargs=pkw)
        pred = parse_tracking_option(raw)
        scores = score_candidates(model, processor, messages,
                                  ("Left", "Middle", "Right"), device,
                                  processor_kwargs=pkw)
        margin = (float(scores[rec["gt_state"]]
                        - scores[rec["gt_prev_state"]])
                  if rec["gt_state"] in scores
                  and rec["gt_prev_state"] in scores else None)
        results.append({
            "sample_id": rec["sample_id"], "trajectory_id":
            rec["trajectory_id"], "t": rec["t"], "group": rec["group"],
            "condition": rec["condition"], "pred_state": pred,
            "state_correct": pred == rec["gt_state"],
            "margin_gt_state_minus_gt_prev": margin,
            "baseline_reproduced": (
                (pred == rec["native_state_pred"])
                if rec["condition"] == "baseline" else ""),
        })
        if (i + 1) % 20 == 0:
            print(f"run: {i + 1}/{len(manifest)}", flush=True)

    # baseline reproduction check vs the aligned behavior
    nat = {r["sample_id"]: r for r in results if r["condition"] == "baseline"}
    repro = [r for r in nat if r["state_correct"]]
    with open(args.out_dir / "revision_rescue_results.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    base = [r for r in results if r["condition"] == "baseline"]
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by[r["group"]][r["condition"]].append(r)
    summary = {
        "mode": "run",
        "n_rows": len(results),
        "baseline_reproduction": {
            "n": len(base),
            "n_state_correct": len(repro),
            "note": ("baseline state_correct should match the aligned "
                     "transformers_behavior.csv state_correct per sample"),
        },
        "per_group": {
            g: {c: {
                "n": len(rs),
                "rescue_rate": (sum(r["state_correct"] for r in rs)
                                / len(rs)),
                "mean_margin": (sum(r["margin_gt_state_minus_gt_prev"]
                                    for r in rs
                                    if r["margin_gt_state_minus_gt_prev"]
                                    is not None)
                                / max(1, sum(r["margin_gt_state_minus_gt_prev"]
                                             is not None for r in rs))),
            } for c, rs in conds.items()}
            for g, conds in by.items()},
    }
    with open(args.out_dir / "revision_rescue_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_dir / 'revision_rescue_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("build", "run"), default="build")
    ap.add_argument("--behavior", type=Path, default=BEHAVIOR_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--controls", type=str,
                    default="maintenance,other_failure,rest",
                    help="control groups to include besides stale")
    ap.add_argument("--model-dir", type=str, default=None,
                    help="required for --mode run")
    args = ap.parse_args()
    if args.mode == "run" and not args.model_dir:
        raise SystemExit("--mode run requires --model-dir")
    if args.mode == "build":
        build(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
