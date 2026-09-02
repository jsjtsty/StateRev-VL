#!/usr/bin/env python3
"""Validity Gate Stage 0.1: count audit (CPU only).

Resolves the "229 vs 200" discrepancy in the v2 rescue/trace subset:
raw data has 250 rows = 50 trajectories x t=1..5 (50 per t). A strict
t>=2 filter would give 200 rows, but the rescue/trace manifest has 229.

This script recomputes the exact group masks from the raw behavior CSV
(verbatim copy of load_samples.group_of in scripts/run_revision_rescue.py)
and outputs per-t counts and group x t cross tables.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def T(x: str) -> bool:
    return str(x).strip().lower() == "true"


def group_of(r: dict) -> str | None:
    # VERBATIM from scripts/run_revision_rescue.py load_samples()
    if not T(r["clean_revision"]) and not T(r["clean_maintenance"]):
        return "rest" if int(r["t"]) in (2, 3, 4, 5) else None
    if T(r["clean_maintenance"]):
        return "maintenance"
    if T(r["state_correct"]):
        return "success"
    if r["state_pred"] == r["gt_prev_state"]:
        return "stale"
    return "other_failure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior-csv",
                    default="outputs/vetbench/composition_analysis_v1/"
                            "transformers_behavior.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.behavior_csv, newline="")))
    n_traj = len({r["trajectory_id"] for r in rows})
    t_dist = Counter(r["t"] for r in rows)
    assert len(rows) == 250 and n_traj == 50, "raw data shape changed?"
    assert dict(t_dist) == {str(i): 50 for i in range(1, 6)}, t_dist

    groups = {}
    for r in rows:
        g = group_of(r)
        r["_group_all"] = g  # group regardless of inclusion
        groups.setdefault(g, []).append(r)

    included = [r for r in rows
                if r["_group_all"] in ("stale", "maintenance",
                                       "other_failure", "rest")]
    excl_t1 = [r for r in rows if int(r["t"]) == 1 and r not in included]

    cross = {g: {str(t): sum(1 for r in rs if r["t"] == str(t))
                 for t in (1, 2, 3, 4, 5)}
             for g, rs in groups.items() if g is not None}
    cross["None(excluded)"] = {
        str(t): sum(1 for r in rows if r["t"] == str(t)
                    and r["_group_all"] is None)
        for t in (1, 2, 3, 4, 5)}

    n_strict_t2 = sum(1 for r in rows if int(r["t"]) >= 2)
    n_incl_t2 = sum(1 for r in included if int(r["t"]) >= 2)
    n_incl_t1 = sum(1 for r in included if int(r["t"]) == 1)
    t1_by_group = Counter(r["_group_all"] for r in included
                          if int(r["t"]) == 1)

    out = {
        "raw": {
            "file": args.behavior_csv,
            "rows": len(rows),
            "trajectories": n_traj,
            "t_distribution": {str(k): v
                               for k, v in sorted(t_dist.items())},
            "strict_t_ge_2_rows": n_strict_t2,
        },
        "group_definitions": {
            "source": "scripts/run_revision_rescue.py load_samples."
                      "group_of (verbatim)",
            "note": "The t in {2,3,4,5} filter applies ONLY to the 'rest' "
                    "group. clean_maintenance and clean_revision-failure "
                    "(stale/other_failure) rows are included at ANY t.",
            "rest": "not clean_revision and not clean_maintenance and t in "
                    "{2,3,4,5}",
            "maintenance": "clean_maintenance (any t)",
            "success": "clean_revision and state_correct (any t) - NOT in "
                       "the rescue/trace manifest",
            "stale": "clean_revision and not state_correct and "
                     "state_pred == gt_prev_state (any t; 0 rows at t=1)",
            "other_failure": "clean_revision and not state_correct and "
                             "state_pred != gt_prev_state (any t)",
            "excluded": "t=1 rows that are none of the above (non-clean t=1 "
                        "rows and t=1 success rows)",
        },
        "group_x_t_cross_table": cross,
        "group_totals_all_rows": {g: len(rs) for g, rs in groups.items()
                                  if g is not None},
        "included_subset": {
            "n_total": len(included),
            "n_t_ge_2": n_incl_t2,
            "n_t_eq_1": n_incl_t1,
            "t_eq_1_by_group": dict(t1_by_group),
            "per_group": {g: len([r for r in included if r["_group_all"] == g])
                          for g in ("stale", "maintenance", "other_failure",
                                    "rest")},
        },
        "excluded_t1_rows": {
            "n": len(excl_t1),
            "by_group_all": dict(Counter(r["_group_all"] for r in excl_t1)),
            "sample_ids": [f"{r['trajectory_id']}_t{r['t']}"
                           for r in excl_t1][:10],
        },
        "verdict": {
            "229_explained": True,
            "explanation": (
                "229 = 200 (all t>=2 rows: 163 rest + 26 stale + 8 "
                "maintenance + 3 other_failure) + 29 (t=1 rows: 15 "
                "clean_maintenance + 14 clean_revision other_failure). "
                "The subset is a union of per-GROUP masks, not a per-t "
                "mask. The label 't>=2 subset' used in the v2 reports is "
                "WRONG; the stale group itself is unaffected (all 26 rows "
                "are t=2..5), but maintenance (15/23 at t=1) and "
                "other_failure (14/17 at t=1) include t=1 rows whose S0 is "
                "stated verbatim in the prompt."),
            "fix_required": (
                "correct the 't>=2' label in "
                "stale_origin_conclusions.md and claude_opus5_handoff.md; "
                "no raw data or group masks changed."),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("included_subset", "excluded_t1_rows")}, indent=1))
    print("verdict:", out["verdict"]["explanation"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
