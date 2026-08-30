"""Build the paired pre/post-event state-retention dataset for StateRev-VL.

For every event-driven transition step t (t=1..5) of every trajectory, a PAIR
is defined:

    h_pre  = h_{t-1}   hidden state at the probe position of the (t-1)-prefix
                       (state question, BEFORE the t-th swap is shown)
    h_post = h_t       hidden state at the probe position of the t-prefix
                       (state question, AFTER the t-th swap is shown)

The pair row joins both hidden-state keys with the GT state/event at step t
and the Transformers-aligned behavioral masks (this script must NOT read the
old vLLM manifest).

The hidden-state store currently holds keys cup_XXX_t1..t5 only (t=0 was never
extracted), so t=1 pairs are INCOMPLETE (h_pre = cup_XXX_t0 missing) and must
be backfilled by a future `run_hidden_state_probe.py --include-t0` run. t=2..5
pairs are fully available now.

Outputs (out-dir, default outputs/vetbench/state_retention_analysis_v1/):
    state_retention_pairs.csv        one row per (trajectory, t) pair
    state_retention_pairs_summary.json
                                     counts, subset sizes, sufficiency verdict,
                                     backfill plan

Verification performed (and printed):
    * every pair's two keys share the trajectory prefix and the correct t;
    * gt_prev_state(t) == gt_state(t-1) for t=2..5 and
      gt_prev_state(t=1) == initial_state, per trajectory;
    * swap invariant gt_state(t) == apply_swap(gt_prev_state(t), gt_event(t));
    * canonical / success / maintenance counts match the aligned behavior
      summary (aligned_behavior_summary.json);
    * available hidden-state keys match the pair keys exactly.

Run from the project root (CPU-only, no model):
    python scripts/build_state_retention_pairs.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "composition_analysis_v1"
ALIGNED_MANIFEST = V1_DIR / "transformers_behavior.csv"
ALIGNED_SUMMARY = V1_DIR / "aligned_behavior_summary.json"
PROBE_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "hidden_state_probe"
HIDDEN_NPZ = PROBE_DIR / "hidden_states.npz"
OUT_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "state_retention_analysis_v1"

STATE_CLASSES = ("Left", "Middle", "Right")
STATE_INDEX = {c: i for i, c in enumerate(STATE_CLASSES)}
POS = {"Left": 0, "Middle": 1, "Right": 2}
POS_NAME = list(STATE_CLASSES)
PAIR = {"Left and Middle": (0, 1), "Middle and Right": (1, 2),
        "Left and Right": (0, 2)}

CSV_COLUMNS = [
    "trajectory_id", "t", "h_pre_key", "h_post_key",
    "h_pre_available", "h_post_available", "pair_complete",
    "gt_prev_state", "gt_state", "gt_event",
    "state_pred", "event_pred",
    "is_transition", "event_correct", "prev_state_correct", "state_correct",
    "clean_revision", "revision_success", "canonical_stale_failure",
    "clean_maintenance", "canonical_group", "joint_class",
]


def apply_swap(pos: int, pair: tuple[int, int]) -> int:
    a, b = pair
    if pos == a:
        return b
    if pos == b:
        return a
    return pos


def load_aligned_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"] = int(r["t"])
        for k in ("is_transition", "event_correct", "prev_state_correct",
                  "state_correct", "clean_revision", "clean_maintenance"):
            r[k] = r[k] == "true"
    return rows


def canonical_group(r: dict) -> str:
    """success / stale / other_failure for clean-revision rows, '' otherwise.
    Definitions identical to build_aligned_behavior.py:
      stale = clean_revision AND NOT state_correct AND state_pred == gt_prev.
    """
    if not r["clean_revision"]:
        return ""
    if r["state_correct"]:
        return "success"
    if r["state_pred"] == r["gt_prev_state"]:
        return "stale"
    return "other_failure"


def build_pairs(manifest: list[dict], hs_keys: set[str]) -> list[dict]:
    pairs: list[dict] = []
    for r in sorted(manifest, key=lambda x: (x["trajectory_id"], x["t"])):
        tr, t = r["trajectory_id"], r["t"]
        pre_key = f"{tr}_t{t - 1}"
        post_key = f"{tr}_t{t}"
        pre_ok = pre_key in hs_keys
        post_ok = post_key in hs_keys
        pairs.append({
            "trajectory_id": tr,
            "t": t,
            "h_pre_key": pre_key,
            "h_post_key": post_key,
            "h_pre_available": pre_ok,
            "h_post_available": post_ok,
            "pair_complete": pre_ok and post_ok,
            "gt_prev_state": r["gt_prev_state"],
            "gt_state": r["gt_state"],
            "gt_event": r["gt_event"],
            "state_pred": r["state_pred"],
            "event_pred": r["event_pred"],
            "is_transition": r["is_transition"],
            "event_correct": r["event_correct"],
            "prev_state_correct": r["prev_state_correct"],
            "state_correct": r["state_correct"],
            "clean_revision": r["clean_revision"],
            "revision_success": bool(r["clean_revision"]) and r["state_correct"],
            "canonical_stale_failure":
                bool(r["clean_revision"]) and not r["state_correct"]
                and r["state_pred"] == r["gt_prev_state"],
            "clean_maintenance": r["clean_maintenance"],
            "canonical_group": canonical_group(r),
            "joint_class": r["joint_class"],
        })
    return pairs


def verify(pairs: list[dict], manifest: list[dict],
           hs_keys: set[str]) -> tuple[list[str], dict]:
    """Return (errors, checks). Aborts the caller on any error."""
    errs: list[str] = []
    by_traj: dict[str, list[dict]] = {}
    for p in pairs:
        by_traj.setdefault(p["trajectory_id"], []).append(p)

    # 1. same-trajectory pairing + correct t offset
    for p in pairs:
        tr = p["trajectory_id"]
        for key, want_t in ((p["h_pre_key"], p["t"] - 1),
                            (p["h_post_key"], p["t"])):
            traj_part, _, t_part = key.rpartition("_t")
            if traj_part != tr:
                errs.append(f"pair {tr} t={p['t']}: key {key} wrong trajectory")
            if t_part != str(want_t):
                errs.append(f"pair {tr} t={p['t']}: key {key} wrong t")

    # 2. GT chain: prev_state(t) == state(t-1); t=1 prev == initial state
    for tr, rows in by_traj.items():
        rows = sorted(rows, key=lambda x: x["t"])
        m0 = next(m for m in manifest
                  if m["trajectory_id"] == tr and m["t"] == 1)
        if rows[0]["t"] != 1 or rows[-1]["t"] != 5:
            errs.append(f"{tr}: expected t=1..5, got {rows[0]['t']}..{rows[-1]['t']}")
            continue
        if rows[0]["gt_prev_state"] != m0["initial_state"]:
            errs.append(f"{tr}: gt_prev_state(t=1) != initial_state")
        for a, b in zip(rows, rows[1:]):
            if b["t"] != a["t"] + 1:
                errs.append(f"{tr}: t gap {a['t']} -> {b['t']}")
            if b["gt_prev_state"] != a["gt_state"]:
                errs.append(f"{tr}: gt_prev_state(t={b['t']}) != "
                            f"gt_state(t={a['t']})")
        # 3. swap invariant on every row
        for p in rows:
            exp = POS_NAME[apply_swap(POS[p["gt_prev_state"]],
                                      PAIR[p["gt_event"]])]
            if exp != p["gt_state"]:
                errs.append(f"{tr} t={p['t']}: swap invariant violated "
                            f"({p['gt_prev_state']}+{p['gt_event']} -> "
                            f"{exp} != {p['gt_state']})")

    # 4. hidden-state availability matches keys exactly
    used = {p["h_post_key"] for p in pairs} | {p["h_pre_key"] for p in pairs}
    expected_keys = {f"{tr}_t{t}" for tr in by_traj for t in range(1, 6)}
    if used - {f"{tr}_t0" for tr in by_traj} != expected_keys:
        errs.append("pair keys do not cover exactly the t=1..5 hidden-state "
                    "keys plus the t=0 backfill keys")
    if not (used & hs_keys and used - hs_keys ==
            {f"{tr}_t0" for tr in by_traj}):
        # the only allowed missing keys are the t=0 pre-states
        missing = used - hs_keys
        extra = hs_keys - used
        if missing - {f"{tr}_t0" for tr in by_traj} or extra:
            errs.append(f"hidden-state key mismatch: missing={sorted(missing)[:4]} "
                        f"extra={sorted(extra)[:4]}")

    # 5. counts
    n_rev = sum(p["clean_revision"] for p in pairs)
    n_succ = sum(p["revision_success"] for p in pairs)
    n_stale = sum(p["canonical_stale_failure"] for p in pairs)
    n_maint = sum(p["clean_maintenance"] for p in pairs)
    n_complete = sum(p["pair_complete"] for p in pairs)
    checks = {
        "n_pairs": len(pairs),
        "n_trajectories": len(by_traj),
        "pairs_complete": n_complete,
        "pairs_missing_h_pre_t0": len(pairs) - n_complete,
        "per_t": {str(t): sum(1 for p in pairs if p["t"] == t)
                  for t in range(1, 6)},
        "per_t_complete": {str(t): sum(1 for p in pairs
                                       if p["t"] == t and p["pair_complete"])
                           for t in range(1, 6)},
        "clean_revision": n_rev,
        "revision_success": n_succ,
        "canonical_stale_failure": n_stale,
        "clean_maintenance": n_maint,
        "canonical_by_t": {str(t): {g: sum(1 for p in pairs
                                           if p["t"] == t and p["canonical_group"] == g)
                                    for g in ("success", "stale", "other_failure")}
                           for t in range(1, 6)},
        "hidden_states_available": len(used & hs_keys),
        "hidden_states_missing_n": len(used - hs_keys),
        "hidden_states_missing_example": sorted(used - hs_keys)[:3],
    }
    # 5b. cross-check against the aligned summary counts
    try:
        s = json.loads(ALIGNED_SUMMARY.read_text())
        want = {
            "clean_revision": s["clean_revision"]["n"],
            "revision_success": s["clean_revision"]["success"],
            "canonical_stale_failure": s["clean_revision"]["stale"],
        }
        for k, v in want.items():
            if checks[k] != v:
                errs.append(f"count mismatch vs aligned summary: {k} "
                            f"pairs={checks[k]} summary={v}")
    except FileNotFoundError:
        print(f"(warning) {ALIGNED_SUMMARY} not found; skipped summary cross-check")
    return errs, checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ALIGNED_MANIFEST)
    ap.add_argument("--npz", type=Path, default=HIDDEN_NPZ)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    manifest = load_aligned_manifest(args.manifest)
    if len(manifest) != 250:
        raise SystemExit(f"expected 250 aligned manifest rows, got {len(manifest)}")
    if any(r["t"] == 0 for r in manifest):
        raise SystemExit("manifest unexpectedly contains t=0 rows")

    with np.load(args.npz, allow_pickle=False) as z:
        hs_keys = set(z.keys())

    pairs = build_pairs(manifest, hs_keys)
    errs, checks = verify(pairs, manifest, hs_keys)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        raise SystemExit(f"pair verification failed with {len(errs)} error(s)")

    suff = all(p["pair_complete"] for p in pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "state_retention_pairs.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for p in pairs:
            row = dict(p)
            row = {k: (str(v).lower() if isinstance(v, bool) else v)
                   for k, v in row.items()}
            w.writerow(row)

    summary = {
        "manifest": str(args.manifest),
        "hidden_states": str(args.npz),
        "hidden_state_keys": sorted(hs_keys)[:3] + [f"... ({len(hs_keys)} total)"],
        "existing_hidden_states_sufficient": suff,
        "missing": ("t=0 prefix hidden states (cup_XXX_t0): needed as h_pre of "
                    "the t=1 pairs; backfill with "
                    "`python scripts/run_hidden_state_probe.py --include-t0` "
                    "(reuse logic extracts only the 50 missing keys)"),
        "backfill_keys": sorted({p["h_pre_key"] for p in pairs
                                 if not p["h_pre_available"]}),
        "checks": checks,
    }
    (args.out_dir / "state_retention_pairs_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"Wrote {args.out_dir}/state_retention_pairs.csv ({len(pairs)} rows)")
    print(f"Wrote {args.out_dir}/state_retention_pairs_summary.json")
    print(f"EXISTING_HIDDEN_STATES_SUFFICIENT={str(suff).lower()}")
    print(f"pairs complete: {checks['pairs_complete']}/{checks['n_pairs']} "
          f"(per t: {checks['per_t_complete']})")
    print(f"clean_revision={checks['clean_revision']} "
          f"success={checks['revision_success']} "
          f"stale={checks['canonical_stale_failure']} "
          f"maintenance={checks['clean_maintenance']}")
    print(f"canonical by t: {checks['canonical_by_t']}")
    print("all verification checks passed")


if __name__ == "__main__":
    main()
