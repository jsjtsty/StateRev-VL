#!/usr/bin/env python3
"""Offline alignment audit for the Qwen native-state 31% vs 37% results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/vetbench/native_state_alignment_audit_v1"
SPLIT = ROOT / "outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"
RECURSIVE = ROOT / "outputs/vetbench/hidden_event_recursive_content_disjoint_v1/validation_prefixes.csv"
MECHANISM_BEHAVIOR = ROOT / "outputs/vetbench/mechanism_gate_final/behavior.csv"
AUDIT_MANIFEST = ROOT / "outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv"


def main() -> None:
    split = json.loads(SPLIT.read_text())
    val_traj = set(split["validation_trajectories"])

    # A: strict hidden-event recursive artifact. Its native prediction was
    # loaded from the behavior_audit_v2 mechanism manifest.
    a = pd.read_csv(RECURSIVE)
    a = a[a.trajectory_id.isin(val_traj)].copy()
    a = a.rename(columns={"native_state_pred": "prediction_A"})
    a["source_artifact_A"] = str(AUDIT_MANIFEST)

    # B: learned_state_updater's Qwen adapter. It uses baseline rows from the
    # mechanism-gate behavior cache and joins GT from the same audit manifest.
    b = pd.read_csv(MECHANISM_BEHAVIOR)
    b = b[b.condition.eq("baseline")].copy()
    b = b.rename(columns={"key": "target_prefix", "state_pred": "prediction_B"})
    b["trajectory_id"] = b.target_prefix.str.rsplit("_t", n=1).str[0]
    b["t"] = b.target_prefix.str.rsplit("_t", n=1).str[1].astype(int)
    b = b[b.trajectory_id.isin(val_traj)].copy()
    b["source_artifact_B"] = str(MECHANISM_BEHAVIOR)

    cols_a = ["target_prefix", "trajectory_id", "t", "gt_state", "prediction_A", "source_artifact_A"]
    cols_b = ["target_prefix", "prediction_B", "source_artifact_B"]
    out = a[cols_a].merge(b[cols_b], on="target_prefix", how="outer", validate="one_to_one")
    if len(out) != 100 or out.target_prefix.nunique() != 100:
        raise AssertionError(f"expected exactly 100 aligned validation prefixes, got {len(out)}")
    if set(out.trajectory_id) != val_traj:
        raise AssertionError("validation trajectories do not match the declared split")
    if out.prediction_A.isna().any() or out.prediction_B.isna().any():
        raise AssertionError("missing native prediction after alignment")
    out["gt_state"] = out.gt_state.astype(str)
    out["match_difference"] = out.prediction_A.eq(out.prediction_B).map({True: "match", False: "difference"})
    out["correct_A"] = out.prediction_A.eq(out.gt_state)
    out["correct_B"] = out.prediction_B.eq(out.gt_state)
    out = out.sort_values(["trajectory_id", "t"])

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "qwen_native_state_comparison.csv", index=False)

    # Verify that the 37% source is exactly the audit manifest state field.
    manifest = pd.read_csv(AUDIT_MANIFEST)
    manifest["target_prefix"] = manifest.trajectory_id.astype(str) + "_t" + manifest.t.astype(int).astype(str)
    m = manifest[manifest.target_prefix.isin(set(out.target_prefix))][["target_prefix", "state_pred"]]
    check = out[["target_prefix", "prediction_A"]].merge(m, on="target_prefix", validate="one_to_one")
    if not check.prediction_A.eq(check.state_pred).all():
        raise AssertionError("recursive artifact native predictions do not match audit manifest")

    report = {
        "split": str(SPLIT),
        "validation_trajectories": sorted(val_traj),
        "validation_trajectory_count": len(val_traj),
        "validation_prefix_count": len(out),
        "source_A": {
            "artifact": str(RECURSIVE),
            "native_prediction_source": str(AUDIT_MANIFEST),
            "accuracy": float(out.correct_A.mean()),
            "parser": "behavior_audit_v2 state_prediction already parsed by run_state_rev_audit.parse_tracking_option",
        },
        "source_B": {
            "artifact": str(MECHANISM_BEHAVIOR),
            "native_prediction_source": "mechanism_gate_final/behavior.csv condition=baseline state_pred",
            "accuracy": float(out.correct_B.mean()),
            "parser": "state_rev_input_pipeline.first_token_logits greedy token/logit answer decode",
        },
        "same_prefix_ids": True,
        "prediction_match_count": int(out.match_difference.eq("match").sum()),
        "prediction_difference_count": int(out.match_difference.eq("difference").sum()),
        "accuracy_difference_source": "13 differing saved native predictions, not split, GT, aggregation, or prefix coverage",
        "prompt": "Both code paths call run_state_rev_audit.state_messages / state_question_text; no prompt text difference found in the checked source.",
        "aggregation": "Both are row-level mean over the same 100 prefixes; trajectory-cluster summaries wrap the same rows but do not explain the 31/37 discrepancy.",
        "recommended_authoritative_baseline": "37% for the current strict content-disjoint hidden-event recursive baseline, because it is the native_state endpoint in the canonical content-disjoint recursive artifact and matches behavior_audit_v2/mechanism_candidates.csv.",
        "legacy_or_alternate_baseline": "31% should be retained as a separate mechanism_gate_final baseline-cache result, not silently merged with the 37% result.",
    }
    (OUT / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Qwen Native State Alignment Audit",
        "",
        "Offline comparison only; no model forward was run.",
        "",
        f"- Same strict validation split: `{SPLIT}`",
        f"- Validation coverage: `{len(val_traj)} trajectories / {len(out)} prefixes`",
        f"- Source A accuracy: `{report['source_A']['accuracy']:.2%}`",
        f"- Source B accuracy: `{report['source_B']['accuracy']:.2%}`",
        f"- Prediction matches: `{report['prediction_match_count']}/100`",
        f"- Prediction differences: `{report['prediction_difference_count']}/100`",
        "",
        "## Conclusion",
        "",
        "The 31% vs 37% discrepancy is caused by different saved native behavior caches. The split, validation prefix IDs, GT labels, prompt family, and row-level aggregation are aligned. Source A uses `behavior_audit_v2/mechanism_candidates.csv`; source B uses `mechanism_gate_final/behavior.csv` baseline rows. The 37% result is the recommended authoritative native baseline for the current strict content-disjoint recursive report. The 31% result must remain separately labeled as an alternate behavior-cache result.",
        "",
        "Old artifacts were not overwritten.",
    ]
    (OUT / "alignment_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
