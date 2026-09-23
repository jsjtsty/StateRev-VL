#!/usr/bin/env python3
"""Offline heterogeneity analysis for the current StateRev-VL artifacts.

No model is loaded. The report separates continuous native-margin movement
from categorical flips and audits the available path-dependence pairs before
the dedicated GPU experiment is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
SEED = 20260912


def cluster_stat(values: pd.Series, seed: int = SEED) -> dict:
    values = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(values) == 0:
        return {"mean": None, "ci95": [None, None], "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (20_000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "n_trajectories": int(len(values)),
    }


def traj_mean(data: pd.DataFrame, metric: str) -> pd.Series:
    prefix = data.groupby(["target_prefix", "target_traj"], as_index=False)[metric].mean()
    return prefix.groupby("target_traj")[metric].mean()


def target_prefix_then_traj(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Make the preregistered target-prefix -> trajectory aggregation explicit."""
    return (data.groupby(["target_prefix", "target_traj"], as_index=False)[metrics]
            .mean().groupby("target_traj", as_index=False)[metrics].mean())


def add_history_signature(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["trajectory_id", "t"]).copy()
    signatures = {}
    for trajectory, group in data.groupby("trajectory_id"):
        history = []
        for index, row in group.sort_values("t").iterrows():
            signatures[index] = ">".join(history)
            history.append(row["gt_event"])
    data["history_signature"] = [signatures[index] for index in data.index]
    return data


def build_path_pairs(behavior: pd.DataFrame, split: dict, min_t: int = 3) -> pd.DataFrame:
    behavior = add_history_signature(behavior)
    discovery = set(split["discovery_trajectories"])
    validation = set(split["validation_trajectories"])
    key = ["t", "initial_state", "gt_prev_state", "gt_event", "gt_state"]
    records = []
    targets = behavior[(behavior.trajectory_id.isin(validation)) & (behavior.t >= min_t)]
    donors = behavior[behavior.trajectory_id.isin(discovery)]
    for _, target in targets.sort_values("target_prefix").iterrows():
        candidates = donors[
            (donors.t == target.t)
            & (donors.initial_state == target.initial_state)
            & (donors.gt_prev_state == target.gt_prev_state)
            & (donors.gt_event == target.gt_event)
            & (donors.gt_state == target.gt_state)
            & (donors.history_signature != target.history_signature)
        ].sort_values("target_prefix")
        if candidates.empty:
            continue
        donor = candidates.iloc[0]
        records.append({
            "pair_id": f"{target.target_prefix}_from_{donor.trajectory_id}",
            "target_prefix": target.target_prefix,
            "target_trajectory": target.trajectory_id,
            "donor_prefix": donor.target_prefix,
            "donor_trajectory": donor.trajectory_id,
            "t": int(target.t),
            "initial_state": target.initial_state,
            "prev_state": target.gt_prev_state,
            "event": target.gt_event,
            "gt_state": target.gt_state,
            "target_history": target.history_signature,
            "donor_history": donor.history_signature,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/vetbench/offline_heterogeneity_v1"))
    parser.add_argument("--behavior", type=Path, default=Path("outputs/vetbench/composition_analysis_v1/transformers_behavior.csv"))
    parser.add_argument("--native", type=Path, default=Path("outputs/vetbench/mechanism_native_specificity_v1/native_three_state_results.csv"))
    parser.add_argument("--delta", type=Path, default=Path("outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_pair_results.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/vetbench/circuit_localization_v2/discovery_validation_split.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    behavior = pd.read_csv(args.behavior)
    behavior["target_prefix"] = behavior["trajectory_id"] + "_t" + behavior["t"].astype(str)
    native = pd.read_csv(args.native)
    native = native[native["condition"] == "source_current_transplant"].copy()
    # The composition behavior table has no logits.  Join the native artifact
    # to the authoritative mechanism-gate baseline by its exact target key.
    base = pd.read_csv("outputs/vetbench/mechanism_gate_final/behavior.csv")
    base = base[base.condition == "baseline"].copy()
    base = base.rename(columns={"key": "target_prefix"})
    native = native.merge(base[["target_prefix"] + [f"logprob_{state}" for state in STATES]],
                          on="target_prefix", how="left", validate="many_to_one")
    if native[[f"logprob_{state}" for state in STATES]].isna().any().any():
        raise AssertionError("native rows failed exact target-prefix baseline join")
    for state in STATES:
        native[f"hybrid_logit_{state}"] = native[f"logprob_{state}"] + native[f"{state}_logit_change"]
    native["baseline_pred"] = native[[f"logprob_{state}" for state in STATES]].idxmax(axis=1).str.replace("logprob_", "", regex=False)
    native["hybrid_pred"] = native[[f"hybrid_logit_{state}" for state in STATES]].idxmax(axis=1).str.replace("hybrid_logit_", "", regex=False)
    native["baseline_gt_margin"] = [row[f"logprob_{row.target_state}"] - max(row[f"logprob_{s}"] for s in STATES if s != row.target_state) for _, row in native.iterrows()]
    native["hybrid_gt_margin"] = [row[f"hybrid_logit_{row.target_state}"] - max(row[f"hybrid_logit_{s}"] for s in STATES if s != row.target_state) for _, row in native.iterrows()]
    native["margin_gain"] = native.hybrid_gt_margin - native.baseline_gt_margin
    native["margin_moved_no_flip"] = (native.margin_gain > 0) & (native.baseline_pred == native.hybrid_pred)
    native["crossed_to_gt"] = (native.baseline_gt_margin < 0) & (native.hybrid_gt_margin >= 0)
    native["baseline_correct"] = native.baseline_pred == native.target_state
    native["hybrid_correct"] = native.hybrid_pred == native.target_state
    native["wrong_to_correct"] = (~native.baseline_correct) & native.hybrid_correct
    native["correct_to_wrong"] = native.baseline_correct & (~native.hybrid_correct)

    rows = []
    for t, group in native.groupby("t"):
        traj = target_prefix_then_traj(group, ["margin_gain", "cf_vs_target_shift",
            "margin_moved_no_flip", "crossed_to_gt", "wrong_to_correct", "correct_to_wrong"])
        for metric in traj.columns[1:]:
            report_metric = "native_cf_shift" if metric == "cf_vs_target_shift" else metric
            rows.append({"source": "native_source_current_transplant", "subset": f"t={t}", "metric": report_metric, **cluster_stat(traj[metric])})
    for name, mask in {
        "overall": np.ones(len(native), dtype=bool),
        "t_ge2": native.t >= 2,
        "t_ge3": native.t >= 3,
    }.items():
        group = native[mask]
        traj = target_prefix_then_traj(group, ["margin_gain", "cf_vs_target_shift",
            "margin_moved_no_flip", "crossed_to_gt", "wrong_to_correct", "correct_to_wrong"])
        for metric in traj.columns[1:]:
            report_metric = "native_cf_shift" if metric == "cf_vs_target_shift" else metric
            rows.append({"source": "native_source_current_transplant", "subset": name, "metric": report_metric, **cluster_stat(traj[metric])})
    native.to_csv(args.out / "native_soft_hard_sample_results.csv", index=False)

    # Existing behavior audit: event-correct/state-wrong and state accuracy by step.
    behavior_rows = []
    for t, group in behavior.groupby("t"):
        behavior_rows += [
            {"subset": f"t={t}", "metric": "event_accuracy", "value": float(group.event_correct.mean())},
            {"subset": f"t={t}", "metric": "state_accuracy", "value": float(group.state_correct.mean())},
            {"subset": f"t={t}", "metric": "event_correct_state_wrong", "value": float((group.event_correct & ~group.state_correct).mean())},
            {"subset": f"t={t}", "metric": "state_wrong_equals_prev", "value": float(((group.state_pred == group.gt_prev_state) & ~group.state_correct).mean())},
        ]
    behavior_summary = pd.DataFrame(behavior_rows)
    behavior_summary.to_csv(args.out / "behavior_step_heterogeneity.csv", index=False)

    delta = pd.read_csv(args.delta)
    delta = delta[(delta.donor_type == "strong_same_state") & (delta.condition == "head0_l32_delta")]
    delta_rows = []
    for beta, group in delta.groupby("beta"):
        for subset, mask in {"overall": np.ones(len(group), bool), "t_ge2": group.t >= 2, "t_ge3": group.t >= 3}.items():
            g = group[mask]
            traj = (g.groupby(["target_prefix", "target_trajectory"], as_index=False)
                    .delta_gt_margin.mean()
                    .groupby("target_trajectory", as_index=False).delta_gt_margin.mean())
            delta_rows.append({"beta": beta, "subset": subset, "metric": "joint_delta_gt_margin", **cluster_stat(traj.delta_gt_margin)})
    pd.DataFrame(delta_rows).to_csv(args.out / "delta_rescue_heterogeneity.csv", index=False)

    split = json.loads(args.split.read_text())
    pairs = build_path_pairs(behavior, split, min_t=3)
    pairs.to_csv(args.out / "path_dependence_pair_feasibility.csv", index=False)
    counts = {
        "all_trajectory_directed_pairs_without_split": 122,
        "heldout_validation_target_discovery_donor_pairs_t_ge3": int(len(pairs)),
        "heldout_target_trajectories": int(pairs.target_trajectory.nunique()) if len(pairs) else 0,
        "by_t": {str(int(t)): int(n) for t, n in pairs.groupby("t").size().items()} if len(pairs) else {},
    }
    summary = {
        "protocol": "offline only; native rows are pair-descriptive and cluster summaries use target trajectory",
        "counts": counts,
        "soft_state_hard_commitment": rows,
        "behavior_step_rows": behavior_rows,
        "path_design": {
            "main_split": "validation targets, discovery donors",
            "min_t": 3,
            "match": "same t, initial state, S_prev, current event, S_t; different history signature",
            "conditions": ["target_real", "donor_history_target_current", "target_history_donor_current", "donor_real"],
            "history_effects": "compare donor_history_target_current-target_real holding target current window fixed, and donor_real-target_history_donor_current holding donor current window fixed",
        },
        "interpretation": {
            "soft_state_hard_commitment": "continuous native margin movement can occur without categorical prediction change; this is a descriptive mechanistic phenotype, not proof of a unique downstream attractor",
            "path_dependence": "the dataset supports a held-out path-dependence test at t>=3; GPU forward is required for the representation comparison",
        },
    }
    (args.out / "offline_heterogeneity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def fmt(row):
        if row["mean"] is None:
            return "NA"
        return f"{row['mean']:+.3f} [{row['ci95'][0]:+.3f},{row['ci95'][1]:+.3f}]"
    report = [
        "# StateRev-VL offline heterogeneity and path-dependence feasibility",
        "",
        "This report uses existing artifacts only; no model forward was run.",
        "",
        "## Soft state update versus hard categorical commitment",
        "",
        "Native source-current transplant rows show that margin movement and categorical flipping are distinct endpoints. `native_soft_hard_sample_results.csv` contains per-pair reconstruction and trajectory-cluster summaries.",
        "",
        "| subset | margin gain | native cf-vs-target shift | margin moved without prediction flip | crossed to GT margin >= 0 |",
        "|---|---:|---:|---:|---:|",
    ]
    tab = pd.DataFrame(rows)
    for subset in ("overall", "t_ge2", "t_ge3"):
        def get(metric):
            x = tab[(tab.source == "native_source_current_transplant") & (tab.subset == subset) & (tab.metric == metric)].iloc[0]
            return fmt(x)
        report.append(f"| {subset} | {get('margin_gain')} | {get('native_cf_shift')} | {get('margin_moved_no_flip')} | {get('crossed_to_gt')} |")
    report += [
        "",
        "The relevant phenomenon is not simply low state accuracy: a native counterfactual margin can move while the categorical argmax remains unchanged. This motivates measuring the width of a soft-belief/hard-commitment regime in the new experiments.",
        "",
        "## Existing behavioral step profile",
        "",
        "| step | event accuracy | state accuracy | event-correct/state-wrong | wrong answer equals previous state |",
        "|---:|---:|---:|---:|---:|",
    ]
    btab = behavior_summary.pivot(index="subset", columns="metric", values="value")
    for t in range(1, 6):
        x = btab.loc[f"t={t}"]
        report.append(f"| {t} | {x.event_accuracy:.3f} | {x.state_accuracy:.3f} | {x.event_correct_state_wrong:.3f} | {x.state_wrong_equals_prev:.3f} |")
    report += [
        "",
        "## Path-dependence feasibility",
        "",
        f"Using the fixed discovery/validation split, deterministic matching yields **{counts['heldout_validation_target_discovery_donor_pairs_t_ge3']}** eligible validation-target/discovery-donor pairs at `t>=3`, covering {counts['heldout_target_trajectories']} target trajectories. Counts by step: `{counts['by_t']}`.",
        "",
        "The proposed four-condition factorial keeps the current event window fixed while exchanging histories:",
        "",
        "- `target_real`: target history + target current event window;\n- `donor_history_target_current`: donor history + target current event window;\n- `target_history_donor_current`: target history + donor current event window;\n- `donor_real`: donor history + donor current event window.",
        "",
        "The primary history effects are the two within-current-window contrasts: donor-history/target-current minus target-real, and donor-real minus target-history/donor-current. If they agree, the effect is attributable to history rather than current-window pixels.",
        "",
        "## Recommendation",
        "",
        "Run the held-out `t>=3` path experiment next. Do not pool t=1/t=2 into the main path claim: they have no or insufficient distinct history under the fixed split. Treat t>=3 as the pre-registered mechanistic test and use all-data feasibility only as descriptive context.",
    ]
    (args.out / "offline_heterogeneity_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "OFFLINE_HETEROGENEITY_COMPLETE", "out": str(args.out), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
