#!/usr/bin/env python3
"""Trajectory-clustered offline analysis for frozen Head 0 dose response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260909
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DONOR_TYPES = ("strong_same_state", "different_event")
SUBSETS = ("overall", "t_ge2", "t_ge3", "weak", "strong", "baseline_margin_lt_zero")


def cluster_stat(values: pd.Series, seed: int = SEED) -> dict:
    values = values.dropna().to_numpy(dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (20_000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_trajectories": int(len(values)),
    }


def subset(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "overall": return data
    if name == "t_ge2": return data[data["t"] >= 2]
    if name == "t_ge3": return data[data["t"] >= 3]
    if name == "weak": return data[data["strength_group"] == "weak"]
    if name == "strong": return data[data["strength_group"] == "strong"]
    if name == "baseline_margin_lt_zero": return data[data["baseline_gt_margin"] < 0]
    raise KeyError(name)


def trajectory_values(data: pd.DataFrame, metric: str) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    prefix = data.groupby(["target_prefix", "target_trajectory"], as_index=False)[metric].mean()
    return prefix.groupby("target_trajectory")[metric].mean()


def conditional_trajectory_rate(data: pd.DataFrame, numerator: str, denominator: str) -> tuple[pd.Series, int]:
    selected = data[data[denominator]].copy()
    return trajectory_values(selected, numerator), int(len(selected))


def first_crossing(group: pd.DataFrame, metric: str) -> float | None:
    crossed = group[group[metric] > 0]
    return None if crossed.empty else float(crossed["alpha"].min())


def target_summaries(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (target, donor_type), group in data.groupby(["target_prefix", "donor_type"]):
        group = group.sort_values("alpha")
        if tuple(group["alpha"].to_numpy(float)) != ALPHAS:
            raise AssertionError(f"incomplete alpha grid for {target}/{donor_type}")
        endpoint = "patched_gt_margin" if donor_type == "strong_same_state" else "patched_desired_margin"
        y = group[endpoint].to_numpy(float)
        alpha = group["alpha"].to_numpy(float)
        slope = float(np.polyfit(alpha, y, 1)[0])
        first = group.iloc[0]
        final = group.iloc[-1]
        records.append({
            "target_prefix": target,
            "target_trajectory": first["target_trajectory"],
            "t": int(first["t"]),
            "donor_type": donor_type,
            "donor_prefix": first["donor_prefix"],
            "gt_state": first["gt_state"],
            "desired_state": first["desired_state"],
            "strength_group": first["strength_group"],
            "baseline_gt_margin": float(first["baseline_gt_margin"]),
            "baseline_desired_margin": float(first["baseline_desired_margin"]),
            "baseline_correct": bool(first["baseline_correct"]),
            "endpoint": endpoint,
            "dose_slope": slope,
            "monotonic_nondecreasing": bool(np.all(np.diff(y) >= -1e-6)),
            "n_negative_adjacent_steps": int(np.count_nonzero(np.diff(y) < -1e-6)),
            "minimum_adjacent_step": float(np.diff(y).min()),
            "decision_boundary_alpha": first_crossing(group, endpoint),
            "prediction_crossing_alpha": (
                None if group[group["patched_pred"] == group["desired_state"]].empty
                else float(group.loc[group["patched_pred"] == group["desired_state"], "alpha"].min())
            ),
            "alpha2_endpoint_change": float(y[-1] - y[0]),
            "alpha2_desired_specificity": float(final["desired_state_specificity"]),
            "alpha2_new_unintended_prediction": bool(final["new_unintended_prediction"]),
            "alpha2_centered_logit_l2": float(final["centered_three_state_logit_l2"]),
            "alpha2_entropy_change": float(final["delta_native_three_state_entropy"]),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/vetbench/head0_dose_response_v1"))
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--rescue-results", type=Path,
                        default=Path("outputs/vetbench/head0_causal_rescue_v1/rescue_pair_results.csv"))
    args = parser.parse_args()
    manifest = json.loads((args.out / "dose_response_manifest.json").read_text())
    files = [args.out / "shards" / f"dose_response_shard_{index}.csv" for index in range(args.num_shards)]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dose-response shards: {missing}")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    expected_targets = {entry["target_prefix"] for entry in manifest["targets"]}
    if set(data["target_prefix"]) != expected_targets:
        raise AssertionError("target coverage differs from frozen manifest")
    expected_rows = len(expected_targets) * len(DONOR_TYPES) * len(ALPHAS)
    if len(data) != expected_rows or data.duplicated(["target_prefix", "donor_type", "alpha"]).any():
        raise AssertionError("dose-response rows are incomplete or duplicated")
    if set(data["donor_type"]) != set(DONOR_TYPES) or set(data["alpha"]) != set(ALPHAS):
        raise AssertionError("unexpected donor type or alpha")
    for column in ("target_input_ids_sha256", "target_pixel_sha256", "target_video_grid_thw"):
        if data.groupby("target_prefix")[column].nunique().max() != 1:
            raise AssertionError(f"target input changed across intervention: {column}")
    alpha0 = data[data["alpha"] == 0]
    alpha0_error = float(alpha0[["delta_gt_margin", "delta_desired_margin",
                                "centered_three_state_logit_l2"]].abs().to_numpy().max())
    if alpha0_error > 1e-5:
        raise AssertionError(f"alpha=0 is not a self intervention: {alpha0_error}")
    if alpha0[["wrong_to_correct", "correct_to_wrong", "flips_to_desired",
               "new_unintended_prediction"]].to_numpy(dtype=bool).any():
        raise AssertionError("alpha=0 changed a categorical prediction")
    donor_mismatch = data.groupby(["target_prefix", "donor_type"])["donor_prefix"].nunique().max()
    if donor_mismatch != 1:
        raise AssertionError("donor changed across alpha")

    alpha1_reproduction = {"available": args.rescue_results.exists(), "max_abs_logit_error": None}
    if args.rescue_results.exists():
        rescue = pd.read_csv(args.rescue_results)
        rescue = rescue[rescue["condition"].isin(
            ["strong_matched_donor", "different_event_matched_history"]
        )].copy()
        rescue["donor_type"] = rescue["condition"].map({
            "strong_matched_donor": "strong_same_state",
            "different_event_matched_history": "different_event",
        })
        current = data[data["alpha"] == 1.0]
        merged = current.merge(rescue, on=["target_prefix", "donor_type"],
                               suffixes=("_dose", "_rescue"), validate="one_to_one")
        if len(merged) != len(expected_targets) * len(DONOR_TYPES):
            raise AssertionError("alpha=1 rescue reproduction coverage mismatch")
        errors = []
        for state in ("Left", "Middle", "Right"):
            errors.extend(np.abs(
                merged[f"patched_logit_{state}_dose"] - merged[f"patched_logit_{state}_rescue"]
            ).tolist())
        alpha1_reproduction["max_abs_logit_error"] = float(max(errors))
        if alpha1_reproduction["max_abs_logit_error"] > 1e-3:
            raise AssertionError(f"alpha=1 does not reproduce rescue v1: {alpha1_reproduction}")

    baseline_entropy = alpha0.set_index(["target_prefix", "donor_type"])["native_three_state_entropy"]
    data = data.join(baseline_entropy.rename("baseline_native_three_state_entropy"),
                     on=["target_prefix", "donor_type"])
    data["delta_native_three_state_entropy"] = (
        data["native_three_state_entropy"] - data["baseline_native_three_state_entropy"]
    )
    data.to_csv(args.out / "dose_response_pair_results.csv", index=False)
    targets = target_summaries(data)
    targets.to_csv(args.out / "dose_response_target_results.csv", index=False)

    effect_rows = []
    adjacent_rows = []
    monotonic_rows = []
    for donor_type in DONOR_TYPES:
        donor_data = data[data["donor_type"] == donor_type]
        target_data = targets[targets["donor_type"] == donor_type]
        endpoint = "delta_gt_margin" if donor_type == "strong_same_state" else "delta_desired_margin"
        for subset_name in SUBSETS:
            selected = subset(donor_data, subset_name)
            selected_targets = subset(target_data, subset_name)
            for alpha in ALPHAS:
                dose = selected[selected["alpha"] == alpha]
                metric_names = [
                    endpoint,
                    "delta_gt_margin",
                    "desired_state_specificity",
                    "centered_three_state_logit_l2",
                    "delta_native_three_state_entropy",
                    "new_unintended_prediction",
                ]
                for metric in dict.fromkeys(metric_names):
                    effect_rows.append({
                        "donor_type": donor_type, "subset": subset_name,
                        "alpha": alpha, "metric": metric,
                        "n_target_prefixes": int(len(dose)),
                        **cluster_stat(trajectory_values(dose, metric)),
                    })
                for metric, numerator, denominator in (
                    ("wrong_to_correct_rate", "wrong_to_correct", "baseline_incorrect_prediction"),
                    ("correct_to_wrong_rate", "correct_to_wrong", "baseline_correct"),
                    ("flips_to_desired_rate", "flips_to_desired", "baseline_not_desired"),
                ):
                    dose = dose.copy()
                    dose["baseline_incorrect_prediction"] = ~dose["baseline_correct"]
                    dose["baseline_not_desired"] = dose["baseline_pred"] != dose["desired_state"]
                    values, denominator_n = conditional_trajectory_rate(dose, numerator, denominator)
                    effect_rows.append({
                        "donor_type": donor_type, "subset": subset_name,
                        "alpha": alpha, "metric": metric,
                        "n_target_prefixes": denominator_n,
                        **cluster_stat(values),
                    })
            for lower, upper in zip(ALPHAS[:-1], ALPHAS[1:]):
                lo = selected[selected["alpha"] == lower].set_index("target_prefix")
                hi = selected[selected["alpha"] == upper].set_index("target_prefix")
                common = lo.index.intersection(hi.index)
                change = pd.DataFrame({
                    "target_prefix": common.to_numpy(),
                    "target_trajectory": hi.loc[common, "target_trajectory"].to_numpy(),
                    "adjacent_change": (
                        hi.loc[common, endpoint] - lo.loc[common, endpoint]
                    ).to_numpy(),
                })
                adjacent_rows.append({
                    "donor_type": donor_type, "subset": subset_name,
                    "alpha_lower": lower, "alpha_upper": upper,
                    "endpoint": endpoint, "n_target_prefixes": int(len(change)),
                    **cluster_stat(trajectory_values(change, "adjacent_change")),
                })
            for metric in ("dose_slope", "monotonic_nondecreasing", "alpha2_endpoint_change",
                           "alpha2_desired_specificity", "alpha2_new_unintended_prediction",
                           "alpha2_centered_logit_l2", "alpha2_entropy_change"):
                monotonic_rows.append({
                    "donor_type": donor_type, "subset": subset_name, "metric": metric,
                    "n_target_prefixes": int(len(selected_targets)),
                    **cluster_stat(trajectory_values(selected_targets, metric)),
                })

    effects = pd.DataFrame(effect_rows)
    adjacent = pd.DataFrame(adjacent_rows)
    monotonic = pd.DataFrame(monotonic_rows)

    crossing = {}
    for donor_type in DONOR_TYPES:
        crossing[donor_type] = {}
        donor_targets = targets[targets["donor_type"] == donor_type]
        for subset_name in SUBSETS:
            selected = subset(donor_targets, subset_name)
            baseline_endpoint = ("baseline_gt_margin" if donor_type == "strong_same_state"
                                 else "baseline_desired_margin")
            eligible = selected[selected[baseline_endpoint] < 0]
            counts = {str(alpha): int((eligible["decision_boundary_alpha"] == alpha).sum()) for alpha in ALPHAS}
            crossing[donor_type][subset_name] = {
                "n_eligible": int(len(eligible)),
                "n_crossed_by_alpha2": int(eligible["decision_boundary_alpha"].notna().sum()),
                "first_crossing_counts": counts,
            }

    def row(frame: pd.DataFrame, donor_type: str, subset_name: str, metric: str,
            alpha: float | None = None) -> dict:
        selected = frame[(frame["donor_type"] == donor_type) & (frame["subset"] == subset_name)
                         & (frame["metric"] == metric)]
        if alpha is not None: selected = selected[selected["alpha"] == alpha]
        if len(selected) != 1: raise AssertionError((donor_type, subset_name, metric, alpha, len(selected)))
        return selected.iloc[0].to_dict()

    criteria = {}
    for donor_type in DONOR_TYPES:
        endpoint = "delta_gt_margin" if donor_type == "strong_same_state" else "delta_desired_margin"
        criteria[donor_type] = {}
        for subset_name in ("overall", "t_ge2", "t_ge3"):
            steps = adjacent[(adjacent["donor_type"] == donor_type) & (adjacent["subset"] == subset_name)]
            slope = row(monotonic, donor_type, subset_name, "dose_slope")
            criteria[donor_type][subset_name] = {
                "endpoint": endpoint,
                "all_adjacent_means_nonnegative": bool((steps["mean"] >= -1e-8).all()),
                "all_adjacent_ci_lows_nonnegative": bool((steps["ci95"].map(lambda ci: ci[0]) >= -1e-8).all()),
                "positive_slope_ci": bool(slope["ci95"][0] > 0),
                "clear_dose_response": bool((steps["mean"] >= -1e-8).all() and slope["ci95"][0] > 0),
            }

    summary = {
        "protocol": "target-prefix effects followed by target-trajectory cluster bootstrap; pairs are not independent",
        "manifest_sha256": file_sha256(args.out / "dose_response_manifest.json"),
        "counts": manifest["counts"],
        "alpha0_max_abs_error": alpha0_error,
        "alpha1_rescue_v1_reproduction": alpha1_reproduction,
        "effects": effect_rows,
        "adjacent_alpha_effects": adjacent_rows,
        "target_monotonicity": monotonic_rows,
        "decision_boundary_crossing": crossing,
        "dose_response_criteria": criteria,
    }
    (args.out / "dose_response_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def fmt(item: dict) -> str:
        if item["mean"] is None: return "NA"
        return f"{item['mean']:+.3f} [{item['ci95'][0]:+.3f},{item['ci95'][1]:+.3f}]"

    report = [
        "# L24 Head 0 dose response", "",
        "Frozen intervention: L24 Head 0, final prompt token, pre-o_proj slice. Target/donor pairs are unchanged from head0_causal_rescue_v1.", "",
        "## Same-state donor: GT-margin movement", "",
        "| alpha | overall | t>=2 | t>=3 | weak | strong | baseline margin<0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in ALPHAS:
        values = [fmt(row(effects, "strong_same_state", subset_name, "delta_gt_margin", alpha)) for subset_name in SUBSETS]
        report.append(f"| {alpha:g} | " + " | ".join(values) + " |")
    report += [
        "", "## Different-event donor: donor-implied counterfactual-margin movement", "",
        "| alpha | overall | t>=2 | t>=3 | weak | strong | baseline margin<0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in ALPHAS:
        values = [fmt(row(effects, "different_event", subset_name, "delta_desired_margin", alpha)) for subset_name in SUBSETS]
        report.append(f"| {alpha:g} | " + " | ".join(values) + " |")
    report += ["", "## Categorical effects", ""]
    for donor_type in DONOR_TYPES:
        report.append(f"### {donor_type}")
        report.append("")
        report.append("| alpha | wrong-to-correct | correct-to-wrong | flips-to-desired | new unintended prediction |")
        report.append("|---:|---:|---:|---:|---:|")
        for alpha in ALPHAS:
            values = [fmt(row(effects, donor_type, "overall", metric, alpha)) for metric in
                      ("wrong_to_correct_rate", "correct_to_wrong_rate", "flips_to_desired_rate",
                       "new_unintended_prediction")]
            report.append(f"| {alpha:g} | " + " | ".join(values) + " |")
        report.append("")
    report += ["## Decision-boundary crossings", ""]
    for donor_type in DONOR_TYPES:
        strict = crossing[donor_type]["overall"]
        counts = ", ".join(
            f"alpha={alpha:g}: {strict['first_crossing_counts'][str(alpha)]}"
            for alpha in ALPHAS
        )
        report.append(
            f"- {donor_type}: {strict['n_crossed_by_alpha2']}/{strict['n_eligible']} "
            f"eligible targets cross by alpha=2; first crossings: {counts}."
        )
    report += ["", "## Large-alpha diagnostics", "",
               "| donor | alpha | desired specificity | centered logit L2 | entropy change | new unintended | harmful flip |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for donor_type in DONOR_TYPES:
        for alpha in (1.0, 1.5, 2.0):
            values = [fmt(row(effects, donor_type, "overall", metric, alpha)) for metric in
                      ("desired_state_specificity", "centered_three_state_logit_l2",
                       "delta_native_three_state_entropy", "new_unintended_prediction",
                       "correct_to_wrong_rate")]
            report.append(f"| {donor_type} | {alpha:g} | " + " | ".join(values) + " |")

    same_clear = all(criteria["strong_same_state"][name]["clear_dose_response"]
                     for name in ("overall", "t_ge2", "t_ge3"))
    different_clear = all(criteria["different_event"][name]["clear_dose_response"]
                          for name in ("overall", "t_ge2", "t_ge3"))
    same_alpha2_margin = row(effects, "strong_same_state", "overall", "delta_gt_margin", 2.0)
    same_alpha2_rescue = row(effects, "strong_same_state", "overall", "wrong_to_correct_rate", 2.0)
    same_alpha2_harm = row(effects, "strong_same_state", "overall", "correct_to_wrong_rate", 2.0)
    strict_cross = crossing["strong_same_state"]["overall"]
    strict_cross_rate = (strict_cross["n_crossed_by_alpha2"] / strict_cross["n_eligible"]
                         if strict_cross["n_eligible"] else 0.0)
    categorical_gain = (
        same_alpha2_rescue["ci95"][0] is not None
        and same_alpha2_rescue["ci95"][0] > 0
    )
    bottleneck_pattern = bool(same_clear and same_alpha2_margin["ci95"][0] > 0
                              and strict_cross_rate < 0.50)
    report += [
        "", "## Answers", "",
        f"**A. Clear dose-response:** {'YES' if same_clear else 'NO'} for same-state GT margin; "
        f"{'YES' if different_clear else 'NO'} for different-event donor-implied margin under the frozen criteria.", "",
        f"**B. Categorical rescue:** {'YES' if categorical_gain else 'NO/RESTRICTED'}. At alpha=2 the "
        f"trajectory-cluster wrong-to-correct estimate is {fmt(same_alpha2_rescue)}, and "
        f"{strict_cross['n_crossed_by_alpha2']}/{strict_cross['n_eligible']} strict negative-margin targets "
        f"cross the decision boundary. The observed harmful-flip estimate is {fmt(same_alpha2_harm)}.", "",
        f"**C. Commitment bottleneck:** {'CONSISTENT' if bottleneck_pattern else 'NOT SUPPORTED BY THE FROZEN PATTERN RULE'}. "
        "The frozen pattern requires a clear continuous response while more than half of initially negative-margin targets remain below boundary at alpha=2; it is not proof of a unique downstream cause.", "",
    ]
    report += [
        "## Frozen decision rules", "",
        "A clear dose response requires direction-consistent adjacent-alpha effects and a positive trajectory-cluster dose-slope CI. Categorical rescue is evaluated separately; continuous margin movement is not treated as behavioral correction. Large-alpha anomalies are assessed using centered three-state logit displacement, desired-state specificity, entropy change, and new unintended predictions.", "",
        "Machine-readable adjacent-step tests, boundary-crossing counts, subgroup results, and anomaly metrics are in `dose_response_summary.json`.", "",
    ]
    (args.out / "dose_response_report.md").write_text("\n".join(report))
    print(json.dumps({
        "status": "DOSE_RESPONSE_ANALYSIS_COMPLETE",
        "rows": len(data),
        "targets": len(expected_targets),
        "target_trajectories": data["target_trajectory"].nunique(),
        "alpha0_max_abs_error": alpha0_error,
        "alpha1_rescue_v1_reproduction": alpha1_reproduction,
        "criteria": criteria,
        "outputs": ["dose_response_pair_results.csv", "dose_response_target_results.csv",
                    "dose_response_summary.json", "dose_response_report.md"],
    }, indent=2))


def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
