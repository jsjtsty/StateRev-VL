#!/usr/bin/env python3
"""Offline target/trajectory-cluster analysis for Head 0 causal rescue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260908
MAIN = "strong_matched_donor"
CONTROLS = ("self_patch", "ordinary_matched_donor", "shuffled_matched_donor",
            "different_event_matched_history")


def cluster_stat(values: pd.Series, seed: int = SEED) -> dict:
    values = values.dropna().to_numpy(dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "p_sign_permutation": None,
                "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (20_000, len(values)), replace=True).mean(axis=1)
    null = (values * rng.choice((-1.0, 1.0), (20_000, len(values)))).mean(axis=1)
    mean = float(values.mean())
    return {
        "mean": mean,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_sign_permutation": float((np.count_nonzero(np.abs(null) >= abs(mean)) + 1) / 20_001),
        "n_trajectories": int(len(values)),
    }


def trajectory_values(data: pd.DataFrame, metric: str) -> pd.Series:
    prefix = data.groupby(["target_prefix", "target_trajectory"], as_index=False)[metric].mean()
    return prefix.groupby("target_trajectory")[metric].mean()


def subset(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "overall": return data
    if name == "t_ge2": return data[data["t"] >= 2]
    if name == "t_ge3": return data[data["t"] >= 3]
    if name == "weak": return data[data["strength_group"] == "weak"]
    if name == "strong": return data[data["strength_group"] == "strong"]
    if name == "baseline_incorrect": return data[data["baseline_gt_margin"] < 0]
    raise KeyError(name)


def conditional_rate(data: pd.DataFrame, event: str) -> pd.Series:
    if event == "wrong_to_correct_rate":
        data = data[~data["baseline_correct"]].copy()
        data[event] = data["wrong_to_correct"].astype(float)
    elif event == "correct_to_wrong_rate":
        data = data[data["baseline_correct"]].copy()
        data[event] = data["correct_to_wrong"].astype(float)
    else:
        raise KeyError(event)
    return trajectory_values(data, event)


def association_stat(data: pd.DataFrame) -> dict:
    """Average within-trajectory slope of gain on baseline margin."""
    slopes = []
    for trajectory, group in data.groupby("target_trajectory"):
        x = group["baseline_gt_margin"].to_numpy(float)
        y = group["delta_gt_margin"].to_numpy(float)
        x = x - x.mean()
        denominator = float(x @ x)
        if denominator > 1e-12:
            slopes.append((trajectory, float((x @ (y - y.mean())) / denominator)))
    series = pd.Series(dict(slopes), dtype=float)
    result = cluster_stat(series)
    result["definition"] = "mean within-target-trajectory OLS slope: delta_gt_margin ~ baseline_gt_margin"
    result["supports_weaker_gets_larger_gain"] = bool(result["ci95"][1] < 0) if result["ci95"][1] is not None else False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/vetbench/head0_causal_rescue_v1"))
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((args.out / "rescue_manifest.json").read_text())
    files = [args.out / "shards" / f"rescue_shard_{index}.csv" for index in range(args.num_shards)]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing rescue shards: {missing}")
    pair_results = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    expected_targets = {row["target_prefix"] for row in manifest["targets"]}
    if set(pair_results["target_prefix"]) != expected_targets:
        raise AssertionError("rescue output target coverage differs from frozen manifest")
    if len(pair_results) != len(expected_targets) * 5:
        raise AssertionError("expected exactly five intervention conditions per target")
    if pair_results.duplicated(["target_prefix", "condition"]).any():
        raise AssertionError("duplicate target-condition rows")
    if pair_results["target_trajectory"].nunique() != manifest["counts"]["target_trajectories"]:
        raise AssertionError("trajectory coverage mismatch")
    self_rows = pair_results[pair_results["condition"] == "self_patch"]
    self_error = float(self_rows[["delta_gt_margin", "gt_state_specificity"]].abs().to_numpy().max())
    if self_error > 1e-5:
        raise AssertionError(f"self patch effect too large: {self_error}")

    pair_results.to_csv(args.out / "rescue_pair_results.csv", index=False)
    numeric = [
        "baseline_gt_margin", "patched_gt_margin", "delta_gt_margin",
        "delta_logit_gt", "delta_logit_other_max", "gt_state_specificity",
        "wrong_to_correct", "correct_to_wrong", "counterfactual_margin_shift",
    ]
    target_aggregated = pair_results.groupby(
        ["target_prefix", "target_trajectory", "t", "condition", "strength_group",
         "baseline_correct", "gt_state"], as_index=False, dropna=False
    )[numeric].mean()
    target_aggregated.to_csv(args.out / "rescue_target_aggregated.csv", index=False)

    subset_names = ("overall", "t_ge2", "t_ge3", "weak", "strong", "baseline_incorrect")
    metrics = ("delta_gt_margin", "gt_state_specificity")
    effect_rows = []
    contrast_rows = []
    for subset_name in subset_names:
        for condition, condition_data in target_aggregated.groupby("condition"):
            selected = subset(condition_data, subset_name)
            for metric in metrics:
                values = trajectory_values(selected, metric)
                effect_rows.append({"subset": subset_name, "condition": condition,
                                    "metric": metric, "n_target_prefixes": len(selected),
                                    **cluster_stat(values)})
            for rate in ("wrong_to_correct_rate", "correct_to_wrong_rate"):
                values = conditional_rate(selected, rate)
                denominator = int((~selected["baseline_correct"]).sum()) if rate.startswith("wrong") else int(selected["baseline_correct"].sum())
                effect_rows.append({"subset": subset_name, "condition": condition,
                                    "metric": rate, "n_target_prefixes": denominator,
                                    **cluster_stat(values)})
        main_data = subset(target_aggregated[target_aggregated["condition"] == MAIN], subset_name)
        for control in CONTROLS:
            control_data = subset(target_aggregated[target_aggregated["condition"] == control], subset_name)
            for metric in metrics:
                left = trajectory_values(main_data, metric)
                right = trajectory_values(control_data, metric)
                left, right = left.align(right, join="inner")
                contrast_rows.append({"subset": subset_name, "control": control,
                                      "metric": metric, **cluster_stat(left - right)})

    # A paired weak-minus-strong contrast within trajectories.
    main = target_aggregated[target_aggregated["condition"] == MAIN]
    weak = trajectory_values(main[main["strength_group"] == "weak"], "delta_gt_margin")
    strong = trajectory_values(main[main["strength_group"] == "strong"], "delta_gt_margin")
    weak, strong = weak.align(strong, join="inner")
    weak_strong = cluster_stat(weak - strong)
    weak_strong["definition"] = "strong-donor rescue gain: weak targets minus strong targets, paired by trajectory"
    association = association_stat(main)

    different_cf = {}
    different = target_aggregated[
        target_aggregated["condition"] == "different_event_matched_history"
    ]
    for subset_name in subset_names:
        selected = subset(different, subset_name)
        different_cf[subset_name] = {
            "n_target_prefixes": int(len(selected)),
            **cluster_stat(trajectory_values(selected, "counterfactual_margin_shift")),
        }

    effects = pd.DataFrame(effect_rows)
    contrasts = pd.DataFrame(contrast_rows)
    summary = {
        "protocol": "donor rows aggregated within target prefix; target trajectory is the independent cluster",
        "independence_caveat": manifest["independence_caveat"],
        "counts": manifest["counts"],
        "self_patch_max_abs_effect": self_error,
        "effects": effect_rows,
        "main_vs_controls": contrast_rows,
        "weak_minus_strong": weak_strong,
        "baseline_margin_gain_association": association,
        "different_event_counterfactual_movement": different_cf,
    }
    (args.out / "rescue_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def get(frame: pd.DataFrame, subset_name: str, condition: str, metric: str) -> pd.Series:
        row = frame[(frame["subset"] == subset_name) & (frame["condition"] == condition)
                    & (frame["metric"] == metric)]
        if len(row) != 1: raise AssertionError((subset_name, condition, metric))
        return row.iloc[0]
    def show(row: pd.Series) -> str:
        if row["mean"] is None or pd.isna(row["mean"]):
            return "NA"
        return f"{row['mean']:+.3f} [{row['ci95'][0]:+.3f},{row['ci95'][1]:+.3f}]"

    report = [
        "# L24 Head 0 causal rescue", "",
        manifest["independence_caveat"], "",
        "## Strong matched-donor rescue", "",
        "| subset | delta GT margin | GT-state specificity | wrong-to-correct | correct-to-wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in subset_names:
        delta = get(effects, name, MAIN, "delta_gt_margin")
        specificity = get(effects, name, MAIN, "gt_state_specificity")
        rescue = get(effects, name, MAIN, "wrong_to_correct_rate")
        harmful = get(effects, name, MAIN, "correct_to_wrong_rate")
        report.append(f"| {name} | {show(delta)} | {show(specificity)} | {show(rescue)} | {show(harmful)} |")
    report += [
        "", "## Weak-target test", "",
        f"Paired weak-minus-strong gain: {weak_strong['mean']:+.3f}, 95% CI [{weak_strong['ci95'][0]:+.3f},{weak_strong['ci95'][1]:+.3f}], p={weak_strong['p_sign_permutation']:.4f}.",
        f"Within-trajectory baseline-margin slope: {association['mean']:+.3f}, 95% CI [{association['ci95'][0]:+.3f},{association['ci95'][1]:+.3f}], p={association['p_sign_permutation']:.4f}. A reliably negative slope supports larger rescue for weaker baselines.",
        "", "## Controls", "",
        "Paired main-minus-control trajectory-cluster results are stored in `rescue_summary.json`. The different-event condition is additionally interpreted through `counterfactual_margin_shift` in both `rescue_summary.json` and `rescue_pair_results.csv`; it is not treated as a same-state rescue control.",
        "", "## Interpretation rule", "",
        "Selective rescue requires positive delta GT margin and GT-state specificity, improved wrong-to-correct rate without a comparable harmful-flip increase, stronger gains for weak targets, and effects exceeding self/ordinary/shuffled controls. Different-event activation should not rescue GT and may instead move the counterfactual margin.", "",
    ]
    (args.out / "head0_causal_rescue_report.md").write_text("\n".join(report))
    print(json.dumps({"status": "RESCUE_ANALYSIS_COMPLETE", "counts": manifest["counts"],
                      "self_patch_max_abs_effect": self_error,
                      "outputs": ["rescue_pair_results.csv", "rescue_target_aggregated.csv",
                                  "rescue_summary.json", "head0_causal_rescue_report.md"]}, indent=2))


if __name__ == "__main__":
    main()
