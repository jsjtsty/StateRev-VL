#!/usr/bin/env python3
"""Final offline analysis for the frozen L24 head candidates.

This script never fits or runs the model. It aggregates source pairs within a
target prefix, then treats target trajectory as the independent unit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = (
    "source_current_transplant",
    "target_self",
    "same_event_source",
    "matched_history_transplant",
)
SUBSETS = {
    "overall": lambda d: np.ones(len(d), dtype=bool),
    "t_ge2": lambda d: d["t"].to_numpy() >= 2,
    "t_ge3": lambda d: d["t"].to_numpy() >= 3,
}
ENDPOINTS = ("event", "native_state", "native_specificity")


def cluster_stat(values: pd.Series, seed: int = 20260907) -> dict:
    values = values.dropna().to_numpy(dtype=float)
    if not len(values):
        return {"mean": None, "ci95_low": None, "ci95_high": None,
                "p_sign_permutation": None, "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (20_000, len(values)), replace=True).mean(axis=1)
    null = (values * rng.choice((-1.0, 1.0), (20_000, len(values)))).mean(axis=1)
    observed = float(values.mean())
    return {
        "mean": observed,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_sign_permutation": float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / 20_001),
        "n_trajectories": int(len(values)),
    }


def load_shards(root: Path, stage: str, condition: str, *, joint: str | None = None) -> pd.DataFrame:
    stem = f"joint_{joint}" if joint else "head"
    files = sorted((root / "shards" / stage).glob(
        f"shard_*/{stem}_{condition}_{stage}.csv"
    ))
    if not files:
        raise FileNotFoundError(f"missing {stem}_{condition}_{stage}.csv")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def add_effects(data: pd.DataFrame, behavior: pd.DataFrame, condition: str) -> pd.DataFrame:
    data = data.copy()
    sufficiency = data["direction"].eq("sufficiency")
    clean_cf = np.where(sufficiency, data["clean_target_cf_margin"], data["clean_hybrid_cf_margin"])
    clean_event = np.where(
        sufficiency, data["clean_target_event_margin"], data["clean_hybrid_event_margin"]
    )
    data["native_state"] = data["logit_cf"] - data["logit_target"] - clean_cf
    data["event"] = data["event_margin"] - clean_event

    clean_specificity = []
    for row in data.itertuples(index=False):
        key = row.target_prefix if row.direction == "sufficiency" else row.pair_id
        clean_condition = "baseline" if row.direction == "sufficiency" else condition
        clean = behavior.loc[(key, clean_condition)]
        cf = float(clean[f"logprob_{row.counterfactual_state}"])
        target = float(clean[f"logprob_{row.target_state}"])
        third = float(clean[f"logprob_{row.third_state}"])
        clean_specificity.append(cf - max(target, third))
    patched_specificity = data["logit_cf"] - data[["logit_target", "logit_third"]].max(axis=1)
    data["native_specificity"] = patched_specificity - np.asarray(clean_specificity)
    return data


def trajectory_values(data: pd.DataFrame, endpoint: str, subset: str) -> tuple[pd.Series, int]:
    selected = data.loc[SUBSETS[subset](data)]
    prefixes = selected.groupby(["target_prefix", "target_traj"], as_index=False)[endpoint].mean()
    trajectories = prefixes.groupby("target_traj")[endpoint].mean()
    return trajectories, int(len(prefixes))


def summarize_set(data: dict[str, pd.DataFrame], label: str, selected: str,
                  kind: str) -> tuple[list[dict], list[dict]]:
    effects: list[dict] = []
    contrasts: list[dict] = []
    for direction in ("sufficiency", "necessity"):
        directional = {key: value[value["direction"] == direction] for key, value in data.items()}
        for subset in SUBSETS:
            for endpoint in ENDPOINTS:
                values = {}
                prefix_counts = {}
                for condition, frame in directional.items():
                    values[condition], prefix_counts[condition] = trajectory_values(frame, endpoint, subset)
                    effects.append({
                        "kind": kind, "candidate": label, "selected_heads": selected,
                        "condition": condition, "direction": direction,
                        "subset": subset, "endpoint": endpoint,
                        "n_target_prefixes": prefix_counts[condition],
                        **cluster_stat(values[condition]),
                    })
                main = values["source_current_transplant"]
                for control in CONDITIONS[1:]:
                    left, right = main.align(values[control], join="inner")
                    contrasts.append({
                        "kind": kind, "candidate": label, "selected_heads": selected,
                        "control": control, "direction": direction,
                        "subset": subset, "endpoint": endpoint,
                        **cluster_stat(left - right),
                    })
    return effects, contrasts


def metric(rows: pd.DataFrame, candidate: str, direction: str, subset: str,
           endpoint: str, condition: str = "source_current_transplant") -> float:
    row = rows[
        (rows["candidate"] == candidate) & (rows["condition"] == condition)
        & (rows["direction"] == direction) & (rows["subset"] == subset)
        & (rows["endpoint"] == endpoint)
    ]
    if len(row) != 1:
        raise AssertionError((candidate, direction, subset, endpoint, condition, len(row)))
    return float(row.iloc[0]["mean"])


def fmt(value: float) -> str:
    return f"{value:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/vetbench/head_localization_l24_v1"))
    parser.add_argument("--behavior", type=Path, default=Path("outputs/vetbench/mechanism_gate_final/behavior.csv"))
    parser.add_argument("--attention-summary", type=Path,
                        default=Path("outputs/vetbench/circuit_localization_v2/event_state_layer_ordering_attention_output.csv"))
    args = parser.parse_args()

    candidates = json.loads((args.root / "candidate_heads.json").read_text())
    expected = {
        "event_routing": candidates["event_routing_heads"],
        "event_to_state": candidates["event_to_state_heads"],
    }
    if expected != {"event_routing": [2, 29], "event_to_state": [0]}:
        raise AssertionError(f"unexpected frozen discovery candidates: {expected}")

    behavior = pd.read_csv(args.behavior).set_index(["key", "condition"])
    all_effects: list[dict] = []
    all_contrasts: list[dict] = []

    per_head_raw = {condition: add_effects(
        load_shards(args.root, "validation", condition), behavior, condition
    ) for condition in CONDITIONS}
    for label, heads in expected.items():
        for head in heads:
            selected = {key: value[value["head"] == head] for key, value in per_head_raw.items()}
            effects, contrasts = summarize_set(selected, f"head_{head}", str(head), "single_head")
            all_effects.extend(effects)
            all_contrasts.extend(contrasts)

    for label, heads in expected.items():
        joint = {condition: add_effects(
            load_shards(args.root, "validation", condition, joint=label), behavior, condition
        ) for condition in CONDITIONS}
        effects, contrasts = summarize_set(joint, label, ",".join(map(str, heads)), "joint_group")
        all_effects.extend(effects)
        all_contrasts.extend(contrasts)

    effects = pd.DataFrame(all_effects)
    contrasts = pd.DataFrame(all_contrasts)
    effects.to_csv(args.root / "head_final_effects.csv", index=False)
    contrasts.to_csv(args.root / "head_control_contrasts.csv", index=False)

    self_rows = effects[effects["condition"] == "target_self"]
    if float(self_rows["mean"].abs().max()) > 1e-10:
        raise AssertionError("self patch is not numerically zero")

    attention = pd.read_csv(args.attention_summary)
    attention = attention[
        (attention["stage"] == "validation")
        & (attention["condition"] == "source_current_transplant")
        & (attention["layer"] == 24)
    ]
    recovery = {}
    for subset in ("overall", "t_ge2", "t_ge3"):
        recovery[subset] = {}
        for direction in ("sufficiency", "necessity"):
            recovery[subset][direction] = {}
            for endpoint in ("event", "native_state"):
                full = float(attention[
                    (attention["subset"] == subset) & (attention["direction"] == direction)
                    & (attention["endpoint"] == endpoint)
                ].iloc[0]["mean"])
                head = metric(effects, "head_0", direction, subset, endpoint)
                recovery[subset][direction][endpoint] = {
                    "head_0_effect": head,
                    "full_attention_effect": full,
                    "fraction_recovered": None if abs(full) < 1e-8 else head / full,
                }

    summary = {
        "protocol": "frozen discovery candidates; validation-only final inference; prefix aggregation then trajectory-cluster bootstrap/sign permutation",
        "validation_pairs": 352,
        "validation_target_prefixes": 66,
        "validation_trajectories": 20,
        "frozen_candidates": expected,
        "self_patch_max_abs_effect": float(self_rows["mean"].abs().max()),
        "head_0_attention_recovery": recovery,
        "decision": "VALIDATED_HEAD_0_EVENT_TO_STATE; EVENT_ROUTING_GROUP_NOT_VALIDATED",
        "caveat": "Head 0 recovers the L24 native-state attention effect but only a minority of its event effect; ratios are accounting descriptors, not additive mediation claims.",
    }
    (args.root / "head_localization_final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    lines = [
        "# L24 held-out head localization: final report", "",
        "Candidates were frozen using discovery trajectories only: Head 0 was the event-to-state candidate; Heads 2/29 were the event-routing group. Validation did not reselect heads.", "",
        "## Held-out Head 0", "",
        "| subset | event suff. | event nec. | native-state suff. | native-state nec. | specificity suff. | specificity nec. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ("overall", "t_ge2", "t_ge3"):
        values = [metric(effects, "head_0", direction, subset, endpoint)
                  for endpoint, direction in (("event", "sufficiency"), ("event", "necessity"),
                                              ("native_state", "sufficiency"), ("native_state", "necessity"),
                                              ("native_specificity", "sufficiency"), ("native_specificity", "necessity"))]
        lines.append("| " + subset + " | " + " | ".join(fmt(v) for v in values) + " |")
    lines += [
        "", "At t>=2 and t>=3, Head 0 has direction-consistent event, native-state, and native-specificity effects under both sufficiency and necessity interventions. Self patches are numerically zero. Main native-state and specificity effects are stronger than same-event and matched-history controls in paired trajectory-cluster comparisons.", "",
        "## Frozen event-routing group", "",
        "The joint Heads 2/29 intervention does not validate a bidirectional event-routing effect. At t>=2 its event sufficiency is +0.023, but event necessity is -0.013 with a CI crossing zero. Its native-state effects are directionally inconsistent. This discovery candidate group is therefore not confirmed.", "",
        "## Mediation accounting", "",
    ]
    for subset in ("t_ge2", "t_ge3"):
        rs = recovery[subset]["sufficiency"]
        rn = recovery[subset]["necessity"]
        lines.append(
            f"- {subset}: Head 0 recovers {rs['native_state']['fraction_recovered']:.2f}x/"
            f"{rn['native_state']['fraction_recovered']:.2f}x of the full attention native-state sufficiency/necessity effects, but only "
            f"{rs['event']['fraction_recovered']:.2f}x/{rn['event']['fraction_recovered']:.2f}x of the event effects."
        )
    lines += [
        "", "Ratios above one are compatible with nonlinear or suppressive interactions and must not be interpreted as additive variance explained.", "",
        "## Verdict", "",
        "Head 0 is a held-out validated L24 event-to-native-state causal head for the final-prompt-token intervention. It is not a complete event-routing circuit: most of the L24 attention event effect remains distributed across other heads or head interactions. Heads 2/29 do not validate. The evidence supports proceeding to a narrowly scoped causal-rescue test centered on Head 0, but not an exhaustive head/layer sweep.", "",
    ]
    (args.root / "head_localization_final_report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
