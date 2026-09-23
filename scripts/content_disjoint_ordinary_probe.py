#!/usr/bin/env python3
"""Ordinary linear probes under the fixed content-disjoint split.

This is the conventional per-layer StandardScaler + L2 logistic-regression
readout, but every C choice and fitted transform stays within discovery
trajectories.  It consumes the validated state-question hidden cache and does
not perform model forward passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from content_disjoint_split import assert_content_disjoint

EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
STATES = ("Left", "Middle", "Right")


def choose_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, candidates: tuple[float, ...]) -> tuple[float, list[float]]:
    splitter = GroupKFold(n_splits=3)
    best: tuple[float, float, list[float]] | None = None
    for c in candidates:
        scores = []
        for train, test in splitter.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            clf = LogisticRegression(C=c, max_iter=2000, random_state=20260918).fit(scaler.transform(x[train]), y[train])
            scores.append(float((clf.predict(scaler.transform(x[test])) == y[test]).mean()))
        candidate = (float(np.mean(scores)), -c, scores)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return -best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"))
    parser.add_argument("--fingerprints", type=Path, default=Path("outputs/vetbench/validity_gate_v1/input_fingerprints.csv"))
    parser.add_argument("--hidden-cache", type=Path, default=Path("outputs/vetbench/hidden_state_probe/hidden_states.npz"))
    parser.add_argument("--behavior", type=Path, default=Path("outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv"))
    parser.add_argument("--candidate-c", default="0.1,1.0,10.0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    assert_content_disjoint(split, args.fingerprints, args.hidden_cache)
    discovery, validation = set(split["discovery_trajectories"]), set(split["validation_trajectories"])
    behavior = pd.read_csv(args.behavior).copy()
    behavior["target_prefix"] = behavior.trajectory_id + "_t" + behavior.t.astype(str)
    cache = np.load(args.hidden_cache, allow_pickle=False)
    train = behavior[behavior.trajectory_id.isin(discovery)].sort_values("target_prefix").reset_index(drop=True)
    test = behavior[behavior.trajectory_id.isin(validation)].sort_values("target_prefix").reset_index(drop=True)
    candidates = tuple(float(x) for x in args.candidate_c.split(",") if x.strip())
    rows = []
    for target, classes in (("event", EVENTS), ("state", STATES)):
        y_train = train[f"gt_{target}"].to_numpy()
        y_test = test[f"gt_{target}"].to_numpy()
        for layer in range(37):
            x_train = np.stack([cache[key][layer] for key in train.target_prefix]).astype(np.float64)
            x_test = np.stack([cache[key][layer] for key in test.target_prefix]).astype(np.float64)
            c, cv = choose_c(x_train, y_train, train.trajectory_id.to_numpy(), candidates)
            scaler = StandardScaler().fit(x_train)
            clf = LogisticRegression(C=c, max_iter=2000, random_state=20260918).fit(scaler.transform(x_train), y_train)
            prediction = clf.predict(scaler.transform(x_test))
            rows.append({"target": target, "layer": layer, "C": c, "discovery_group_cv_accuracy": float(np.mean(cv)), "validation_accuracy": float((prediction == y_test).mean()), "validation_balanced_accuracy": float(np.mean([(prediction[y_test == label] == label).mean() for label in classes])), "discovery_rows": len(train), "validation_rows": len(test)})
    result = pd.DataFrame(rows)
    result.to_csv(args.out / "ordinary_probe_metrics.csv", index=False)
    best = result.sort_values(["target", "validation_accuracy", "layer"], ascending=[True, False, True]).groupby("target", as_index=False).first()
    summary = {"protocol": "content-disjoint frozen ordinary probe; C selected by discovery GroupKFold only", "split": str(args.split), "counts": {"discovery_trajectories": len(discovery), "validation_trajectories": len(validation), "discovery_prefixes": len(train), "validation_prefixes": len(test)}, "best_validation_layers_descriptive_only": best.to_dict(orient="records"), "content_disjoint_assertion": "PASS"}
    (args.out / "ordinary_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Content-disjoint ordinary probe", "", "All scalers, C choices, and classifiers are fitted only on discovery trajectories. Validation accuracy is held-out; the listed best validation layer is descriptive and must not be used for a downstream selection.", "", result.to_markdown(index=False), ""]
    (args.out / "ordinary_probe_report.md").write_text("\n".join(lines))
    print(json.dumps({"ORDINARY_PROBE_PASS": True, "out": str(args.out), "rows": len(result), "content_disjoint": True}, indent=2))


if __name__ == "__main__":
    main()
