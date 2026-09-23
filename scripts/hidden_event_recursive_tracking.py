#!/usr/bin/env python3
"""Frozen hidden-event probe -> recursive state tracking.

The script is deliberately an offline companion to the existing event-probe
and recursive-state experiments. It reads the already extracted hidden-state
NPZ and the existing explicit-event probability shards. A decoder is fit only
on the fixed discovery trajectories; prediction/analysis commands reject
discovery rows, so validation cannot silently become a tuning set.

The current StateRev-VL cache uses three *unordered* swap-event classes:
``Left and Middle``, ``Middle and Right`` and ``Left and Right``. This script
does not invent a six-class label mapping. The schema is recorded in every
artifact and report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


STATES = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
PAIRS = {
    "Left and Middle": ("Left", "Middle"),
    "Middle and Right": ("Middle", "Right"),
    "Left and Right": ("Left", "Right"),
}
SEED = 20260917
N_TRAJECTORIES = 50
N_PREFIXES = 250
N_STEPS = 5

STATE_METHOD_COLUMNS = {
    "native_state": "native_state_correct",
    "old_symbolic": "old_symbolic_correct",
    "native_explicit_hard": "native_hard_state_correct",
    "native_explicit_prob": "native_prob_state_correct",
    "hidden_event_hard": "hidden_hard_state_correct",
    "hidden_event_prob": "hidden_prob_state_correct",
    "oracle_event_hard": "oracle_hard_state_correct",
    "oracle_event_prob": "oracle_prob_state_correct",
}
EVENT_METHOD_COLUMNS = {
    "explicit_behavior_argmax": "explicit_event_correct",
    "explicit_cached_prob_argmax": "native_prob_event_correct",
    "hidden_probe_argmax": "hidden_event_correct",
}


def update(state: str, event: str) -> str:
    """Apply one unordered swap to a state."""
    if state not in STATES:
        raise ValueError(f"unknown state: {state!r}")
    if event not in PAIRS:
        raise ValueError(f"unknown event: {event!r}")
    left, right = PAIRS[event]
    return right if state == left else left if state == right else state


def normalize_probability(p: Sequence[float], *, name: str = "probability") -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all() or (arr < -1e-8).any():
        raise ValueError(f"{name} contains invalid values: {arr!r}")
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(f"{name} has zero mass")
    return arr / total


def transition(state_belief: Sequence[float], event_probability: Sequence[float]) -> np.ndarray:
    """Propagate a Left/Middle/Right belief through an event distribution."""
    sb = normalize_probability(state_belief, name="state belief")
    ep = normalize_probability(event_probability, name="event probability")
    out = np.zeros(3, dtype=float)
    for i, state in enumerate(STATES):
        for j, event in enumerate(EVENTS):
            out[STATES.index(update(state, event))] += sb[i] * ep[j]
    return normalize_probability(out, name="next state belief")


def one_hot(index: int) -> np.ndarray:
    result = np.zeros(3, dtype=float)
    result[index] = 1.0
    return result


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty list")
    return result


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty list")
    return result


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=Path("outputs/vetbench/hidden_event_recursive_v1"))
    parser.add_argument("--behavior", type=Path, default=Path("outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/vetbench/circuit_localization_v2/discovery_validation_split.json"))
    parser.add_argument("--fingerprint-file", type=Path, default=None,
                        help="require a content-disjoint split against this fingerprint CSV")
    parser.add_argument("--hidden-cache", type=Path, default=Path("outputs/vetbench/hidden_state_probe/hidden_states.npz"))
    parser.add_argument("--native-event-prob-dir", type=Path, default=Path("outputs/vetbench/recursive_state_recovery_v1/shards"))
    parser.add_argument("--symbolic", type=Path, default=Path("outputs/vetbench/composition_analysis_v1/symbolic_composition.csv"))
    parser.add_argument("--artifact", type=Path, default=None)


def target_prefixes(behavior: pd.DataFrame) -> pd.Series:
    return behavior["trajectory_id"].astype(str) + "_t" + behavior["t"].astype(str)


def validate_split(behavior: pd.DataFrame, split: Mapping[str, object]) -> tuple[set[str], set[str]]:
    discovery = set(split["discovery_trajectories"])
    validation = set(split["validation_trajectories"])
    observed = set(behavior["trajectory_id"])
    if discovery & validation:
        raise AssertionError("discovery and validation trajectories overlap")
    if discovery | validation != observed:
        raise AssertionError("split does not cover exactly the behavior trajectories")
    if len(discovery) != 30 or len(validation) != 20:
        raise AssertionError(f"expected 30/20 split, got {len(discovery)}/{len(validation)}")
    counts = behavior.groupby("trajectory_id")["t"].agg(list)
    if len(counts) != N_TRAJECTORIES or any(sorted(x) != list(range(1, N_STEPS + 1)) for x in counts):
        raise AssertionError("expected exactly t=1..5 for every trajectory")
    return discovery, validation


def assert_content_disjoint_split(split: Mapping[str, object], fingerprint_file: Path | None, hidden_cache: Path) -> None:
    """Reject split manifests whose train/validation sides reuse exact pixels."""
    manifest_fingerprint = split.get("fingerprint_source")
    fingerprint_file = fingerprint_file or (Path(str(manifest_fingerprint)) if manifest_fingerprint else None)
    if fingerprint_file is None:
        return
    from content_disjoint_split import assert_content_disjoint
    assert_content_disjoint(dict(split), fingerprint_file, hidden_cache)


def load_native_probability_table(prob_dir: Path, expected_keys: Iterable[str]) -> pd.DataFrame:
    files = sorted(prob_dir.glob("event_probs_shard_*.csv"))
    if not files:
        raise FileNotFoundError(f"no event probability shards in {prob_dir}")
    tables = []
    for path in files:
        marker = path.with_suffix(".complete.json")
        if not marker.exists():
            raise FileNotFoundError(f"missing completion marker for {path}")
        try:
            metadata = json.loads(marker.read_text())
        except json.JSONDecodeError as exc:
            raise AssertionError(f"invalid completion marker: {marker}") from exc
        table = pd.read_csv(path)
        if "rows" in metadata and int(metadata["rows"]) != len(table):
            raise AssertionError(f"marker row count mismatch for {path}")
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    pcols = [f"event_prob_{event}" for event in EVENTS]
    required = {"target_prefix", "trajectory_id", "t", *pcols}
    missing = required - set(result.columns)
    if missing:
        raise AssertionError(f"native event probability schema missing {sorted(missing)}")
    if result["target_prefix"].duplicated().any():
        raise AssertionError("duplicate target prefixes in native event probability shards")
    expected = set(expected_keys)
    if set(result["target_prefix"]) != expected:
        raise AssertionError("native event probability shards do not cover behavior keys")
    probabilities = result[pcols].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < -1e-8).any():
        raise AssertionError("native event probability shard has invalid values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("native event probabilities are not normalized")
    return result.sort_values("target_prefix").reset_index(drop=True)


def load_inputs(args: argparse.Namespace) -> dict[str, object]:
    behavior = pd.read_csv(args.behavior)
    required = {"trajectory_id", "t", "initial_state", "gt_event", "gt_prev_state", "gt_state", "event_pred", "state_pred"}
    missing = required - set(behavior.columns)
    if missing:
        raise AssertionError(f"behavior schema missing {sorted(missing)}")
    behavior["t"] = behavior["t"].astype(int)
    behavior["target_prefix"] = target_prefixes(behavior)
    if len(behavior) != N_PREFIXES or behavior["target_prefix"].duplicated().any():
        raise AssertionError("expected 250 unique behavior prefixes")
    if set(behavior["initial_state"]) - set(STATES):
        raise AssertionError("unknown initial state in behavior cache")
    if set(behavior["gt_event"]) != set(EVENTS) or set(behavior["event_pred"]) - set(EVENTS):
        raise AssertionError("expected current three-class event schema")
    split = json.loads(args.split.read_text())
    discovery, validation = validate_split(behavior, split)

    assert_content_disjoint_split(split, args.fingerprint_file, args.hidden_cache)

    hidden_cache = np.load(args.hidden_cache, allow_pickle=False)
    if len(hidden_cache.files) != N_PREFIXES:
        raise AssertionError(f"expected {N_PREFIXES} hidden cache keys, got {len(hidden_cache.files)}")
    bad_shapes = [key for key in hidden_cache.files if hidden_cache[key].shape != (37, 4096)]
    if bad_shapes:
        raise AssertionError(f"hidden cache shape mismatch, examples: {bad_shapes[:3]}")
    if set(hidden_cache.files) != set(behavior["target_prefix"]):
        raise AssertionError("hidden cache keys do not match behavior prefixes")

    native_probs = load_native_probability_table(args.native_event_prob_dir, behavior["target_prefix"])
    symbolic = pd.read_csv(args.symbolic)
    if "key" not in symbolic or "symbolic_acc" not in symbolic:
        raise AssertionError("old symbolic cache needs key and symbolic_acc columns")
    if symbolic["key"].duplicated().any() or set(symbolic["key"]) != set(behavior["target_prefix"]):
        raise AssertionError("old symbolic cache does not cover behavior prefixes")
    symbolic = symbolic[["key", "symbolic_acc"]].copy()
    symbolic["old_symbolic_correct"] = symbolic["symbolic_acc"].astype(float).eq(1.0)
    return {"behavior": behavior, "split": split, "discovery": discovery, "validation": validation, "hidden_cache": hidden_cache, "native_probs": native_probs, "symbolic": symbolic}


def feature_matrix(rows: pd.DataFrame, cache: object, layer: int) -> np.ndarray:
    return np.stack([np.asarray(cache[key][layer], dtype=np.float64) for key in rows["target_prefix"]])


def fit_frozen_decoder(
    inputs: Mapping[str, object], artifact_path: Path,
    candidate_layers: Sequence[int] = (24, 28), candidate_pca_dims: Sequence[int] = (0, 40), candidate_cs: Sequence[float] = (0.1, 1.0, 10.0),
) -> dict[str, object]:
    """Select and fit the probe using discovery trajectories only."""
    behavior, discovery, cache = inputs["behavior"], inputs["discovery"], inputs["hidden_cache"]
    train = behavior[behavior["trajectory_id"].isin(discovery)].sort_values("target_prefix").reset_index(drop=True)
    y = train["gt_event"].to_numpy()
    groups = train["trajectory_id"].to_numpy()
    if set(y) != set(EVENTS):
        raise AssertionError("discovery fit rows do not contain all event classes")
    layers, pca_dims, cs = tuple(map(int, candidate_layers)), tuple(map(int, candidate_pca_dims)), tuple(map(float, candidate_cs))
    if any(x < 0 or x > 36 for x in layers):
        raise ValueError("candidate layer must be in [0,36]")
    if any(x < 0 for x in pca_dims) or any(x <= 0 for x in cs):
        raise ValueError("PCA dimensions must be nonnegative and C must be positive")
    splitter = GroupKFold(n_splits=min(3, len(set(groups))))
    scores: list[dict[str, object]] = []
    best: tuple[tuple[float, int, int, float], dict[str, object]] | None = None
    for layer in layers:
        for pca_dim in pca_dims:
            for c in cs:
                fold_scores = []
                for train_idx, test_idx in splitter.split(train, y, groups):
                    scaler = StandardScaler().fit(feature_matrix(train.iloc[train_idx], cache, layer))
                    x_train = scaler.transform(feature_matrix(train.iloc[train_idx], cache, layer))
                    x_test = scaler.transform(feature_matrix(train.iloc[test_idx], cache, layer))
                    pca = None
                    if pca_dim:
                        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=SEED).fit(x_train)
                        x_train, x_test = pca.transform(x_train), pca.transform(x_test)
                    clf = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(x_train, y[train_idx])
                    fold_scores.append(float(np.mean(clf.predict(x_test) == y[test_idx])))
                score = float(np.mean(fold_scores))
                record = {"layer": layer, "pca_dim": pca_dim, "C": c, "cv_accuracy": score, "fold_accuracy": fold_scores}
                scores.append(record)
                # Fixed tie-break: higher CV accuracy, then lower layer/PCA/C.
                rank = (-score, layer, pca_dim, c)
                if best is None or rank < best[0]:
                    best = (rank, record)
    if best is None:
        raise RuntimeError("no decoder candidate was evaluated")
    selected = best[1]
    layer, pca_dim, c = int(selected["layer"]), int(selected["pca_dim"]), float(selected["C"])
    scaler = StandardScaler().fit(feature_matrix(train, cache, layer))
    x = scaler.transform(feature_matrix(train, cache, layer))
    pca = None
    if pca_dim:
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=SEED).fit(x)
        x = pca.transform(x)
    clf = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(x, y)
    if set(clf.classes_) != set(EVENTS):
        raise AssertionError("frozen decoder lost an event class")
    metadata = {
        "schema_version": 1,
        "protocol": "frozen hidden-event decoder; fit and selection on discovery trajectories only",
        "event_schema": list(EVENTS), "state_schema": list(STATES),
        "probe_position": "last input token before assistant turn",
        "hidden_cache_shape": [37, 4096], "candidate_layers": list(layers), "candidate_pca_dims": list(pca_dims), "candidate_C": list(cs),
        "selected": selected, "fit_trajectories": sorted(discovery), "fit_prefixes": sorted(train["target_prefix"]),
        "n_fit_trajectories": len(discovery), "n_fit_prefixes": len(train),
        "validation_trajectories_not_used_for_fit": sorted(inputs["validation"]),
        "selection": "3-fold GroupKFold by trajectory inside discovery", "random_seed": SEED, "classifier_classes": list(clf.classes_),
    }
    payload = {"schema_version": 1, "event_schema": list(EVENTS), "state_schema": list(STATES), "layer": layer, "pca_dim": pca_dim, "C": c, "scaler": scaler, "pca": pca, "clf": clf, "fit_trajectories": sorted(discovery), "fit_prefixes": sorted(train["target_prefix"]), "cv_scores": scores, "metadata": metadata}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)
    artifact_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_frozen_decoder(path: Path, inputs: Mapping[str, object]) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"frozen decoder not found: {path}; run fit first")
    artifact = joblib.load(path)
    if tuple(artifact.get("event_schema", ())) != EVENTS:
        raise AssertionError("decoder event schema does not match current cache")
    if set(artifact.get("fit_trajectories", ())) != set(inputs["discovery"]):
        raise AssertionError("decoder was not fitted on exactly the frozen discovery trajectories")
    if set(artifact.get("fit_trajectories", ())) & set(inputs["validation"]):
        raise AssertionError("decoder artifact contains validation trajectories")
    expected_fit_prefixes = set(
        inputs["behavior"].loc[
            inputs["behavior"]["trajectory_id"].isin(inputs["discovery"]),
            "target_prefix",
        ]
    )
    if set(artifact.get("fit_prefixes", ())) != expected_fit_prefixes:
        raise AssertionError("decoder fit-prefix provenance does not match frozen discovery rows")
    if not 0 <= int(artifact["layer"]) <= 36:
        raise AssertionError("decoder layer outside hidden cache")
    return artifact


def decode_hidden_event_probabilities(rows: pd.DataFrame, cache: object, artifact: Mapping[str, object]) -> dict[str, np.ndarray]:
    layer = int(artifact["layer"])
    x = artifact["scaler"].transform(feature_matrix(rows, cache, layer))
    if artifact["pca"] is not None:
        x = artifact["pca"].transform(x)
    raw = artifact["clf"].predict_proba(x)
    classes = list(artifact["clf"].classes_)
    result: dict[str, np.ndarray] = {}
    for key, values in zip(rows["target_prefix"], raw):
        probabilities = np.zeros(3, dtype=float)
        for cls, value in zip(classes, values):
            probabilities[EVENTS.index(cls)] = float(value)
        result[str(key)] = normalize_probability(probabilities, name=f"hidden event probability {key}")
    return result


def native_probability_map(table: pd.DataFrame) -> dict[str, np.ndarray]:
    pcols = [f"event_prob_{event}" for event in EVENTS]
    return {str(row["target_prefix"]): normalize_probability(row[pcols].to_numpy(float), name=f"native event probability {row['target_prefix']}") for _, row in table.iterrows()}


def recursive_paths(rows: pd.DataFrame, hidden_probabilities: Mapping[str, np.ndarray], native_probabilities: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Produce one row per prefix for all recursive/oracle baselines."""
    output: list[dict[str, object]] = []
    for trajectory_id, group in rows.sort_values(["trajectory_id", "t"]).groupby("trajectory_id", sort=True):
        group = group.sort_values("t")
        initial = str(group.iloc[0]["initial_state"])
        hidden_hard = native_hard = oracle_hard = initial
        hidden_belief = native_belief = oracle_belief = one_hot(STATES.index(initial))
        hidden_history: list[bool] = []
        explicit_history: list[bool] = []
        for _, row in group.iterrows():
            key = str(row["target_prefix"])
            hidden_probability = normalize_probability(hidden_probabilities[key], name=f"hidden event probability {key}")
            native_probability = normalize_probability(native_probabilities[key], name=f"native event probability {key}")
            hidden_event = EVENTS[int(np.argmax(hidden_probability))]
            native_probability_event = EVENTS[int(np.argmax(native_probability))]
            explicit_event = str(row["event_pred"])
            gt_event = str(row["gt_event"])
            hidden_event_correct = hidden_event == gt_event
            explicit_event_correct = explicit_event == gt_event
            native_probability_event_correct = native_probability_event == gt_event
            hidden_history.append(hidden_event_correct)
            explicit_history.append(explicit_event_correct)

            hidden_hard = update(hidden_hard, hidden_event)
            hidden_belief = transition(hidden_belief, hidden_probability)
            native_hard = update(native_hard, explicit_event)
            native_belief = transition(native_belief, native_probability)
            oracle_hard = update(oracle_hard, gt_event)
            oracle_belief = transition(oracle_belief, one_hot(EVENTS.index(gt_event)))
            hidden_hard_pred = hidden_hard
            hidden_prob_pred = STATES[int(np.argmax(hidden_belief))]
            native_hard_pred = native_hard
            native_prob_pred = STATES[int(np.argmax(native_belief))]
            oracle_hard_pred = oracle_hard
            oracle_prob_pred = STATES[int(np.argmax(oracle_belief))]
            record: dict[str, object] = {
                "target_prefix": key, "trajectory_id": trajectory_id, "t": int(row["t"]), "initial_state": initial,
                "gt_prev_state": row["gt_prev_state"], "gt_state": row["gt_state"], "gt_event": gt_event,
                "native_state_pred": row["state_pred"], "native_state_correct": str(row["state_pred"]) == str(row["gt_state"]),
                "old_symbolic_correct": bool(row["old_symbolic_correct"]),
                "explicit_event_pred": explicit_event, "explicit_event_correct": explicit_event_correct,
                "native_prob_event_pred": native_probability_event, "native_prob_event_correct": native_probability_event_correct,
                "hidden_event_pred": hidden_event, "hidden_event_correct": hidden_event_correct,
                "hidden_event_confidence": float(hidden_probability.max()), "native_event_confidence": float(native_probability.max()),
                "hidden_prob_sum": float(hidden_probability.sum()), "native_prob_sum": float(native_probability.sum()),
                "hidden_hard_state_pred": hidden_hard_pred, "hidden_hard_state_correct": hidden_hard_pred == row["gt_state"],
                "hidden_prob_state_pred": hidden_prob_pred, "hidden_prob_state_correct": hidden_prob_pred == row["gt_state"],
                "native_hard_state_pred": native_hard_pred, "native_hard_state_correct": native_hard_pred == row["gt_state"],
                "native_prob_state_pred": native_prob_pred, "native_prob_state_correct": native_prob_pred == row["gt_state"],
                "oracle_hard_state_pred": oracle_hard_pred, "oracle_hard_state_correct": oracle_hard_pred == row["gt_state"],
                "oracle_prob_state_pred": oracle_prob_pred, "oracle_prob_state_correct": oracle_prob_pred == row["gt_state"],
                "hidden_belief_Left": float(hidden_belief[0]), "hidden_belief_Middle": float(hidden_belief[1]), "hidden_belief_Right": float(hidden_belief[2]),
                "native_belief_Left": float(native_belief[0]), "native_belief_Middle": float(native_belief[1]), "native_belief_Right": float(native_belief[2]),
                "oracle_belief_Left": float(oracle_belief[0]), "oracle_belief_Middle": float(oracle_belief[1]), "oracle_belief_Right": float(oracle_belief[2]),
                "hidden_all_events_correct_so_far": bool(all(hidden_history)), "explicit_all_events_correct_so_far": bool(all(explicit_history)),
            }
            for event, value in zip(EVENTS, hidden_probability):
                record[f"hidden_prob_{event}"] = float(value)
            for event, value in zip(EVENTS, native_probability):
                record[f"native_prob_{event}"] = float(value)
            output.append(record)
    result = pd.DataFrame(output).sort_values(["trajectory_id", "t"]).reset_index(drop=True)
    if len(result) != len(rows) or result["target_prefix"].duplicated().any():
        raise AssertionError("recursive output does not preserve one row per prefix")
    for column in ("hidden_prob_sum", "native_prob_sum"):
        if not np.allclose(result[column].to_numpy(float), 1.0, atol=1e-6):
            raise AssertionError(f"{column} is not normalized")
    return result


def cluster_stat(values: Sequence[float], groups: Sequence[str], n_boot: int = 4000) -> dict[str, object]:
    data = pd.DataFrame({"value": np.asarray(values, dtype=float), "group": np.asarray(groups)})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {"mean": None, "ci95": [None, None], "p_sign_permutation": None, "n_trajectories": 0}
    by_group = data.groupby("group")["value"].mean().to_numpy(float)
    rng = np.random.default_rng(SEED)
    bootstrap = by_group[rng.integers(0, len(by_group), size=(n_boot, len(by_group)))].mean(axis=1)
    null = (by_group[None, :] * rng.choice((-1.0, 1.0), size=(n_boot, len(by_group)))).mean(axis=1)
    mean = float(by_group.mean())
    return {"mean": mean, "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "p_sign_permutation": float(np.mean(np.abs(null) >= abs(mean))), "n_trajectories": int(len(by_group))}


def metric_subsets(frame: pd.DataFrame, column: str) -> dict[str, dict[str, object]]:
    masks: list[tuple[str, pd.Series]] = [("overall", pd.Series(True, index=frame.index)), ("t_ge2", frame["t"] >= 2), ("t_ge3", frame["t"] >= 3)]
    masks.extend((f"t{t}", frame["t"] == t) for t in range(1, 6))
    return {name: {"accuracy": cluster_stat(frame.loc[mask, column].astype(float), frame.loc[mask, "trajectory_id"]), "n_prefixes": int(mask.sum())} for name, mask in masks}


def first_persistent_error(group: pd.DataFrame, correct_column: str) -> int | None:
    q = group.sort_values("t")
    bad = ~q[correct_column].to_numpy(bool)
    steps = q["t"].to_numpy(int)
    for i, step in enumerate(steps):
        if bool(bad[i]) and bool(bad[i:].all()):
            return int(step)
    return None


def trajectory_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for trajectory_id, group in frame.groupby("trajectory_id", sort=True):
        group = group.sort_values("t")
        record: dict[str, object] = {"trajectory_id": trajectory_id}
        for method, column in STATE_METHOD_COLUMNS.items():
            record[f"{method}_final_t5_correct"] = bool(group.iloc[-1][column])
            record[f"{method}_first_persistent_error_t"] = first_persistent_error(group, column)
            record[f"{method}_error_count"] = int((~group[column].astype(bool)).sum())
        record["hidden_event_correct_count"] = int(group["hidden_event_correct"].sum())
        record["explicit_event_correct_count"] = int(group["explicit_event_correct"].sum())
        record["native_prob_event_correct_count"] = int(group["native_prob_event_correct"].sum())
        record["hidden_all_events_correct"] = bool(group.iloc[-1]["hidden_all_events_correct_so_far"])
        records.append(record)
    result = pd.DataFrame(records)
    if len(result) != frame["trajectory_id"].nunique():
        raise AssertionError("trajectory summary coverage mismatch")
    return result


def analyze_predictions(inputs: Mapping[str, object], predictions: pd.DataFrame) -> dict[str, object]:
    discovery, validation = set(inputs["discovery"]), set(inputs["validation"])
    observed = set(predictions["trajectory_id"])
    if observed & discovery:
        raise AssertionError("validation analysis received discovery rows")
    if observed != validation:
        raise AssertionError(f"validation analysis coverage mismatch: {len(observed)} trajectories")
    if len(predictions) != len(validation) * N_STEPS:
        raise AssertionError("validation analysis expects 5 prefixes per trajectory")
    frame = predictions.copy()
    frame["directed_transition"] = frame["gt_prev_state"].astype(str) + "->" + frame["gt_state"].astype(str)
    frame = frame.sort_values(["trajectory_id", "t"]).reset_index(drop=True)
    summary: dict[str, object] = {
        "protocol": "frozen hidden-event probe; discovery-only fit/selection; held-out validation tracking",
        "event_schema": list(EVENTS), "state_schema": list(STATES),
        "event_schema_note": "Existing cache has 3 unordered swap classes; no six-class mapping was introduced.",
        "split": {"discovery_n": len(discovery), "validation_n": len(validation), "discovery_trajectories": sorted(discovery), "validation_trajectories": sorted(validation)},
        "counts": {"validation_trajectories": len(validation), "validation_prefixes": len(predictions), "steps": N_STEPS},
        "methods": {}, "event_accuracy": {}, "event_conditioned_state_accuracy": {}, "error_accumulation": {}, "first_persistent_error": {}, "paired_comparisons": {}, "event_groups": {"by_event": {}, "by_directed_transition": {}, "by_step": {}}, "error_source": {},
    }
    for method, column in STATE_METHOD_COLUMNS.items():
        summary["methods"][method] = metric_subsets(frame, column)
        error = ~frame[column].astype(bool)
        previous = frame.groupby("trajectory_id")[column].shift(1)
        current_from_t2 = frame["t"] >= 2
        summary["error_accumulation"][method] = {
            "error_rate_by_step": {str(t): cluster_stat(error[frame["t"] == t], frame.loc[frame["t"] == t, "trajectory_id"]) for t in range(1, 6)},
            "error_given_previous_error": cluster_stat(error[current_from_t2 & previous.eq(False)], frame.loc[current_from_t2 & previous.eq(False), "trajectory_id"]),
            "error_given_previous_correct": cluster_stat(error[current_from_t2 & previous.eq(True)], frame.loc[current_from_t2 & previous.eq(True), "trajectory_id"]),
        }
        # pandas 2.2+ may expand an all-None scalar groupby.apply result into a
        # DataFrame. Build the one-value-per-trajectory series explicitly so
        # the persistent-error distribution has a stable shape across pandas
        # versions.
        first = pd.Series(
            {
                trajectory_id: first_persistent_error(group, column)
                for trajectory_id, group in frame.groupby("trajectory_id", sort=True)
            },
            name="first_persistent_error_t",
            dtype="float64",
        )
        values = first.dropna().astype(int)
        summary["first_persistent_error"][method] = {"count": int(values.size), "rate": float(values.size / len(validation)), "distribution": {str(t): int((values == t).sum()) for t in range(1, 6)}}
    for method, column in EVENT_METHOD_COLUMNS.items():
        summary["event_accuracy"][method] = metric_subsets(frame, column)
    for method, state_column, event_column in (("hidden_event_hard", "hidden_hard_state_correct", "hidden_event_correct"), ("hidden_event_prob", "hidden_prob_state_correct", "hidden_event_correct"), ("native_explicit_hard", "native_hard_state_correct", "explicit_event_correct"), ("native_explicit_prob", "native_prob_state_correct", "native_prob_event_correct")):
        summary["event_conditioned_state_accuracy"][method] = {}
        for condition, mask in (("event_correct", frame[event_column]), ("event_wrong", ~frame[event_column])):
            summary["event_conditioned_state_accuracy"][method][condition] = metric_subsets(frame.loc[mask], state_column)
    comparisons = (("hidden_event_hard_minus_native_state", "hidden_hard_state_correct", "native_state_correct"), ("hidden_event_prob_minus_native_state", "hidden_prob_state_correct", "native_state_correct"), ("hidden_event_hard_minus_native_explicit_hard", "hidden_hard_state_correct", "native_hard_state_correct"), ("hidden_event_prob_minus_native_explicit_prob", "hidden_prob_state_correct", "native_prob_state_correct"), ("hidden_event_prob_minus_old_symbolic", "hidden_prob_state_correct", "old_symbolic_correct"))
    for name, left, right in comparisons:
        summary["paired_comparisons"][name] = {}
        for subset, mask in (("overall", pd.Series(True, index=frame.index)), ("t_ge2", frame["t"] >= 2), ("t_ge3", frame["t"] >= 3), ("t5", frame["t"] == 5)):
            summary["paired_comparisons"][name][subset] = cluster_stat(frame.loc[mask, left].astype(float).to_numpy() - frame.loc[mask, right].astype(float).to_numpy(), frame.loc[mask, "trajectory_id"])
    for group_name, group_column in (("by_event", "gt_event"), ("by_directed_transition", "directed_transition")):
        for value, group in frame.groupby(group_column, sort=True):
            summary["event_groups"][group_name][str(value)] = {"n_prefixes": int(len(group)), "n_trajectories": int(group.trajectory_id.nunique()), "hidden_probe_accuracy": cluster_stat(group.hidden_event_correct, group.trajectory_id), "explicit_event_accuracy": cluster_stat(group.explicit_event_correct, group.trajectory_id), "hidden_hard_state_accuracy": cluster_stat(group.hidden_hard_state_correct, group.trajectory_id), "hidden_prob_state_accuracy": cluster_stat(group.hidden_prob_state_correct, group.trajectory_id)}
    for step, group in frame.groupby("t", sort=True):
        summary["event_groups"]["by_step"][str(step)] = {"hidden_probe_accuracy": cluster_stat(group.hidden_event_correct, group.trajectory_id), "explicit_event_accuracy": cluster_stat(group.explicit_event_correct, group.trajectory_id)}
    hidden_all = frame["hidden_all_events_correct_so_far"]
    for method, column in (("hidden_event_hard", "hidden_hard_state_correct"), ("hidden_event_prob", "hidden_prob_state_correct")):
        summary["error_source"][method] = {"state_accuracy_when_all_hidden_events_so_far_correct": metric_subsets(frame.loc[hidden_all], column), "state_accuracy_when_current_hidden_event_correct": metric_subsets(frame.loc[frame.hidden_event_correct], column), "state_accuracy_when_current_hidden_event_wrong": metric_subsets(frame.loc[~frame.hidden_event_correct], column), "n_current_event_correct_state_wrong": int((frame.hidden_event_correct & ~frame[column]).sum()), "n_all_events_correct_so_far_state_wrong": int((hidden_all & ~frame[column]).sum())}
    summary["sanity"] = {"oracle_hard_exact": bool(frame["oracle_hard_state_correct"].all()), "oracle_prob_exact": bool(frame["oracle_prob_state_correct"].all()), "hidden_probability_normalized": bool(np.allclose(frame["hidden_prob_sum"], 1.0, atol=1e-6)), "native_probability_normalized": bool(np.allclose(frame["native_prob_sum"], 1.0, atol=1e-6))}
    return {"summary": summary, "frame": frame, "trajectory": trajectory_summary(frame)}


def fmt(stat: Mapping[str, object]) -> str:
    mean = stat.get("mean")
    if mean is None:
        return "NA"
    ci = stat["ci95"]
    return f"{float(mean):.3f} [{float(ci[0]):.3f},{float(ci[1]):.3f}]"


def write_report(summary: Mapping[str, object], path: Path) -> None:
    methods, event_accuracy = summary["methods"], summary["event_accuracy"]
    order = ("overall", "t_ge2", "t_ge3", "t1", "t2", "t3", "t4", "t5")
    lines = ["# Frozen hidden-event probe → recursive state tracking", "", "## Protocol", "", "The hidden-state cache is decoded with a frozen probe fit and selected on discovery trajectories only. Validation trajectories are held out for prediction and reporting. Uncertainty is a trajectory-cluster bootstrap; prefixes are not independent units.", "", f"Event schema: `{', '.join(EVENTS)}`. The existing cache has three unordered swap classes; this experiment does not manufacture six directed event labels.", "", "## Event accuracy", "", "| method | overall | t>=2 | t>=3 | t1 | t2 | t3 | t4 | t5 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in EVENT_METHOD_COLUMNS:
        lines.append("| " + method + " | " + " | ".join(fmt(event_accuracy[method][key]["accuracy"]) for key in order) + " |")
    lines += ["", "## State accuracy", "", "| method | overall | t>=2 | t>=3 | t1 | t2 | t3 | t4 | t5 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in STATE_METHOD_COLUMNS:
        lines.append("| " + method + " | " + " | ".join(fmt(methods[method][key]["accuracy"]) for key in order) + " |")
    lines += ["", "## Event-correct / event-wrong state accuracy", "", "| method | event condition | overall | t>=2 | t>=3 | t5 |", "|---|---|---:|---:|---:|---:|"]
    for method, conditions in summary["event_conditioned_state_accuracy"].items():
        for condition in ("event_correct", "event_wrong"):
            q = conditions[condition]
            lines.append(f"| {method} | {condition} | {fmt(q['overall']['accuracy'])} | {fmt(q['t_ge2']['accuracy'])} | {fmt(q['t_ge3']['accuracy'])} | {fmt(q['t5']['accuracy'])} |")
    lines += ["", "## Paired differences", "", "| comparison | overall | t>=2 | t>=3 | t5 |", "|---|---:|---:|---:|---:|"]
    for name, q in summary["paired_comparisons"].items():
        lines.append(f"| {name} | {fmt(q['overall'])} | {fmt(q['t_ge2'])} | {fmt(q['t_ge3'])} | {fmt(q['t5'])} |")
    lines += ["", "## Error accumulation and first persistent error", "", "For each method, `error_given_previous_*` is evaluated only at t>=2. A first persistent error is the first wrong step followed by wrong predictions at every remaining step through t=5.", "", "| method | error t1 | t2 | t3 | t4 | t5 | given previous error | given previous correct | persistent rate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in STATE_METHOD_COLUMNS:
        error = summary["error_accumulation"][method]
        persistent = summary["first_persistent_error"][method]
        rates = [error["error_rate_by_step"][str(t)] for t in range(1, 6)]
        lines.append("| " + method + " | " + " | ".join(fmt(x) for x in rates) + f" | {fmt(error['error_given_previous_error'])} | {fmt(error['error_given_previous_correct'])} | {persistent['rate']:.3f} |")
    lines += ["", "## Event-type, directed-transition and step checks", "", "See `hidden_event_recursive_event_groups.csv` for event-type and directed-transition breakdowns, including hidden probe accuracy and recursive state accuracy. The step rows directly test probe degradation from t=1 to t=5.", ""]
    for t in range(1, 6):
        q = summary["event_groups"]["by_step"].get(str(t), {})
        if q:
            lines.append(f"- t={t}: hidden probe {fmt(q['hidden_probe_accuracy'])}; explicit event {fmt(q['explicit_event_accuracy'])}")
    lines += ["", "## Direct answers", ""]
    hidden_prob, native_state = methods["hidden_event_prob"], methods["native_state"]
    lines.append(f"1. On held-out validation trajectories, hidden-probe event accuracy is {fmt(event_accuracy['hidden_probe_argmax']['overall']['accuracy'])}; probabilistic hidden-event recursion reaches {fmt(hidden_prob['t5']['accuracy'])} at t=5 and {fmt(hidden_prob['overall']['accuracy'])} over all prefixes.")
    lines.append(f"2. Hidden probabilistic recursion versus native state has paired difference {fmt(summary['paired_comparisons']['hidden_event_prob_minus_native_state']['overall'])}; versus native explicit probabilistic recursion it is {fmt(summary['paired_comparisons']['hidden_event_prob_minus_native_explicit_prob']['overall'])}.")
    lines.append("3. Hidden-event versus explicit-event degradation is visible by step in the event table; the paired state table tests whether the information is useful for tracking rather than merely decodable.")
    lines.append(f"4. Oracle hard/probability exactness is `{summary['sanity']['oracle_hard_exact']}` / `{summary['sanity']['oracle_prob_exact']}`. Correct hidden hard events through a prefix are sufficient in principle; probabilistic recursion can still retain alternative-event mass.")
    lines.append("5. Residual errors while the current hidden event is correct, and errors after all hidden events so far are correct, are recorded under `error_source`; these distinguish event decoding from recursive uncertainty/maintenance.")
    lines += ["", "## Artifacts", "", "- `hidden_event_recursive_prefix.csv`", "- `hidden_event_recursive_trajectory.csv`", "- `hidden_event_recursive_metrics.csv`", "- `hidden_event_recursive_event_groups.csv`", "- `hidden_event_recursive_summary.json`", "- `frozen_event_decoder.joblib` and `frozen_event_decoder.json`"]
    path.write_text("\n".join(lines) + "\n")


def write_formal_outputs(analysis: Mapping[str, object], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame, trajectory, summary = analysis["frame"], analysis["trajectory"], analysis["summary"]
    frame.to_csv(out / "hidden_event_recursive_prefix.csv", index=False)
    trajectory.to_csv(out / "hidden_event_recursive_trajectory.csv", index=False)
    metric_rows = []
    for group_name, group in (("state", summary["methods"]), ("event", summary["event_accuracy"])):
        for method, subsets in group.items():
            for subset, value in subsets.items():
                metric = value["accuracy"]
                metric_rows.append({"group": group_name, "method": method, "subset": subset, "metric": "accuracy", "mean": metric["mean"], "ci_low": metric["ci95"][0], "ci_high": metric["ci95"][1], "p_sign_permutation": metric["p_sign_permutation"], "n_trajectories": metric["n_trajectories"], "n_prefixes": value["n_prefixes"]})
    pd.DataFrame(metric_rows).to_csv(out / "hidden_event_recursive_metrics.csv", index=False)
    group_rows = []
    for grouping, values in summary["event_groups"].items():
        if grouping == "by_step":
            continue
        for value, stats in values.items():
            for metric_name in ("hidden_probe_accuracy", "explicit_event_accuracy", "hidden_hard_state_accuracy", "hidden_prob_state_accuracy"):
                metric = stats[metric_name]
                group_rows.append({"grouping": grouping, "group": value, "metric": metric_name, "mean": metric["mean"], "ci_low": metric["ci95"][0], "ci_high": metric["ci95"][1], "n_trajectories": metric["n_trajectories"], "n_prefixes": stats["n_prefixes"]})
    for step, stats in summary["event_groups"]["by_step"].items():
        for metric_name in ("hidden_probe_accuracy", "explicit_event_accuracy"):
            metric = stats[metric_name]
            group_rows.append({"grouping": "by_step", "group": step, "metric": metric_name, "mean": metric["mean"], "ci_low": metric["ci95"][0], "ci_high": metric["ci95"][1], "n_trajectories": metric["n_trajectories"], "n_prefixes": None})
    pd.DataFrame(group_rows).to_csv(out / "hidden_event_recursive_event_groups.csv", index=False)
    (out / "hidden_event_recursive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, out / "hidden_event_recursive_report.md")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prediction_shard(frame: pd.DataFrame, out: Path, shard_index: int, num_shards: int) -> Path:
    path = out / "shards" / f"prediction_shard_{shard_index}.csv"
    marker_path = path.with_suffix(".complete.json")
    if path.exists() or marker_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard or marker: {path}")
    atomic_csv(frame, path)
    marker = {"schema_version": 1, "shard_index": shard_index, "num_shards": num_shards, "rows": len(frame), "trajectory_ids": sorted(frame["trajectory_id"].unique()), "target_prefixes": sorted(frame["target_prefix"]), "sha256": file_sha256(path)}
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return path


def merge_prediction_shards(out: Path, num_shards: int, validation_ids: set[str], expected_prefixes: set[str]) -> pd.DataFrame:
    frames = []
    for index in range(num_shards):
        path = out / "shards" / f"prediction_shard_{index}.csv"
        marker_path = path.with_suffix(".complete.json")
        if not path.exists() or not marker_path.exists():
            raise FileNotFoundError(f"incomplete prediction shard {index}")
        marker = json.loads(marker_path.read_text())
        if int(marker.get("shard_index", -1)) != index or int(marker.get("num_shards", -1)) != num_shards:
            raise AssertionError(f"shard marker mismatch for {path}")
        if file_sha256(path) != marker.get("sha256"):
            raise AssertionError(f"shard checksum mismatch for {path}")
        frame = pd.read_csv(path)
        if len(frame) != int(marker.get("rows", -1)):
            raise AssertionError(f"shard marker row count mismatch for {path}")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged["target_prefix"].duplicated().any():
        raise AssertionError("duplicate target prefix across prediction shards")
    if set(merged["trajectory_id"]) != validation_ids or set(merged["target_prefix"]) != expected_prefixes:
        raise AssertionError("merged shards do not provide exact validation coverage")
    if len(merged) != len(expected_prefixes):
        raise AssertionError("merged shard row count mismatch")
    return merged.sort_values(["trajectory_id", "t"]).reset_index(drop=True)


def run_unit_tests() -> None:
    for state in STATES:
        for event in EVENTS:
            deterministic = transition(one_hot(STATES.index(state)), one_hot(EVENTS.index(event)))
            assert np.array_equal(deterministic, one_hot(STATES.index(update(state, event))))
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        result = transition(normalize_probability(rng.random(3)), normalize_probability(rng.random(3)))
        assert np.isfinite(result).all() and (result >= 0).all() and np.isclose(result.sum(), 1.0)
    synthetic = pd.DataFrame([
        {"target_prefix": "toy_t1", "trajectory_id": "toy", "t": 1, "initial_state": "Left", "gt_prev_state": "Left", "gt_state": "Middle", "gt_event": "Left and Middle", "state_pred": "Left", "old_symbolic_correct": False, "event_pred": "Left and Middle"},
        {"target_prefix": "toy_t2", "trajectory_id": "toy", "t": 2, "initial_state": "Left", "gt_prev_state": "Middle", "gt_state": "Right", "gt_event": "Middle and Right", "state_pred": "Left", "old_symbolic_correct": False, "event_pred": "Middle and Right"},
    ])
    probabilities = {"toy_t1": one_hot(0), "toy_t2": one_hot(1)}
    result = recursive_paths(synthetic, probabilities, probabilities)
    assert result["oracle_hard_state_correct"].all() and result["oracle_prob_state_correct"].all()
    assert result["hidden_hard_state_correct"].all() and result["hidden_prob_state_correct"].all()
    assert first_persistent_error(result.assign(test_correct=[True, False]), "test_correct") == 2
    with tempfile.TemporaryDirectory(prefix="hidden-event-merge-") as temp:
        root = Path(temp)
        write_prediction_shard(pd.DataFrame([{"target_prefix": "a_t1", "trajectory_id": "a", "t": 1}]), root, 0, 2)
        write_prediction_shard(pd.DataFrame([{"target_prefix": "b_t1", "trajectory_id": "b", "t": 1}]), root, 1, 2)
        assert len(merge_prediction_shards(root, 2, {"a", "b"}, {"a_t1", "b_t1"})) == 2
    print("UNIT_PASS: update, one-hot equivalence, probability normalization, oracle recursion, persistent-error definition, shard marker/merge")


def default_artifact(args: argparse.Namespace) -> Path:
    return args.artifact or args.out / "frozen_event_decoder.joblib"


def command_schema_check(args: argparse.Namespace) -> None:
    inputs = load_inputs(args)
    print(json.dumps({"SCHEMA_CHECK_PASS": True, "behavior_rows": len(inputs["behavior"]), "hidden_cache_keys": len(inputs["hidden_cache"].files), "hidden_cache_shape": [37, 4096], "discovery_trajectories": len(inputs["discovery"]), "validation_trajectories": len(inputs["validation"]), "event_schema": list(EVENTS), "native_probability_shards": len(list(args.native_event_prob_dir.glob("event_probs_shard_*.csv"))), "validation_leakage": False}, indent=2))


def command_fit(args: argparse.Namespace) -> None:
    inputs = load_inputs(args)
    metadata = fit_frozen_decoder(inputs, default_artifact(args), parse_int_list(args.candidate_layers), parse_int_list(args.candidate_pca_dims), parse_float_list(args.candidate_c))
    print(json.dumps({"FROZEN_DECODER_PASS": True, "artifact": str(default_artifact(args)), "selected": metadata["selected"], "fit_trajectories": metadata["n_fit_trajectories"], "validation_used_for_fit": False}, indent=2))


def validation_rows(inputs: Mapping[str, object], ids: Sequence[str]) -> pd.DataFrame:
    validation = set(inputs["validation"])
    requested = set(ids)
    if not requested <= validation:
        raise AssertionError("prediction IDs must be validation trajectories; discovery prediction is refused")
    return inputs["behavior"][inputs["behavior"]["trajectory_id"].isin(requested)].copy().sort_values(["trajectory_id", "t"])


def predict_for_ids(args: argparse.Namespace, ids: Sequence[str], output: Path | None = None) -> pd.DataFrame:
    inputs = load_inputs(args)
    artifact = load_frozen_decoder(default_artifact(args), inputs)
    rows = validation_rows(inputs, ids)
    hidden = decode_hidden_event_probabilities(rows, inputs["hidden_cache"], artifact)
    native = native_probability_map(inputs["native_probs"])
    merged = inputs["behavior"].merge(inputs["symbolic"], left_on="target_prefix", right_on="key", validate="one_to_one")
    merged = merged[merged["trajectory_id"].isin(set(ids))].copy()
    result = recursive_paths(merged, hidden, native)
    if output is not None:
        atomic_csv(result, output)
    return result


def command_smoke(args: argparse.Namespace) -> None:
    ids = tuple(x.strip() for x in args.trajectory_ids.split(",") if x.strip()) if args.trajectory_ids else ("cup_001", "cup_002")
    result = predict_for_ids(args, ids, args.out / "smoke_prefixes.csv")
    trajectory = trajectory_summary(result)
    trajectory.to_csv(args.out / "smoke_trajectories.csv", index=False)
    payload = {"SMOKE_PASS": True, "trajectories": sorted(result.trajectory_id.unique()), "prefixes": len(result), "hidden_probability_normalized": bool(np.allclose(result.hidden_prob_sum, 1.0, atol=1e-6)), "native_probability_normalized": bool(np.allclose(result.native_prob_sum, 1.0, atol=1e-6)), "oracle_hard_exact": bool(result.oracle_hard_state_correct.all()), "oracle_prob_exact": bool(result.oracle_prob_state_correct.all()), "validation_tuning": False, "artifact": str(default_artifact(args))}
    (args.out / "smoke_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    inputs = load_inputs(args)
    validation = sorted(inputs["validation"])
    if args.shard_index is not None:
        if args.num_shards is None or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index and num-shards must define a valid shard")
        ids = [trajectory for i, trajectory in enumerate(validation) if i % args.num_shards == args.shard_index]
        result = predict_for_ids(args, ids)
        path = write_prediction_shard(result, args.out, args.shard_index, args.num_shards)
        print(json.dumps({"PREDICTION_SHARD_PASS": True, "path": str(path), "rows": len(result), "trajectories": ids}, indent=2))
    else:
        result = predict_for_ids(args, validation, args.out / "validation_prefixes.csv")
        print(json.dumps({"PREDICTION_PASS": True, "path": str(args.out / 'validation_prefixes.csv'), "rows": len(result), "trajectories": len(validation)}, indent=2))


def command_merge(args: argparse.Namespace) -> None:
    inputs = load_inputs(args)
    expected = set(inputs["behavior"].loc[inputs["behavior"].trajectory_id.isin(inputs["validation"]), "target_prefix"])
    merged = merge_prediction_shards(args.out, args.num_shards, set(inputs["validation"]), expected)
    atomic_csv(merged, args.out / "validation_prefixes.csv")
    marker = {"status": "complete", "rows": len(merged), "num_shards": args.num_shards, "sha256": file_sha256(args.out / "validation_prefixes.csv")}
    (args.out / "validation_prefixes.complete.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(json.dumps({"MERGE_PASS": True, "rows": len(merged), "trajectories": len(inputs["validation"])}, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    inputs = load_inputs(args)
    decoder = load_frozen_decoder(default_artifact(args), inputs)
    predictions = pd.read_csv(args.predictions or args.out / "validation_prefixes.csv")
    analysis = analyze_predictions(inputs, predictions)
    analysis["summary"]["frozen_decoder"] = decoder["metadata"]
    analysis["summary"]["source_artifacts"] = {
        "behavior": {"path": str(args.behavior), "sha256": file_sha256(args.behavior)},
        "split": {"path": str(args.split), "sha256": file_sha256(args.split)},
        "hidden_cache": {"path": str(args.hidden_cache), "sha256": file_sha256(args.hidden_cache)},
        "native_event_probability_dir": str(args.native_event_prob_dir),
        "old_symbolic": {"path": str(args.symbolic), "sha256": file_sha256(args.symbolic)},
        "frozen_decoder": {"path": str(default_artifact(args)), "sha256": file_sha256(default_artifact(args))},
    }
    write_formal_outputs(analysis, args.out)
    print(json.dumps({"ANALYSIS_PASS": True, "out": str(args.out), "rows": len(predictions)}, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    unit = sub.add_parser("unit", help="run dependency-free algebra and shard tests")
    unit.set_defaults(func=lambda args: run_unit_tests())
    for name in ("schema-check", "fit-smoke", "fit", "smoke", "predict", "merge", "analyze"):
        p = sub.add_parser(name)
        add_common_args(p)
        p.set_defaults(func=None)
    for name in ("smoke",):
        sub.choices[name].add_argument("--trajectory-ids", default=None, help="comma-separated validation IDs; default cup_001,cup_002")
    sub.choices["predict"].add_argument("--shard-index", type=int, default=None)
    sub.choices["predict"].add_argument("--num-shards", type=int, default=None)
    sub.choices["merge"].add_argument("--num-shards", type=int, required=True)
    sub.choices["analyze"].add_argument("--predictions", type=Path, default=None)
    for name in ("fit-smoke", "fit"):
        sub.choices[name].add_argument("--candidate-layers", default="24,28")
        sub.choices[name].add_argument("--candidate-pca-dims", default="0,40")
        sub.choices[name].add_argument("--candidate-c", default="0.1,1.0,10.0")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "unit":
        run_unit_tests()
    elif args.command == "schema-check":
        command_schema_check(args)
    elif args.command in ("fit-smoke", "fit"):
        command_fit(args)
    elif args.command == "smoke":
        command_smoke(args)
    elif args.command == "predict":
        command_predict(args)
    elif args.command == "merge":
        command_merge(args)
    elif args.command == "analyze":
        command_analyze(args)


if __name__ == "__main__":
    main()
