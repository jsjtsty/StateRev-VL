#!/usr/bin/env python3
"""Frozen VLM + hidden-event decoder + learned state updater.

CPU-only offline pipeline.  The only trainable objects are the 27 logits of
three 3x3 event transition matrices and, optionally, one event temperature.
All fitting/selection is discovery-only; validation is frozen evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from content_disjoint_split import assert_content_disjoint  # noqa: E402

STATES = ("Left", "Middle", "Right")
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
SI = {x: i for i, x in enumerate(STATES)}
EI = {x: i for i, x in enumerate(EVENTS)}
SPLIT = ROOT / "outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"
FP = ROOT / "outputs/vetbench/validity_gate_v1/input_fingerprints.csv"
DEFAULT_OUT = ROOT / "outputs/vetbench/learned_state_updater_v1"

MODEL_CONFIG = {
    "qwen": {
        "behavior": ROOT / "outputs/vetbench/mechanism_gate_final/behavior.csv",
        "hidden": ROOT / "outputs/vetbench/hidden_state_probe/hidden_states.npz",
        "decoder": ROOT / "outputs/vetbench/hidden_event_recursive_content_disjoint_v1/frozen_event_decoder.joblib",
        "native_probs": ROOT / "outputs/vetbench/recursive_state_recovery_v1/shards",
        "native_state_col": "state_pred",
        "event_prob_source": "frozen_hidden_decoder",
    },
    "llava": {
        "behavior": ROOT / "outputs/vetbench/llava_next_video_7b_replication_v1/behavior.csv",
        "hidden": ROOT / "outputs/vetbench/llava_next_video_7b_replication_v1/hidden_states.npz",
        "decoder": ROOT / "outputs/vetbench/llava_next_video_7b_replication_v1/frozen_hidden_event_decoder.joblib",
        "native_probs": None,
        "native_state_col": "state_pred",
        "event_prob_source": "frozen_hidden_decoder",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target_prefixes(frame: pd.DataFrame) -> pd.Series:
    return frame.trajectory_id.astype(str) + "_t" + frame.t.astype(int).astype(str)


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 0.0)
    return p / p.sum(axis=-1, keepdims=True)


def transition_numpy(belief: np.ndarray, event_p: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=float)
    for e in range(3):
        out += float(event_p[e]) * (belief @ matrices[e])
    return normalize(out)


def handwritten_update(state: str, event: str) -> str:
    a, b = event.split(" and ")
    return b if state == a else a if state == b else state


def handwritten_matrix(e: str) -> np.ndarray:
    m = np.zeros((3, 3), dtype=float)
    for s in STATES:
        m[SI[s], SI[handwritten_update(s, e)]] = 1.0
    return m


def handwritten_matrices() -> np.ndarray:
    return np.stack([handwritten_matrix(e) for e in EVENTS])


def load_cache(name: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict, dict]:
    if name not in MODEL_CONFIG:
        raise ValueError(f"unknown model {name}")
    cfg = MODEL_CONFIG[name]
    behavior = pd.read_csv(cfg["behavior"])
    if name == "qwen":
        behavior = behavior[behavior.condition == "baseline"].copy()
        behavior = behavior.rename(columns={"key": "target_prefix"})
        # mechanism_gate_final stores the native state prediction in baseline.
        behavior["trajectory_id"] = behavior.target_prefix.str.rsplit("_t", n=1).str[0]
        behavior["t"] = behavior.target_prefix.str.rsplit("_t", n=1).str[1].astype(int)
        labels = pd.read_csv(ROOT / "outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv")
    else:
        labels = behavior.copy()
        behavior["target_prefix"] = behavior.trajectory_id.astype(str) + "_t" + behavior.t.astype(int).astype(str)
    if name == "qwen":
        # Use the canonical aligned behavior manifest for GT and initial state.
        labels["target_prefix"] = target_prefixes(labels)
        keep = ["target_prefix", "trajectory_id", "t", "initial_state", "gt_prev_state", "gt_state", "gt_event", "event_pred", "state_pred"]
        labels = labels[keep]
        behavior = behavior.drop(columns=[c for c in ["trajectory_id", "t"] if c in behavior.columns], errors="ignore").merge(labels, on="target_prefix", suffixes=("_native", ""))
        behavior["native_state_pred"] = behavior["state_pred_native"]
    else:
        behavior["native_state_pred"] = behavior["state_pred"]
    required = {"target_prefix", "trajectory_id", "t", "initial_state", "gt_state", "gt_event", "native_state_pred"}
    missing = required - set(behavior.columns)
    if missing:
        raise AssertionError(f"{name} behavior schema missing {sorted(missing)}")
    behavior = behavior.sort_values(["trajectory_id", "t"]).reset_index(drop=True)
    assert len(behavior) == 250 and behavior.target_prefix.nunique() == 250
    with np.load(cfg["hidden"], allow_pickle=False) as z:
        hidden = {k: z[k] for k in z.files}
    assert set(hidden) == set(behavior.target_prefix)
    decoder = joblib.load(cfg["decoder"])
    layer = int(decoder["layer"])
    # Recompute frozen decoder probabilities from hidden cache. No fitting occurs.
    X = np.stack([hidden[k][layer] for k in behavior.target_prefix])
    raw = decoder["clf"].predict_proba(decoder["scaler"].transform(X))
    probs = np.zeros((len(behavior), 3), dtype=float)
    for j, cls in enumerate(decoder["clf"].classes_):
        probs[:, EI[str(cls)]] = raw[:, j]
    probs = normalize(probs)
    behavior[[f"hidden_event_prob_{e}" for e in EVENTS]] = probs
    behavior["hidden_event_pred"] = [EVENTS[int(x)] for x in probs.argmax(1)]
    # Native event probabilities are optional. LLaVA stores them in JSON.
    if name == "llava":
        rows = []
        for raw_json in behavior.event_scores_json:
            d = json.loads(raw_json)
            rows.append([float(d[f"{e}__prob"]) for e in EVENTS])
        behavior[[f"native_event_prob_{e}" for e in EVENTS]] = normalize(np.asarray(rows))
    else:
        pdir = cfg["native_probs"]
        tables = []
        for p in sorted(pdir.glob("event_probs_shard_*.csv")):
            tables.append(pd.read_csv(p))
        native = pd.concat(tables, ignore_index=True)
        cols = [f"event_prob_{e}" for e in EVENTS]
        native = native[["target_prefix"] + cols].rename(columns={c: f"native_{c}" for c in cols})
        behavior = behavior.merge(native, on="target_prefix", validate="one_to_one")
    split = json.loads(SPLIT.read_text())
    # Re-check the exact split at cache load time, including hidden tensors.
    # This fails closed before any fitting or evaluation can start.
    assert_content_disjoint(split, FP, cfg["hidden"])
    return behavior, hidden, decoder, split


def event_probs(frame: pd.DataFrame, source: str = "hidden") -> np.ndarray:
    prefix = "hidden_event_prob_" if source == "hidden" else "native_event_prob_"
    return normalize(frame[[prefix + e for e in EVENTS]].to_numpy(float))


def fit_transition(rows: pd.DataFrame, event_source: str = "hidden", steps: int = 1200, lr: float = 0.08,
                   seed: int = 20260918) -> dict:
    """Fit 27 unconstrained logits; each event matrix is row-softmax."""
    torch.manual_seed(seed)
    discovery = rows.sort_values(["trajectory_id", "t"])
    logits = torch.nn.Parameter(torch.zeros((3, 3, 3), dtype=torch.float64))
    optimizer = torch.optim.Adam([logits], lr=lr)
    y_by_traj = []
    for _, group in discovery.groupby("trajectory_id", sort=True):
        init = SI[str(group.iloc[0].initial_state)]
        qs = torch.tensor(event_probs(group, event_source), dtype=torch.float64)
        target = torch.tensor([SI[x] for x in group.gt_state], dtype=torch.long)
        y_by_traj.append((init, qs, target))
    for _ in range(steps):
        optimizer.zero_grad()
        matrices = torch.softmax(logits, dim=-1)
        loss = torch.zeros((), dtype=torch.float64)
        for init, qs, target in y_by_traj:
            belief = torch.nn.functional.one_hot(torch.tensor(init), 3).double()
            for t in range(len(target)):
                # Teacher-forced transition likelihood identifies T_e rows
                # directly from (S_{t-1}, S_t), while the recursive term below
                # trains the actual deployed belief dynamics.
                prev = torch.nn.functional.one_hot(
                    target[t - 1] if t else torch.tensor(init), 3
                ).double()
                teacher_next = torch.einsum("e,eij,i->j", qs[t], matrices, prev)
                loss = loss - 0.5 * torch.log(teacher_next[target[t]].clamp_min(1e-12))
                belief = torch.einsum("e,s,eij->j", qs[t], belief, matrices)
                belief = belief / belief.sum()
                loss = loss - 0.5 * torch.log(belief[target[t]].clamp_min(1e-12))
        loss = loss / len(y_by_traj)
        loss.backward(); optimizer.step()
    matrices = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
    return {"schema_version": 1, "state_schema": list(STATES), "event_schema": list(EVENTS),
            "parameter_count": 27, "event_source": event_source, "seed": seed,
            "train_trajectories": sorted(rows.trajectory_id.unique()), "steps": steps,
            "learning_rate": lr, "transition_logits": logits.detach().cpu().numpy(),
            "transition_matrices": matrices}


def fit_temperature(rows: pd.DataFrame, event_source: str = "hidden", seed: int = 20260918) -> dict:
    """Discovery-only scalar temperature on frozen event probabilities."""
    q = np.clip(event_probs(rows.sort_values(["trajectory_id", "t"]), event_source), 1e-8, 1.0)
    y = rows.sort_values(["trajectory_id", "t"]).gt_event.map(EI).to_numpy()
    logq = np.log(q)
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    scores = []
    for T in candidates:
        z = logq / T; z -= z.max(axis=1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        scores.append(float(-np.log(p[np.arange(len(y)), y]).mean()))
    best = int(np.argmin(scores)); T = float(candidates[best])
    return {"schema_version": 1, "temperature": T, "seed": seed, "event_source": event_source,
            "train_trajectories": sorted(rows.trajectory_id.unique()), "selection": "discovery NLL grid",
            "raw_nll": float(scores[candidates.tolist().index(1.0)]), "calibrated_nll": float(scores[best])}


def apply_temperature(q: np.ndarray, temperature: float) -> np.ndarray:
    z = np.log(np.clip(q, 1e-8, 1.0)) / float(temperature); z -= z.max(axis=1, keepdims=True)
    return normalize(np.exp(z))


def recursive(rows: pd.DataFrame, matrices: np.ndarray, source: str = "hidden", temperature: float = 1.0) -> pd.DataFrame:
    output = []
    for _, group in rows.sort_values(["trajectory_id", "t"]).groupby("trajectory_id", sort=True):
        state = SI[str(group.iloc[0].initial_state)]
        belief = np.eye(3)[state]
        qs = apply_temperature(event_probs(group, source), temperature)
        event_pred = np.asarray(qs).argmax(1)
        for i, (_, row) in enumerate(group.iterrows()):
            belief = transition_numpy(belief, qs[i], matrices)
            pred = STATES[int(belief.argmax())]
            output.append({"target_prefix": row.target_prefix, "trajectory_id": row.trajectory_id, "t": int(row.t),
                           "gt_state": row.gt_state, "gt_event": row.gt_event, "pred_state": pred,
                           "correct": pred == row.gt_state, "all_events_correct_so_far": bool(
                               (event_pred[: i + 1] == np.asarray([EI[x] for x in group.gt_event.iloc[: i + 1]])).all()),
                           "event_pred": EVENTS[int(event_pred[i])],
                           "belief_Left": belief[0], "belief_Middle": belief[1], "belief_Right": belief[2]})
    return pd.DataFrame(output)


def cluster(values: Iterable[float], trajectories: Iterable[str], seed: int = 7, n_boot: int = 4000) -> dict:
    d = pd.DataFrame({"v": np.asarray(list(values), float), "g": list(trajectories)})
    z = d.groupby("g").v.mean().to_numpy(float); rng = np.random.default_rng(seed)
    boot = z[rng.integers(0, len(z), size=(n_boot, len(z)))].mean(1)
    return {"mean": float(z.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "n_trajectories": len(z)}


def metrics(frame: pd.DataFrame, col: str = "correct") -> dict:
    result = {}
    for name, mask in [("overall", np.ones(len(frame), bool)), ("t_ge2", frame.t >= 2), ("t_ge3", frame.t >= 3)] + [(f"t{t}", frame.t == t) for t in range(1, 6)]:
        x = frame.loc[mask]; result[name] = cluster(x[col], x.trajectory_id, 100 + len(result))
    return result


def trajectory_error_metrics(frame: pd.DataFrame) -> dict:
    """Trajectory-level persistence and wrong-trajectory counts.

    A persistent error is the first wrong prediction followed by wrong
    predictions at every remaining step through t=5, matching the recursive
    tracker convention.  Counts are reported separately from prefix accuracy.
    """
    first_persistent = 0
    any_wrong = 0
    final_wrong = 0
    for _, group in frame.sort_values(["trajectory_id", "t"]).groupby("trajectory_id", sort=True):
        correct = group.correct.to_numpy(bool)
        any_wrong += int((~correct).any())
        final_wrong += int(not correct[-1])
        if any((not bool(correct[i])) and bool((~correct[i:]).all()) for i in range(len(correct))):
            first_persistent += 1
    n = int(frame.trajectory_id.nunique())
    return {"persistent_error_rate": float(first_persistent / n),
            "persistent_error_trajectory_count": first_persistent,
            "wrong_trajectory_count_any_step": any_wrong,
            "wrong_trajectory_count_final_t5": final_wrong,
            "n_trajectories": n}


def save_artifact(path: Path, artifact: dict):
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(artifact, path)
    meta = {k: v for k, v in artifact.items() if k not in ("transition_logits",)}
    meta["transition_matrices"] = np.asarray(artifact["transition_matrices"]).round(8).tolist()
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def write_manifest(path: Path, model: str, artifact: dict):
    cfg = MODEL_CONFIG[model]
    manifest = {
        "schema_version": 1,
        "model": model,
        "split": str(SPLIT),
        "split_sha256": sha256(SPLIT),
        "behavior_cache": str(cfg["behavior"]),
        "behavior_cache_sha256": sha256(cfg["behavior"]),
        "hidden_cache": str(cfg["hidden"]),
        "hidden_cache_sha256": sha256(cfg["hidden"]),
        "decoder": str(cfg["decoder"]),
        "decoder_sha256": sha256(cfg["decoder"]),
        "train_trajectories": artifact.get("train_trajectories", []),
        "parameter_count": int(artifact.get("parameter_count", 0)),
        "event_source": artifact.get("event_source", "hidden"),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def fit_predict_correct(rows: pd.DataFrame, seed: int = 20260918) -> dict:
    """Return a deliberately small correction-module interface.

    This is an opt-in placeholder for later discovery-only experiments.  It
    records the feature contract and parameter budget, but is not used by the
    default recursion/evaluation path.
    """
    return {"schema_version": 1, "enabled": False, "seed": seed,
            "feature_schema": ["belief_Left", "belief_Middle", "belief_Right",
                               "state_evidence_Left", "state_evidence_Middle",
                               "state_evidence_Right", "event_confidence"],
            "parameter_count": 0, "train_trajectories": sorted(rows.trajectory_id.unique()),
            "note": "interface only; no correction module trained in this stage"}


def calibration_smoke(args):
    rows, _, _, split = load_cache(args.model)
    ids = split["discovery_trajectories"][: args.trajectories]
    train = rows[rows.trajectory_id.isin(ids)]
    temp = fit_temperature(train, "hidden", args.seed)
    raw = event_probs(train, "hidden")
    calibrated = apply_temperature(raw, temp["temperature"])
    assert np.allclose(calibrated.sum(1), 1.0)
    print(json.dumps({"status": "CALIBRATION_SMOKE_PASS", "model": args.model,
                      "trajectories": ids, "temperature": temp["temperature"],
                      "raw_nll": temp["raw_nll"], "calibrated_nll": temp["calibrated_nll"],
                      "raw_mean_max_prob": float(raw.max(1).mean()),
                      "calibrated_mean_max_prob": float(calibrated.max(1).mean())}, indent=2))


def cross_smoke(args):
    out = args.out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for model in MODEL_CONFIG:
        rows, _, _, split = load_cache(model)
        ids = split["discovery_trajectories"][: args.trajectories]
        artifact = fit_transition(rows[rows.trajectory_id.isin(ids)], steps=300, seed=args.seed)
        artifact["smoke_only"] = True
        path = out / f"{model}_updater.joblib"
        save_artifact(path, artifact)
        write_manifest(out / f"{model}_manifest.json", model, artifact)
        artifacts[model] = path
    # Load each frozen artifact against both cache adapters.  This verifies the
    # updater has no model-specific input/schema dependency without running val.
    records = []
    for source_model, path in artifacts.items():
        artifact = joblib.load(path)
        for target_model in MODEL_CONFIG:
            rows, _, _, split = load_cache(target_model)
            ids = split["discovery_trajectories"][: args.trajectories]
            sample = rows[rows.trajectory_id.isin(ids)]
            rec = recursive(sample, np.asarray(artifact["transition_matrices"]))
            records.append({"source_model": source_model, "target_model": target_model,
                            "trajectories": ids, "accuracy": float(rec.correct.mean()),
                            "parameter_count": int(artifact["parameter_count"])})
    (out / "cross_model_smoke.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"status": "CROSS_MODEL_SMOKE_PASS", "records": records,
                      "output": str(out)}, indent=2))


def curve(args):
    """Formal few-shot command; intentionally not invoked by this change."""
    rng = np.random.default_rng(args.seed)
    rows, _, _, split = load_cache(args.model)
    discovery = np.array(split["discovery_trajectories"])
    validation = rows[rows.trajectory_id.isin(split["validation_trajectories"])]
    records = []
    for n in args.trajectory_counts:
        for repeat in range(args.repeats):
            ids = rng.choice(discovery, size=min(n, len(discovery)), replace=False).tolist()
            train = rows[rows.trajectory_id.isin(ids)]
            artifact = fit_transition(train, seed=args.seed + repeat + n)
            rec = recursive(validation, artifact["transition_matrices"])
            for metric, value in metrics(rec).items():
                records.append({"model": args.model, "trajectories": n, "repeat": repeat,
                                "metric": metric, "mean": value["mean"],
                                "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1]})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out / f"{args.model}_fewshot_curve.csv", index=False)
    print(json.dumps({"status": "CURVE_COMPLETE", "output": str(args.out)}, indent=2))


def smoke(args):
    rows, _, _, split = load_cache(args.model)
    ids = split["discovery_trajectories"][: args.trajectories]
    train = rows[rows.trajectory_id.isin(ids)]
    artifact = fit_transition(train, steps=300, seed=17)
    rec = recursive(train, artifact["transition_matrices"])
    print(json.dumps({"status": "SMOKE_PASS", "model": args.model, "trajectories": ids,
                      "parameter_count": artifact["parameter_count"],
                      "train_accuracy": float(rec.correct.mean()),
                      "transition_matrices": np.asarray(artifact["transition_matrices"]).round(4).tolist()}, indent=2))


def fit(args):
    rows, _, _, split = load_cache(args.model)
    train = rows[rows.trajectory_id.isin(split["discovery_trajectories"])]
    artifact = fit_transition(train, event_source=args.event_source, steps=args.steps, lr=args.lr, seed=args.seed)
    out = args.out / f"{args.model}_updater.joblib"; save_artifact(out, artifact)
    write_manifest(args.out / f"{args.model}_manifest.json", args.model, artifact)
    temp = fit_temperature(train, args.event_source, args.seed)
    (args.out / f"{args.model}_temperature.json").write_text(json.dumps(temp, indent=2) + "\n")
    print(json.dumps({"status": "FIT_COMPLETE", "model": args.model, "artifact": str(out), "temperature": temp}, indent=2))


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    source_rows, _, _, split = load_cache(args.source_model)
    target_rows, _, _, target_split = load_cache(args.target_model)
    assert split["discovery_trajectories"] == target_split["discovery_trajectories"]
    artifact = joblib.load(args.artifact)
    val = target_rows[target_rows.trajectory_id.isin(split["validation_trajectories"])]
    learned = recursive(val, np.asarray(artifact["transition_matrices"]), args.event_source, args.temperature)
    handwritten = recursive(val, handwritten_matrices(), args.event_source, args.temperature)
    learned["method"] = "learned_updater"
    handwritten["method"] = "handwritten_rule"
    rec = pd.concat([learned, handwritten], ignore_index=True)
    rec.to_csv(args.out / f"{args.source_model}_to_{args.target_model}_validation.csv", index=False)
    method_metrics = {method: metrics(group) for method, group in rec.groupby("method", sort=True)}
    method_error_metrics = {method: trajectory_error_metrics(group) for method, group in rec.groupby("method", sort=True)}
    native_state = val["native_state_pred"].eq(val["gt_state"])
    native_frame = val[["target_prefix", "trajectory_id", "t"]].copy()
    native_frame["correct"] = native_state.to_numpy()
    summary = {"source_model": args.source_model, "target_model": args.target_model,
               "event_source": args.event_source, "temperature": args.temperature,
               "metrics": method_metrics,
               "trajectory_error_metrics": method_error_metrics,
               "native_state_metrics": metrics(native_frame),
               "parameter_count": int(artifact["parameter_count"]),
               "fit_trajectories": artifact["train_trajectories"],
               "validation_trajectories": sorted(val.trajectory_id.unique())}
    (args.out / f"{args.source_model}_to_{args.target_model}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "EVAL_COMPLETE", **summary}, indent=2))


def unit(_args):
    m = handwritten_matrices();
    for e in range(3):
        for s in range(3): assert np.argmax(m[e, s]) == SI[handwritten_update(STATES[s], EVENTS[e])]
    q = np.array([[1., 0., 0.]]); assert np.allclose(transition_numpy(q[0], np.array([1., 0., 0.]), m), m[0, 0])
    rows, _, _, _ = load_cache("llava")
    assert len(rows) == 250 and np.allclose(event_probs(rows, "hidden").sum(1), 1.0)
    assert fit_predict_correct(rows)["enabled"] is False
    print("UPDATER_UNIT_PASS: matrix algebra, cache schema, decoder probability normalization")


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    u = sp.add_parser("unit"); u.set_defaults(fn=unit)
    s = sp.add_parser("smoke"); s.add_argument("--model", choices=MODEL_CONFIG, required=True); s.add_argument("--trajectories", type=int, default=2); s.set_defaults(fn=smoke)
    f = sp.add_parser("fit"); f.add_argument("--model", choices=MODEL_CONFIG, required=True); f.add_argument("--out", type=Path, default=DEFAULT_OUT); f.add_argument("--event-source", choices=("hidden", "native"), default="hidden"); f.add_argument("--steps", type=int, default=1200); f.add_argument("--lr", type=float, default=.08); f.add_argument("--seed", type=int, default=20260918); f.set_defaults(fn=fit)
    e = sp.add_parser("evaluate"); e.add_argument("--source-model", choices=MODEL_CONFIG, required=True); e.add_argument("--target-model", choices=MODEL_CONFIG, required=True); e.add_argument("--artifact", type=Path, required=True); e.add_argument("--out", type=Path, default=DEFAULT_OUT); e.add_argument("--event-source", choices=("hidden", "native"), default="hidden"); e.add_argument("--temperature", type=float, default=1.0); e.set_defaults(fn=evaluate)
    c = sp.add_parser("calibrate-smoke"); c.add_argument("--model", choices=MODEL_CONFIG, required=True); c.add_argument("--trajectories", type=int, default=2); c.add_argument("--seed", type=int, default=20260918); c.set_defaults(fn=calibration_smoke)
    x = sp.add_parser("cross-smoke"); x.add_argument("--trajectories", type=int, default=2); x.add_argument("--seed", type=int, default=20260918); x.add_argument("--out", type=Path, default=DEFAULT_OUT); x.set_defaults(fn=cross_smoke)
    k = sp.add_parser("curve"); k.add_argument("--model", choices=MODEL_CONFIG, required=True); k.add_argument("--out", type=Path, default=DEFAULT_OUT); k.add_argument("--trajectory-counts", nargs="+", type=int, default=[1, 2, 5, 10, 20, 30]); k.add_argument("--repeats", type=int, default=5); k.add_argument("--seed", type=int, default=20260918); k.set_defaults(fn=curve)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__": main()
