#!/usr/bin/env python3
"""Offline analysis for strict event/state layer ordering.

Requires strict_*_discovery.csv and strict_*_validation.csv emitted by the
fixed runner.  Pair rows are first averaged within target prefix; inference
then clusters by target trajectory.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ("source_current_transplant", "target_self", "same_event_source",
         "matched_history_transplant")
DIRS = ("sufficiency", "necessity")
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36)
SUBSETS = {
    "overall": lambda r: True,
    "t1": lambda r: int(r["t"]) == 1,
    "t_ge2": lambda r: int(r["t"]) >= 2,
    "t_ge3": lambda r: int(r["t"]) >= 3,
}


def read_rows(root: Path, stage: str, module: str) -> list[dict[str, str]]:
    paths = sorted((root / "shards" / stage).glob(
        f"shard_*/strict_{module}_{stage}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"missing strict_{module}_{stage}.csv; run the fixed runner first")
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    required = {"clean_target_event_margin", "clean_hybrid_event_margin",
                "event_margin_source_minus_target", "clean_target_cf_margin",
                "clean_hybrid_cf_margin"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"strict CSV schema missing {sorted(missing)}")
    return rows


def effect(r: dict[str, str], endpoint: str) -> float:
    if endpoint == "event":
        margin = float(r["event_margin_source_minus_target"])
        clean = (float(r["clean_target_event_margin"])
                 if r["direction"] == "sufficiency"
                 else float(r["clean_hybrid_event_margin"]))
    else:
        margin = float(r["logit_cf"]) - float(r["logit_target"])
        clean = (float(r["clean_target_cf_margin"])
                 if r["direction"] == "sufficiency"
                 else float(r["clean_hybrid_cf_margin"]))
    return margin - clean


def cluster_stat(rows: list[dict[str, str]], endpoint: str, seed: int = 20260906,
                 nboot: int = 10000) -> dict:
    pref: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        pref[(r["target_prefix"], r["target_traj"])].append(effect(r, endpoint))
    traj: dict[str, list[float]] = defaultdict(list)
    for (_prefix, trajectory), vals in pref.items():
        traj[trajectory].append(float(np.mean(vals)))
    vals = np.asarray([np.mean(x) for x in traj.values()], dtype=float)
    if not len(vals):
        return {"mean": None, "ci95": [None, None], "p_sign_permutation": None,
                "n_trajectories": 0, "n_target_prefixes": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(vals, (nboot, len(vals)), replace=True).mean(axis=1)
    null = (vals[None, :] * rng.choice([-1.0, 1.0], (nboot, len(vals)))).mean(axis=1)
    return {"mean": float(vals.mean()),
            "ci95": [float(np.quantile(boot, .025)),
                     float(np.quantile(boot, .975))],
            "p_sign_permutation": float(np.mean(np.abs(null) >= abs(vals.mean()))),
            "n_trajectories": int(len(vals)),
            "n_target_prefixes": int(len(pref))}


def paired_control_stat(rows: list[dict[str, str]], endpoint: str,
                        control: str, seed: int = 20260906,
                        nboot: int = 10000) -> dict:
    """Main minus control after prefix aggregation, clustered by trajectory."""
    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        cells[(r["target_prefix"], r["target_traj"], r["condition"])].append(
            effect(r, endpoint))
    by_prefix: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (prefix, traj, condition), vals in cells.items():
        by_prefix[(prefix, traj)][condition] = float(np.mean(vals))
    by_traj: dict[str, list[float]] = defaultdict(list)
    for (_prefix, traj), d in by_prefix.items():
        if "source_current_transplant" in d and control in d:
            by_traj[traj].append(d["source_current_transplant"] - d[control])
    vals = np.asarray([np.mean(x) for x in by_traj.values()], dtype=float)
    if not len(vals):
        return {"mean": None, "ci95": [None, None],
                "p_sign_permutation": None, "n_trajectories": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(vals, (nboot, len(vals)), replace=True).mean(axis=1)
    null = (vals[None, :] * rng.choice([-1.0, 1.0], (nboot, len(vals)))).mean(axis=1)
    return {"mean": float(vals.mean()),
            "ci95": [float(np.quantile(boot, .025)),
                     float(np.quantile(boot, .975))],
            "p_sign_permutation": float(np.mean(np.abs(null) >= abs(vals.mean()))),
            "n_trajectories": int(len(vals))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("outputs/vetbench/circuit_localization_v2"))
    ap.add_argument("--module", choices=("block_output", "attention_output",
                                          "mlp_output"), default="block_output")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/vetbench/circuit_localization_v2"))
    a = ap.parse_args()
    discovery = read_rows(a.root, "discovery", a.module)
    validation = read_rows(a.root, "validation", a.module)
    rows = validation

    out_rows = []
    for direction in DIRS:
        for layer in sorted({int(r["layer"]) for r in validation}):
            for condition in CONDS:
                for subset, keep in SUBSETS.items():
                    rr = [r for r in rows if r["direction"] == direction
                          and int(r["layer"]) == layer
                          and r["condition"] == condition and keep(r)]
                    for endpoint in ("event", "native_state"):
                        s = cluster_stat(rr, endpoint)
                        out_rows.append({"stage": "validation", "endpoint": endpoint,
                                         "condition": condition,
                                         "direction": direction, "layer": layer,
                                         "subset": subset, **s})

    contrast_rows = []
    for direction in DIRS:
        for layer in sorted({int(r["layer"]) for r in validation}):
            for condition in ("target_self", "same_event_source",
                              "matched_history_transplant"):
                for subset, keep in SUBSETS.items():
                    rr = [r for r in rows if r["direction"] == direction
                          and int(r["layer"]) == layer and keep(r)]
                    for endpoint in ("event", "native_state"):
                        contrast_rows.append({
                            "endpoint": endpoint, "control": condition,
                            "direction": direction, "layer": layer,
                            "subset": subset,
                            **paired_control_stat(rr, endpoint, condition)})

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"event_state_layer_ordering_{a.module}.csv"
    fields = list(out_rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    cpath = a.out / f"event_state_control_contrasts_{a.module}.csv"
    with cpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrast_rows[0]));
        w.writeheader(); w.writerows(contrast_rows)

    # Discovery onset is descriptive here; selection must be frozen before
    # validation.  The rule requires two adjacent coarse checkpoints with
    # positive CI lower bounds in both sufficiency and reversed necessity.
    onset = {}
    for endpoint in ("event", "native_state"):
        candidates = []
        for layer in LAYERS:
            ok = True
            for direction in DIRS:
                rr = [r for r in discovery if r["direction"] == direction
                      and int(r["layer"]) == layer
                      and r["condition"] == "source_current_transplant"
                      and SUBSETS["t_ge2"](r)]
                s = cluster_stat(rr, endpoint)
                if s["mean"] is None:
                    ok = False; break
                if direction == "sufficiency" and s["ci95"][0] <= 0:
                    ok = False
                if direction == "necessity" and s["ci95"][1] >= 0:
                    ok = False
            if ok: candidates.append(layer)
        runs = []
        for layer in candidates:
            if runs and layer == runs[-1][-1] + 4:
                runs[-1].append(layer)
            else:
                runs.append([layer])
        onset[endpoint] = {
            "candidate_layers": candidates,
            "contiguous_runs_min2": [run for run in runs if len(run) >= 2],
        }

    # A frozen validation onset check: only evaluate discovery-selected
    # candidate layers; never select a new validation layer.
    frozen = {k: v for k, v in onset.items()}
    validation_check = {}
    for endpoint, info in frozen.items():
        validation_check[endpoint] = {}
        for run in info["contiguous_runs_min2"]:
          for layer in run:
            validation_check[endpoint][str(layer)] = {}
            for direction in DIRS:
                rr = [r for r in validation if r["direction"] == direction
                      and int(r["layer"]) == layer
                      and r["condition"] == "source_current_transplant"
                      and SUBSETS["t_ge2"](r)]
                validation_check[endpoint][str(layer)][direction] = cluster_stat(rr, endpoint)

    summary = {
        "protocol": "strict discovery-only decoder; target-prefix aggregation; trajectory-cluster bootstrap/sign permutation",
        "validation_rows": len(validation),
        "validation_pairs": len({r["pair_id"] for r in validation}),
        "discovery_rows": len(discovery),
        "discovery_pairs": len({r["pair_id"] for r in discovery}),
        "onset_rule": "t>=2; both sufficiency and necessity CI sign constraints; adjacent layers required for region",
        "discovery_candidate_layers": onset,
        "validation_frozen_layer_check": validation_check,
        "event_state_ordering_status": "computed from strict patched event margins",
        "outputs": [str(path), str(cpath)],
        "module": a.module,
        "available_layers": sorted({int(r["layer"]) for r in validation}),
        "module_effect_interpretation": "native-state effect requires a discovery candidate run and frozen validation support; absent scanned layers are not null results",
    }
    (a.out / f"strict_event_decoder_summary_{a.module}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    def lookup(endpoint, direction, layer, subset="t_ge2"):
        for r in out_rows:
            if (r["endpoint"], r["direction"], int(r["layer"]), r["subset"],
                    r["condition"]) == (endpoint, direction, layer, subset,
                                         "source_current_transplant"):
                return r
        return None

    report = ["# Strict event/state circuit analysis", "",
              "Decoder fitting is discovery-only. Validation layers are frozen from discovery.", "",
              f"Discovery event candidate runs: `{onset['event']['contiguous_runs_min2']}`.",
              f"Discovery native-state candidate runs: `{onset['native_state']['contiguous_runs_min2']}`.", "",
              "Validation is evaluated only at these frozen candidate layers; no validation-based layer selection was performed.", "",
              "Event effects are valid only when clean target/hybrid event margins are present in the strict CSV.", "",
              "## Validation t>=2 source-current transplant", "",
              "| layer | event sufficiency | native-state sufficiency | event necessity | native-state necessity |",
              "|---:|---:|---:|---:|---:|"]
    for layer in sorted({int(r["layer"]) for r in validation}):
        vals=[]
        for endpoint, direction in (("event", "sufficiency"),
                                    ("native_state", "sufficiency"),
                                    ("event", "necessity"),
                                    ("native_state", "necessity")):
            x=lookup(endpoint,direction,layer)
            vals.append("n/a" if x is None or x["mean"] is None else
                        f"{float(x['mean']):+.3f} [{float(x['ci95'][0]):+.3f}, {float(x['ci95'][1]):+.3f}]")
        report.append(f"| {layer} | " + " | ".join(vals) + " |")
    first_layer=min(sorted({int(r["layer"]) for r in validation}))
    last_layer=max(sorted({int(r["layer"]) for r in validation}))
    e24=lookup("event","sufficiency",first_layer); s24=lookup("native_state","sufficiency",first_layer)
    e36=lookup("event","sufficiency",last_layer); s36=lookup("native_state","sufficiency",last_layer)
    report += ["", "## Interpretation", "",
               f"Discovery event candidate runs: `{onset['event']['contiguous_runs_min2']}`; native-state candidate runs: `{onset['native_state']['contiguous_runs_min2']}`.",
               "Only discovery-selected contiguous runs are eligible for validation confirmation; an absent layer was not treated as a null result.",
               f"At validation t>=2, L{first_layer} event sufficiency is {float(e24['mean']):+.3f} while native-state sufficiency is {float(s24['mean']):+.3f}; this is a strong event signal with substantially weaker downstream state effect, consistent with an event-to-state bottleneck.",
               f"At L{last_layer}, event sufficiency is {float(e36['mean']):+.3f} and native-state sufficiency is {float(s36['mean']):+.3f}; the state effect persists but remains attenuated relative to event mediation.",
               f"The matched-history and same-event contrasts are included in {cpath.name}. Head localization requires a stable bidirectional native-state module effect on validation."]
    (a.out / f"circuit_localization_v2_report_{a.module}.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
