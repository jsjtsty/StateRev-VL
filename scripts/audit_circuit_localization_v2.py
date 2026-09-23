#!/usr/bin/env python3
"""Audit whether existing circuit artifacts can support strict v2 analysis.

This is deliberately offline.  It never imports torch/model code and never
starts a model forward.  It distinguishes reusable clean activations from
the patched activations required for strict event mediation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def npz_shape(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    z = np.load(path, mmap_mode="r")
    shapes = {k: list(z[k].shape) for k in z.files[:3]}
    return {"exists": True, "samples": len(z.files), "example_shapes": shapes}


def csv_columns(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        first = next(r, None)
        return {"exists": True, "columns": r.fieldnames or [],
                "has_patch_activation": bool(first and any(
                    "activation" in k.lower() or "hidden" in k.lower()
                    for k in first))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/vetbench/circuit_localization_v2"))
    ap.add_argument("--v1", type=Path,
                    default=Path("outputs/vetbench/circuit_localization_v1"))
    ap.add_argument("--source", type=Path,
                    default=Path("outputs/vetbench/mechanism_gate_final"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    hidden = {p.stem: npz_shape(p) for p in sorted(a.source.glob("hidden_*.npz"))}
    shard_csvs = sorted(a.v1.glob("shards/*/shard_*/*.csv"))
    cols = {str(p): csv_columns(p) for p in shard_csvs}
    split_path = a.v1 / "discovery_validation_split.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else None

    # The old runner records event probabilities, not the patched final hidden
    # used by a frozen decoder.  In addition, its event endpoint is computed
    # from va[35], the clean hybrid final-layer capture, for every patch layer.
    missing = [
        "per-layer patched final-token activations for sufficiency and necessity",
        "attention-output and MLP-output patched activations in validated layers",
        "a frozen discovery-only event decoder artifact (scaler/PCA/C/classes)",
    ]
    result = {
        "status": "INSUFFICIENT_FOR_STRICT_V2_OFFLINE_ANALYSIS",
        "model_forward_started": False,
        "split": split,
        "clean_hidden_artifacts": hidden,
        "circuit_shard_csv_count": len(shard_csvs),
        "circuit_shard_columns": cols,
        "reusable_offline": [
            "discovery-only decoder can be fit from baseline hidden_*.npz",
            "clean baseline/hybrid native layer profiles already in v1 CSVs",
        ],
        "missing_artifacts": missing,
        "leakage_findings": [
            "run_circuit_localization.py fits event_decoder from all 250 baseline rows",
            "event endpoint uses va[35] for every patch layer, so it is not a patched event endpoint",
        ],
        "required_minimal_rerun": {
            "discovery": "rerun fixed decoder and selected patch layers on discovery pairs",
            "validation": "freeze decoder and selected region, then rerun all validation pairs",
            "module": "capture selected-layer attention/MLP outputs online and emit event/native endpoints",
        },
    }
    (a.out / "strict_event_decoder_summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "model_forward_started": False,
        "hidden_files": len(hidden),
        "circuit_csvs": len(shard_csvs),
        "missing_artifacts": missing,
        "output": str(a.out / "strict_event_decoder_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
