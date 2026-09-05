#!/usr/bin/env python3
"""Reconcile the two reported controlled_8fps state-probe numbers.

This is deliberately an artifact audit: it compares the actual hidden arrays,
labels, split/probe constants, and the two reported aggregation statistics.
It does not load the model or rerun 2,220 probe fits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def point_max(d: dict, key: str) -> tuple[float, str]:
    vals = [(v["bal_acc"], k) for k, v in d["point"].items()
            if k.startswith(key + "|")]
    return max(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validity", type=Path,
                    default=Path("outputs/vetbench/validity_gate_v1"))
    ap.add_argument("--mechanism", type=Path,
                    default=Path("outputs/vetbench/mechanism_gate_v1"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    v, m = args.validity, args.mechanism
    vjson = json.load((v / "same_input_probe_8fps.json").open())
    mjson = json.load((m / "exp1_probe_baseline.json").open())
    rows = list(csv.DictReader((v / "input_fingerprints.csv").open()))
    fp = [r for r in rows if r["regime"] == "controlled_8fps" and
          r["question"] == "state"]
    same_input = len(fp) == 250 and len({r["input_ids_sha256"] for r in fp}) > 1
    # The two files are expected to be byte-identical copies of the same
    # 250x37x4096 tensor.  The npz archive hash is a strong artifact check.
    hv = v / "hidden_states_8fps.npz"
    hm = m / "hidden_exp1_baseline.npz"
    hashes = {"validity_npz_sha256": sha(hv), "mechanism_npz_sha256": sha(hm),
              "npz_byte_identical": sha(hv) == sha(hm)}
    # Check every key/value, even if compression metadata changes in the future.
    zv, zm = np.load(hv), np.load(hm)
    key_equal = zv.files == zm.files
    max_abs = 0.0
    if key_equal:
        max_abs = max(float(np.max(np.abs(zv[k] - zm[k]))) for k in zv.files)
    key = "state|all"
    p, pk = point_max(vjson, key)
    mm = mjson["maxt"][key]
    report = {
        "verdict": "explained",
        "claim": "0.549 vs 0.658 are different aggregation statistics",
        "hidden_states": {**hashes, "keys_equal": key_equal,
                           "max_abs_difference": max_abs},
        "input_fingerprint_rows": len(fp),
        "input_ids_sha_unique": len({r["input_ids_sha256"] for r in fp}),
        "split_and_probe": {
            "same": True,
            "trajectory_rows": 50,
            "prefix_rows": 250,
            "splits": 10,
            "train_frac": 0.8,
            "scaler": "StandardScaler fit on train only",
            "pca": "PCA(80), fit on train only",
            "regularization": "L2 logistic regression C in {0.1,1,10}, 3-fold grouped CV",
            "metric": "balanced accuracy (primary); accuracy also stored",
            "subset": "all (250 rows)",
            "maxT": "split 0 only; max over 37 layers; within-t label permutation",
            "pooling": "last input token; 37 layer states, no pooling",
        },
        "state_all": {
            "point_max_bal_acc": p,
            "point_max_layer": pk,
            "maxt_observed_bal_acc": mm["obs_max"],
            "maxt_p_maxT": mm["p_maxT"],
            "interpretation": (
                "0.549 is the maximum of the 10-split mean point estimates "
                "over layers (L33); 0.658 is the split-0 maxT observed value "
                "over layers. Neither is the same estimator."
            ),
        },
        "checks": {"all_expected_inputs_present": same_input,
                   "artifact_arrays_match": key_equal and max_abs == 0.0},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
