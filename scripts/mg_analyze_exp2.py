#!/usr/bin/env python3
"""Mechanism Gate Experiment 2 - Gate B analysis (CPU).

Question: is the current state S_t only "not routed" to the last input token,
i.e. is it linearly present at OTHER tokens/positions?

For each token-position family, report the linear-decode obs_max (best layer)
and p_maxT for state / prev_state / event. Compare state decodability across
families: if state is decodable from video-window / question / other positions
at a level comparable to the last-input-token reference, S_t is present there
(a routing issue); if not, S_t is only (weakly) formed at the readout token.

Outputs: exp2_gateB_summary.json, exp2_gateB.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FAMILIES = ["final_prompt", "question_mean", "currently", "video_all",
            "video_last25", "video_window", "video_pre"]
TARGETS = ["state", "prev_state", "event"]
SUBSETS = ["all", "true_transition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/vetbench/mechanism_gate_v1")
    args = ap.parse_args()
    out = Path(args.out_dir)
    probe = json.load(open(out / "exp2_probe_summary.json"))

    summary = {}
    for fam in FAMILIES:
        if fam not in probe:
            continue
        d = probe[fam]
        entry = {}
        for tgt in TARGETS:
            for sub in SUBSETS:
                m = d["maxt"].get(f"{tgt}|{sub}")
                if m and not m.get("empty"):
                    entry[f"{tgt}|{sub}"] = {"obs_max": m["obs_max"],
                                             "p_maxT": m["p_maxT"]}
        summary[fam] = entry

    with open(out / "exp2_gateB_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(out / "exp2_gateB.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "target", "subset", "obs_max", "p_maxT"])
        for fam in FAMILIES:
            for key, v in summary.get(fam, {}).items():
                tgt, sub = key.split("|")
                w.writerow([fam, tgt, sub, f"{v['obs_max']:.4f}",
                            f"{v['p_maxT']:.4f}"])
        fh.close()

    print(f"wrote {out}/exp2_gateB_summary.json and exp2_gateB.csv")
    print("\n=== Gate B: obs_max (best-layer bal_acc) by family ===")
    for tgt, sub in [("state", "all"), ("state", "true_transition"),
                     ("prev_state", "all"), ("event", "all")]:
        key = f"{tgt}|{sub}"
        print(f"  {key}:")
        line = ""
        for fam in FAMILIES:
            v = summary.get(fam, {}).get(key)
            line += f"{fam}={v['obs_max']:.3f} " if v else f"{fam}=NA "
        print(f"    {line}")
    # state: reference (final_prompt) vs best other family
    ref = summary.get("final_prompt", {}).get("state|all", {})
    others = {fam: summary.get(fam, {}).get("state|all", {})
              for fam in FAMILIES if fam != "final_prompt"}
    best_other = max(others.items(),
                     key=lambda kv: kv[1].get("obs_max", -1)) \
        if any(others.values()) else (None, {})
    print(f"\n  state|all: final_prompt(ref)={ref.get('obs_max','NA')}  "
          f"best_other={best_other[0]}={best_other[1].get('obs_max','NA')}")


if __name__ == "__main__":
    main()
