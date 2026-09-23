#!/usr/bin/env python3
"""Build and audit a trajectory split with no exact visual-content overlap.

The graph has one vertex per trajectory.  An exact pixel fingerprint shared by
two trajectories makes an undirected edge, so every connected component must
remain entirely in discovery or validation.  This is intentionally stricter
than trajectory-ID disjointness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")
DEFAULT_FP = Path("outputs/vetbench/validity_gate_v1/input_fingerprints.csv")
DEFAULT_BEHAVIOR = Path("outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv")
DEFAULT_HIDDEN = Path("outputs/vetbench/hidden_state_probe/hidden_states.npz")
DEFAULT_OUT = Path("outputs/vetbench/content_disjoint_split_v1")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target_prefixes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_prefix"] = result["trajectory_id"].astype(str) + "_t" + result["t"].astype(int).astype(str)
    return result


def state_fingerprints(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"trajectory_id", "t", "regime", "question", "pixel_values_videos_sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"fingerprint schema missing {sorted(missing)}")
    frame = frame[(frame["regime"] == "controlled_8fps") & (frame["question"] == "state")].copy()
    frame = target_prefixes(frame)
    if frame["target_prefix"].duplicated().any():
        raise AssertionError("duplicate state fingerprint prefix")
    return frame.sort_values("target_prefix").reset_index(drop=True)


def components(fingerprints: pd.DataFrame) -> list[list[str]]:
    ids = sorted(fingerprints["trajectory_id"].unique())
    neighbors: dict[str, set[str]] = {trajectory: set() for trajectory in ids}
    for _, group in fingerprints.groupby("pixel_values_videos_sha256"):
        group_ids = sorted(set(group["trajectory_id"]))
        for trajectory in group_ids:
            neighbors[trajectory].update(other for other in group_ids if other != trajectory)
    result: list[list[str]] = []
    seen: set[str] = set()
    for root in ids:
        if root in seen:
            continue
        stack, component = [root], []
        seen.add(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for other in sorted(neighbors[current], reverse=True):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(sorted(component))
    return sorted(result, key=lambda x: (x[0], len(x)))


def component_stats(parts: list[list[str]], behavior: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for index, trajectories in enumerate(parts):
        rows = behavior[behavior["trajectory_id"].isin(trajectories)]
        record: dict[str, object] = {
            "component_id": index,
            "trajectory_count": len(trajectories),
            "prefix_count": len(rows),
            "trajectories": ",".join(trajectories),
        }
        for event in EVENTS:
            record[f"event_{event}"] = int((rows["gt_event"] == event).sum())
        records.append(record)
    return records


def choose_discovery(parts: list[list[str]], behavior: pd.DataFrame, target_n: int = 30) -> set[str]:
    """Exhaustively select component assignments with deterministic balance tie-break."""
    stats = component_stats(parts, behavior)
    total = Counter(behavior["gt_event"])
    candidates: list[tuple[tuple[float, tuple[str, ...]], set[str]]] = []
    for mask in range(1 << len(parts)):
        selected = [i for i in range(len(parts)) if mask & (1 << i)]
        n = sum(len(parts[i]) for i in selected)
        if n != target_n:
            continue
        counts = Counter()
        for i in selected:
            counts.update({event: int(stats[i][f"event_{event}"]) for event in EVENTS})
        # Match 60% of every event count.  Integer count deviations are more
        # meaningful than raw rate deviations for this small fixed dataset.
        score = sum((counts[event] - 0.6 * total[event]) ** 2 for event in EVENTS)
        ids = tuple(sorted(trajectory for i in selected for trajectory in parts[i]))
        candidates.append(((float(score), ids), set(ids)))
    if not candidates:
        raise AssertionError(f"no whole-component assignment contains {target_n} trajectories")
    return min(candidates, key=lambda x: x[0])[1]


def cross_split_pixel_groups(fingerprints: pd.DataFrame, discovery: set[str], validation: set[str]) -> list[dict[str, object]]:
    duplicates = []
    for fingerprint, group in fingerprints.groupby("pixel_values_videos_sha256"):
        disc = sorted(set(group.loc[group["trajectory_id"].isin(discovery), "trajectory_id"]))
        val = sorted(set(group.loc[group["trajectory_id"].isin(validation), "trajectory_id"]))
        if disc and val:
            duplicates.append({"pixel_values_videos_sha256": fingerprint, "discovery_trajectories": disc, "validation_trajectories": val})
    return duplicates


def full_hidden_duplicates(hidden_path: Path, fingerprints: pd.DataFrame, discovery: set[str], validation: set[str]) -> list[tuple[str, str]]:
    hidden = np.load(hidden_path, allow_pickle=False)
    duplicates = []
    for _, group in fingerprints.groupby("pixel_values_videos_sha256"):
        disc = group[group["trajectory_id"].isin(discovery)]
        val = group[group["trajectory_id"].isin(validation)]
        for drow in disc.itertuples():
            for vrow in val.itertuples():
                if np.array_equal(hidden[drow.target_prefix], hidden[vrow.target_prefix]):
                    duplicates.append((drow.target_prefix, vrow.target_prefix))
    return duplicates


def assert_content_disjoint(split: dict, fingerprint_path: Path = DEFAULT_FP, hidden_path: Path | None = None) -> dict:
    """Fail closed when any exact visual content crosses the declared split."""
    discovery = set(split["discovery_trajectories"])
    validation = set(split["validation_trajectories"])
    if discovery & validation:
        raise AssertionError("trajectory IDs overlap across split")
    fingerprints = state_fingerprints(fingerprint_path)
    if set(fingerprints["trajectory_id"]) != discovery | validation:
        raise AssertionError("split does not cover exactly the fingerprint trajectories")
    pixel_duplicates = cross_split_pixel_groups(fingerprints, discovery, validation)
    if pixel_duplicates:
        raise AssertionError(f"content leakage: {len(pixel_duplicates)} exact pixel fingerprints cross split")
    hidden_duplicates: list[tuple[str, str]] = []
    if hidden_path is not None:
        hidden_duplicates = full_hidden_duplicates(hidden_path, fingerprints, discovery, validation)
        if hidden_duplicates:
            raise AssertionError(f"content leakage: {len(hidden_duplicates)} full hidden tensors cross split")
    return {
        "trajectory_overlap": 0,
        "cross_split_exact_pixel_fingerprint_groups": 0,
        "cross_split_exact_full_hidden_tensor_pairs": 0,
        "fingerprint_path": str(fingerprint_path),
        "hidden_path": str(hidden_path) if hidden_path else None,
    }


def event_distribution(behavior: pd.DataFrame, trajectories: set[str]) -> dict[str, int]:
    rows = behavior[behavior["trajectory_id"].isin(trajectories)]
    return {event: int((rows["gt_event"] == event).sum()) for event in EVENTS}


def target_ids_from_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    for column in ("target_trajectory", "target_traj", "trajectory_id"):
        if column in frame:
            return set(frame[column].dropna().astype(str))
    return set()


def target_ids_from_pair_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    return {str(row["target_traj"]) for row in payload.get("validation_pairs", [])}


def old_split_scan(fingerprints: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Report experiments tied to an old split; no model results are changed."""
    configs = [
        ("circuit localization v1", root / "circuit_localization_v1" / "discovery_validation_split.json", lambda: target_ids_from_pair_manifest(root / "circuit_localization_v1" / "discovery_pair_manifest.json")),
        ("circuit localization v2", root / "circuit_localization_v2" / "discovery_validation_split.json", lambda: target_ids_from_pair_manifest(root / "circuit_localization_v2" / "discovery_pair_manifest.json")),
        ("L24 attention/module localization", root / "head_localization_l24_v1" / "discovery_validation_split.json", lambda: target_ids_from_pair_manifest(root / "head_localization_l24_v1" / "discovery_pair_manifest.json")),
        ("Head 0 localization", root / "head_localization_l24_v1" / "discovery_validation_split.json", lambda: target_ids_from_csv(root / "head_localization_l24_v1" / "validation_head_results.csv")),
        ("Head 0 rescue", root / "head_localization_l24_v1" / "discovery_validation_split.json", lambda: target_ids_from_csv(root / "head0_causal_rescue_v1" / "rescue_pair_results.csv")),
        ("path/history experiments", root / "circuit_localization_v2" / "discovery_validation_split.json", lambda: target_ids_from_csv(root / "history_readout_mediation_v1" / "history_readout_mediation_results.csv")),
        ("conditional L32", root / "circuit_localization_v2" / "discovery_validation_split.json", lambda: target_ids_from_csv(root / "conditional_l32_min_intervention_v1" / "conditional_l32_pair_results.csv")),
        ("multi-position rescue", root / "circuit_localization_v2" / "discovery_validation_split.json", lambda: target_ids_from_csv(root / "multi_position_delta_rescue_v1" / "multi_position_pair_results.csv")),
    ]
    records = []
    for experiment, split_path, target_loader in configs:
        if not split_path.exists():
            records.append({"experiment": experiment, "contaminated": "unknown", "duplicate_count": None, "affected_trajectories": "missing split", "needs_rerun": "review"})
            continue
        split = json.loads(split_path.read_text())
        discovery, validation = set(split["discovery_trajectories"]), set(split["validation_trajectories"])
        target_ids = target_loader() or validation
        if not target_ids <= validation:
            raise AssertionError(f"{experiment}: recorded targets are not validation trajectories")
        all_dupes = cross_split_pixel_groups(fingerprints, discovery, validation)
        dupes = [
            row for row in all_dupes
            if set(row["validation_trajectories"]) & target_ids
        ]
        affected = sorted({trajectory for row in dupes for trajectory in row["discovery_trajectories"] + row["validation_trajectories"]})
        records.append({
            "experiment": experiment,
            "contaminated": bool(dupes),
            "duplicate_count": len(dupes),
            "target_trajectory_count": len(target_ids),
            "source_pool": "all discovery trajectories used for selection/donor eligibility",
            "affected_trajectories": ",".join(affected),
            "needs_rerun": bool(dupes),
            "split_path": str(split_path),
        })
    return pd.DataFrame(records)


def write_report(out: Path, summary: dict, scan: pd.DataFrame) -> None:
    components = summary["components"]
    assertions = summary["assertions"]
    distribution = summary["event_distribution"]
    lines = [
        "# Content-disjoint split v1",
        "",
        "## Verdict",
        "",
        "PASS: trajectories are component-disjoint under exact controlled-8fps state pixel fingerprints.",
        "",
        "## Split",
        "",
        f"- Discovery / validation: {summary['split']['discovery_n']} / {summary['split']['validation_n']} trajectories; {summary['prefix_counts']['discovery']} / {summary['prefix_counts']['validation']} prefixes.",
        f"- Components: {components['count']}; size distribution: {components['size_distribution']}.",
        f"- Cross-split exact pixel fingerprint groups: {assertions['cross_split_exact_pixel_fingerprint_groups']}.",
        f"- Cross-split exact full hidden tensor pairs: {assertions['cross_split_exact_full_hidden_tensor_pairs']}.",
        "",
        "## Event Distribution",
        "",
        pd.DataFrame(distribution).T.to_markdown(),
        "",
        "## Historical Contamination Scan",
        "",
        scan[["experiment", "contaminated", "duplicate_count", "target_trajectory_count", "needs_rerun"]].to_markdown(index=False),
        "",
        "`duplicate_count` counts exact visual fingerprint groups that cross the recorded experiment's discovery source pool and its observed validation target scope.",
        "",
    ]
    (out / "content_disjoint_report.md").write_text("\n".join(lines))


def command_build(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    fingerprints = state_fingerprints(args.fingerprints)
    behavior = target_prefixes(pd.read_csv(args.behavior))
    if set(behavior["target_prefix"]) != set(fingerprints["target_prefix"]):
        raise AssertionError("behavior/fingerprint prefix keys differ")
    parts = components(fingerprints)
    discovery = choose_discovery(parts, behavior, args.discovery_n)
    validation = set(fingerprints["trajectory_id"]) - discovery
    split = {
        "schema_version": 1,
        "seed": args.seed,
        "unit": "trajectory connected component over exact controlled_8fps state pixel fingerprints",
        "construction": "components are inseparable; exhaustive component assignment minimizes squared event-count deviation from 60/40",
        "fingerprint_source": str(args.fingerprints),
        "fingerprint_source_sha256": file_sha256(args.fingerprints),
        "discovery_trajectories": sorted(discovery),
        "validation_trajectories": sorted(validation),
        "discovery_n": len(discovery),
        "validation_n": len(validation),
    }
    check = assert_content_disjoint(split, args.fingerprints, args.hidden_cache)
    stats = component_stats(parts, behavior)
    for row in stats:
        row["partition"] = "discovery" if set(row["trajectories"].split(",")) <= discovery else "validation"
    pd.DataFrame(stats).to_csv(out / "components.csv", index=False)
    (out / "discovery_validation_split.json").write_text(json.dumps(split, indent=2) + "\n")
    scan = old_split_scan(fingerprints, Path("outputs/vetbench"))
    scan.to_csv(out / "historical_contamination_scan.csv", index=False)
    summary = {
        "status": "PASS",
        "split": split,
        "assertions": check,
        "components": {"count": len(parts), "size_distribution": dict(sorted(Counter(map(len, parts)).items()))},
        "prefix_counts": {"discovery": len(discovery) * 5, "validation": len(validation) * 5},
        "event_distribution": {"all": event_distribution(behavior, set(fingerprints.trajectory_id)), "discovery": event_distribution(behavior, discovery), "validation": event_distribution(behavior, validation)},
        "source_sha256": {"fingerprints": file_sha256(args.fingerprints), "behavior": file_sha256(args.behavior), "hidden_cache": file_sha256(args.hidden_cache)},
        "historical_contamination_scan": scan.to_dict(orient="records"),
    }
    (out / "content_disjoint_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(out, summary, scan)
    manifest_files = sorted([*out.glob("*.json"), *out.glob("*.csv")])
    (out / "SHA256SUMS").write_text("".join(f"{file_sha256(p)}  {p.name}\n" for p in manifest_files))
    print(json.dumps({"BUILD_PASS": True, "out": str(out), "discovery": len(discovery), "validation": len(validation), "components": len(parts), **check}, indent=2))


def command_audit(args: argparse.Namespace) -> None:
    split = json.loads(args.split.read_text())
    result = assert_content_disjoint(split, args.fingerprints, args.hidden_cache)
    print(json.dumps({"CONTENT_DISJOINT_AUDIT_PASS": True, **result}, indent=2))


def command_scan(args: argparse.Namespace) -> None:
    out = args.out
    fingerprints = state_fingerprints(args.fingerprints)
    scan = old_split_scan(fingerprints, Path("outputs/vetbench"))
    scan.to_csv(out / "historical_contamination_scan.csv", index=False)
    summary_path = out / "content_disjoint_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["historical_contamination_scan"] = scan.to_dict(orient="records")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        write_report(out, summary, scan)
    manifest_files = sorted([*out.glob("*.json"), *out.glob("*.csv")])
    (out / "SHA256SUMS").write_text("".join(f"{file_sha256(p)}  {p.name}\n" for p in manifest_files))
    print(json.dumps({"SCAN_PASS": True, "out": str(out), "experiments": len(scan), "contaminated": int(scan["contaminated"].eq(True).sum())}, indent=2))


def command_smoke(args: argparse.Namespace) -> None:
    split = json.loads(args.split.read_text())
    check = assert_content_disjoint(split, args.fingerprints, args.hidden_cache)
    validation = split["validation_trajectories"]
    chosen = validation[: min(2, len(validation))]
    fingerprints = state_fingerprints(args.fingerprints)
    cache = np.load(args.hidden_cache, allow_pickle=False)
    keys = [f"{trajectory}_t{step}" for trajectory in chosen for step in range(1, 6)]
    if any(key not in cache.files or cache[key].shape != (37, 4096) for key in keys):
        raise AssertionError("smoke trajectories are absent from the expected hidden cache schema")
    print(json.dumps({"SMOKE_PASS": True, "validation_trajectories": chosen, "prefixes": keys, "hidden_shape": [37, 4096], **check}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "audit", "scan", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--fingerprints", type=Path, default=DEFAULT_FP)
        command.add_argument("--hidden-cache", type=Path, default=DEFAULT_HIDDEN)
    build = sub.choices["build"]
    build.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--discovery-n", type=int, default=30)
    build.add_argument("--seed", type=int, default=20260917)
    for name in ("audit", "smoke"):
        sub.choices[name].add_argument("--split", type=Path, default=DEFAULT_OUT / "discovery_validation_split.json")
    sub.choices["scan"].add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    {"build": command_build, "audit": command_audit, "scan": command_scan, "smoke": command_smoke}[args.command](args)


if __name__ == "__main__":
    main()
