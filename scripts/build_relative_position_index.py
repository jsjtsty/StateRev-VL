from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _theory_of_space_utils import (
    DEFAULT_PROCESSED_DIR,
    extract_canonical_pair,
    locate_theory_of_space_source,
    load_run_json,
    member_path,
)


RELATIVE_REVISION_PATH = DEFAULT_PROCESSED_DIR / "theory_of_space_relative_revision.jsonl"
RELATIVE_MAINTAIN_PATH = DEFAULT_PROCESSED_DIR / "theory_of_space_relative_maintain.jsonl"
ASSETS_SUBDIR = "theory_of_space_assets"

LABELS = ("LEFT", "RIGHT", "ABOVE", "BELOW")


def classify_relative_label(dx: float, dz: float, min_primary: float, max_secondary: float) -> str | None:
    """Image-relative position of the target w.r.t. the reference.

    World axes (verified against ``topdown_map`` in the renderer's own
    ``meta_data.json``): +x is image-right, +z is image-up (north).  dx = x_t - x_r,
    dz = z_t - z_r, so:
      - |dx| dominant  -> LEFT (dx < 0) / RIGHT (dx > 0)
      - |dz| dominant  -> BELOW (dz < 0) / ABOVE (dz > 0)
    Returns None when the relation is ambiguous (too close on both axes, i.e.
    diagonal or adjacent), so those samples are filtered out.
    """
    if abs(dx) >= min_primary and abs(dz) <= max_secondary:
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dz) >= min_primary and abs(dx) <= max_secondary:
        return "ABOVE" if dz > 0 else "BELOW"
    return None


def xz(pos: dict[str, Any]) -> tuple[float, float]:
    return float(pos["x"]), float(pos["z"])


def build_index(min_primary: float, max_secondary: float, max_refs_per_target: int) -> tuple[list[dict], list[dict]]:
    layout = locate_theory_of_space_source()
    revision: list[dict] = []
    maintain: list[dict] = []
    dropped: Counter[str] = Counter()

    for i in range(100):
        run_name = f"run{i:02d}"
        try:
            meta = load_run_json(layout, run_name, "meta_data.json")
            fb = load_run_json(layout, run_name, "falsebelief_exp.json")
        except (KeyError, FileNotFoundError):
            continue

        objects = {int(o["object_id"]): o for o in meta.get("objects", []) if "pos" in o and "object_id" in o}
        changed = {int(c["object_id"]) for c in fb.get("_fb_changes", []) if c.get("object_id") is not None}
        changed_pos = {
            int(c["object_id"])
            for c in fb.get("_fb_changes", [])
            if c.get("object_id") is not None and bool(c.get("pos"))
        }
        refs = [o for oid, o in objects.items() if oid not in changed and "door" not in str(o.get("name", "")).lower()]
        unchanged = [o for oid, o in objects.items() if oid not in changed and "door" not in str(o.get("name", "")).lower()]

        old_image, new_image, image_source = extract_canonical_pair(
            layout,
            run_name,
            DEFAULT_PROCESSED_DIR / ASSETS_SUBDIR,
        )
        source_metadata = {
            "meta_data": member_path(run_name, "meta_data.json"),
            "falsebelief_exp": member_path(run_name, "falsebelief_exp.json"),
            "image_source": image_source,
        }

        # ---- revision: changed target vs. stationary reference ----
        for change in fb.get("_fb_changes", []):
            if not bool(change.get("pos")):
                continue
            target_id = int(change["object_id"])
            old_pos = xz(change["pos_from"])
            new_pos = xz(change["pos_to"])
            used_refs = 0
            for ref in refs:
                ref_pos = xz(ref["pos"])
                old_label = classify_relative_label(old_pos[0] - ref_pos[0], old_pos[1] - ref_pos[1], min_primary, max_secondary)
                new_label = classify_relative_label(new_pos[0] - ref_pos[0], new_pos[1] - ref_pos[1], min_primary, max_secondary)
                if old_label is None or new_label is None:
                    dropped["revision_ambiguous"] += 1
                    continue
                if old_label == new_label:
                    dropped["revision_same_label"] += 1
                    continue
                revision.append(
                    {
                        "sample_id": f"{run_name}_relative_{target_id}_ref_{ref['object_id']}",
                        "scene_id": run_name,
                        "target_object_id": target_id,
                        "target_name": change.get("name") or objects[target_id].get("name"),
                        "reference_object_id": ref["object_id"],
                        "reference_name": ref.get("name"),
                        "old_image": str(old_image),
                        "new_image": str(new_image),
                        "old_label": old_label,
                        "new_label": new_label,
                        "transition": f"{old_label}->{new_label}",
                        "old_pos": {"x": old_pos[0], "z": old_pos[1]},
                        "new_pos": {"x": new_pos[0], "z": new_pos[1]},
                        "reference_pos": {"x": ref_pos[0], "z": ref_pos[1]},
                        "filter": {"min_primary": min_primary, "max_secondary": max_secondary},
                        "source_metadata": source_metadata,
                    }
                )
                used_refs += 1
                if max_refs_per_target and used_refs >= max_refs_per_target:
                    break

        # ---- maintain: unchanged target vs. unchanged reference ----
        for target in unchanged:
            target_pos = xz(target["pos"])
            used_refs = 0
            for ref in unchanged:
                if ref["object_id"] == target["object_id"]:
                    continue
                ref_pos = xz(ref["pos"])
                label = classify_relative_label(target_pos[0] - ref_pos[0], target_pos[1] - ref_pos[1], min_primary, max_secondary)
                if label is None:
                    dropped["maintain_ambiguous"] += 1
                    continue
                maintain.append(
                    {
                        "sample_id": f"{run_name}_relative_maintain_{target['object_id']}_ref_{ref['object_id']}",
                        "scene_id": run_name,
                        "target_object_id": target["object_id"],
                        "target_name": target.get("name"),
                        "reference_object_id": ref["object_id"],
                        "reference_name": ref.get("name"),
                        "old_image": str(old_image),
                        "new_image": str(new_image),
                        "label": label,
                        "old_label": label,
                        "new_label": label,
                        "target_pos": {"x": target_pos[0], "z": target_pos[1]},
                        "reference_pos": {"x": ref_pos[0], "z": ref_pos[1]},
                        "filter": {"min_primary": min_primary, "max_secondary": max_secondary},
                        "source_metadata": source_metadata,
                    }
                )
                used_refs += 1
                if max_refs_per_target and used_refs >= max_refs_per_target:
                    break

    return revision, maintain


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relative-position revision/maintain indexes for Theory of Space.")
    parser.add_argument("--min-primary", type=float, default=2.0, help="Minimum separation on the dominant axis (default: 2.0).")
    parser.add_argument("--max-secondary", type=float, default=1.0, help="Maximum separation allowed on the secondary axis (default: 1.0).")
    parser.add_argument("--max-refs-per-target", type=int, default=0, help="Cap references per target (0 = keep all).")
    args = parser.parse_args()

    if args.max_secondary >= args.min_primary:
        raise ValueError("--max-secondary must be smaller than --min-primary")

    revision, maintain = build_index(args.min_primary, args.max_secondary, args.max_refs_per_target)
    DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(revision, RELATIVE_REVISION_PATH)
    write_jsonl(maintain, RELATIVE_MAINTAIN_PATH)

    transitions = Counter(r["transition"] for r in revision)
    labels = Counter(r["label"] for r in maintain)
    print(f"Wrote revision index: {RELATIVE_REVISION_PATH} ({len(revision)} records)")
    print(f"Wrote maintain index: {RELATIVE_MAINTAIN_PATH} ({len(maintain)} records)")
    print("Revision transitions:", dict(sorted(transitions.items())))
    print("Maintain labels:", dict(sorted(labels.items())))


if __name__ == "__main__":
    main()
