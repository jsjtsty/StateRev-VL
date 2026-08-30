from __future__ import annotations

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
    run_name_from_member,
    state_snapshot,
)


ARCHIVE_SUBDIR = "theory_of_space_assets"
INDEX_PATH = DEFAULT_PROCESSED_DIR / "theory_of_space_state_revision.jsonl"
MAINTAIN_PATH = DEFAULT_PROCESSED_DIR / "theory_of_space_state_maintain.jsonl"


def object_by_id(objects: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(obj["object_id"]): obj for obj in objects if "object_id" in obj}


def object_state(obj: dict[str, Any]) -> dict[str, Any]:
    return state_snapshot(obj)


def load_run_members(layout) -> list[str]:
    if layout.kind == "zip":
        from zipfile import ZipFile

        with ZipFile(layout.path) as zf:
            return [n for n in zf.namelist() if n.endswith("meta_data.json")]
    return [str(p.relative_to(layout.path)) for p in layout.path.rglob("meta_data.json")]


def make_record(
    *,
    sample_id: str,
    scene_id: str,
    object_id: int,
    object_name: str,
    change_type: str,
    old_image: Path,
    new_image: Path,
    old_state: dict[str, Any],
    new_state: dict[str, Any],
    source_metadata: dict[str, Any],
    valid: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "object_id": object_id,
        "object_name": object_name,
        "change_type": change_type,
        "old_image": str(old_image),
        "new_image": str(new_image),
        "old_state": old_state,
        "new_state": new_state,
        "source_metadata": source_metadata,
        "valid": valid,
    }
    if extra:
        record.update(extra)
    return record


def main() -> None:
    layout = locate_theory_of_space_source()
    DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_PROCESSED_DIR / ARCHIVE_SUBDIR).mkdir(parents=True, exist_ok=True)

    revision_records: list[dict[str, Any]] = []
    maintain_records: list[dict[str, Any]] = []
    image_source_counter: Counter[str] = Counter()

    for meta_member in sorted(load_run_members(layout)):
        run_name = run_name_from_member(meta_member)
        meta = load_run_json(layout, run_name, "meta_data.json")
        fb = load_run_json(layout, run_name, "falsebelief_exp.json")
        objects = object_by_id(meta.get("objects", []))
        old_image, new_image, image_source = extract_canonical_pair(
            layout,
            run_name,
            DEFAULT_PROCESSED_DIR / ARCHIVE_SUBDIR,
        )
        image_source_counter[image_source] += 1

        changed_ids: set[int] = set()
        for change in fb.get("_fb_changes", []):
            object_id = change.get("object_id")
            if object_id is not None:
                changed_ids.add(int(object_id))

        for change in fb.get("_fb_changes", []):
            object_id = change.get("object_id")
            if object_id is None or bool(change.get("ori")) or not bool(change.get("pos")):
                continue

            object_id = int(object_id)
            obj = objects.get(object_id)
            if not obj:
                continue

            old_state = {
                "position": change.get("pos_from"),
                "orientation": change.get("orientation_from"),
                "yaw": change.get("yaw_from"),
            }
            new_state = {
                "position": change.get("pos_to"),
                "orientation": change.get("orientation_from"),
                "yaw": change.get("yaw_from"),
            }
            if old_state["position"] == new_state["position"]:
                continue

            sample_id = f"{run_name}_position_change_{object_id}"
            revision_records.append(
                make_record(
                    sample_id=sample_id,
                    scene_id=run_name,
                    object_id=object_id,
                    object_name=str(change.get("name") or obj.get("name") or ""),
                    change_type="position_change",
                    old_image=old_image,
                    new_image=new_image,
                    old_state=old_state,
                    new_state=new_state,
                    source_metadata={
                        "meta_data": member_path(run_name, "meta_data.json"),
                        "falsebelief_exp": member_path(run_name, "falsebelief_exp.json"),
                        "image_source": image_source,
                    },
                    valid=bool(object_id and obj.get("name")),
                    extra={
                        "object_label": obj.get("label"),
                        "source_object": object_state(obj),
                    },
                )
            )

        for object_id, obj in objects.items():
            if object_id in changed_ids:
                continue
            current_state = object_state(obj)
            maintain_records.append(
                make_record(
                    sample_id=f"{run_name}_maintain_{object_id}",
                    scene_id=run_name,
                    object_id=object_id,
                    object_name=str(obj.get("name") or ""),
                    change_type="no_change",
                    old_image=old_image,
                    new_image=new_image,
                    old_state=current_state,
                    new_state=current_state,
                    source_metadata={
                        "meta_data": member_path(run_name, "meta_data.json"),
                        "falsebelief_exp": member_path(run_name, "falsebelief_exp.json"),
                        "image_source": image_source,
                    },
                    valid=bool(obj.get("name")),
                    extra={"object_label": obj.get("label")},
                )
            )

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for record in revision_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(MAINTAIN_PATH, "w", encoding="utf-8") as f:
        for record in maintain_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote revision index: {INDEX_PATH} ({len(revision_records)} records)")
    print(f"Wrote maintain index: {MAINTAIN_PATH} ({len(maintain_records)} records)")
    print(f"Canonical image sources: {dict(image_source_counter)}")


if __name__ == "__main__":
    main()

