from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR


INDEX_PATH = Path("data/processed/theory_of_space_state_revision.jsonl")
MAINTAIN_PATH = Path("data/processed/theory_of_space_state_maintain.jsonl")
OUT_DIR = DEFAULT_OUTPUT_DIR / "data_inspection"
JSON_OUT = OUT_DIR / "dataset_validation.json"
TXT_OUT = OUT_DIR / "dataset_validation.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_record(record: dict[str, Any]) -> str:
    return (
        f"{record.get('sample_id')} | {record.get('scene_id')} | "
        f"{record.get('object_name')} ({record.get('object_id')}) | "
        f"{record.get('change_type')} | "
        f"{record.get('old_state')} -> {record.get('new_state')}"
    )


def main() -> None:
    revision = load_jsonl(INDEX_PATH)
    maintain = load_jsonl(MAINTAIN_PATH)
    all_records = revision + maintain

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_images = 0
    empty_object_fields = 0
    revision_state_violations = 0
    maintain_state_violations = 0
    same_image_pairs = 0
    signatures: Counter[tuple[Any, ...]] = Counter()
    change_types: Counter[str] = Counter()
    scene_ids: set[str] = set()
    object_ids: set[Any] = set()
    issues: list[str] = []

    for record in all_records:
        scene_ids.add(str(record.get("scene_id")))
        object_ids.add(record.get("object_id"))
        change_types[str(record.get("change_type"))] += 1

        old_image = Path(str(record.get("old_image", "")))
        new_image = Path(str(record.get("new_image", "")))
        if not old_image.exists() or not new_image.exists():
            missing_images += 1
            issues.append(f"Missing image: {record.get('sample_id')}")
        if old_image == new_image:
            same_image_pairs += 1
        if not record.get("object_name"):
            empty_object_fields += 1
            issues.append(f"Empty object name: {record.get('sample_id')}")

        signature = (
            record.get("scene_id"),
            record.get("object_id"),
            record.get("change_type"),
            json.dumps(record.get("old_state"), sort_keys=True, ensure_ascii=False),
            json.dumps(record.get("new_state"), sort_keys=True, ensure_ascii=False),
            str(old_image),
            str(new_image),
        )
        signatures[signature] += 1

        if record.get("change_type") == "position_change":
            if record.get("old_state") == record.get("new_state"):
                revision_state_violations += 1
                issues.append(f"Revision state not changed: {record.get('sample_id')}")
        elif record.get("change_type") == "no_change":
            if record.get("old_state") != record.get("new_state"):
                maintain_state_violations += 1
                issues.append(f"Maintain state changed: {record.get('sample_id')}")

    duplicate_count = sum(count - 1 for count in signatures.values() if count > 1)

    def sample_lines(records: list[dict[str, Any]], seed: int) -> list[str]:
        if not records:
            return []
        rng = random.Random(seed)
        picks = records if len(records) <= 10 else rng.sample(records, 10)
        return [summarize_record(rec) for rec in picks]

    stats: dict[str, Any] = {
        "revision_count": len(revision),
        "maintain_count": len(maintain),
        "total_count": len(all_records),
        "missing_image_count": missing_images,
        "empty_object_name_count": empty_object_fields,
        "revision_state_violation_count": revision_state_violations,
        "maintain_state_violation_count": maintain_state_violations,
        "same_image_pair_count": same_image_pairs,
        "duplicate_count": duplicate_count,
        "duplicate_rate": (duplicate_count / len(all_records)) if all_records else 0.0,
        "change_type_counts": dict(change_types),
        "scene_count": len(scene_ids),
        "object_count": len({obj for obj in object_ids if obj is not None}),
        "revision_samples": sample_lines(revision, 13),
        "maintain_samples": sample_lines(maintain, 37),
        "issues": issues[:200],
    }

    JSON_OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"Revision samples: {len(revision)}")
    lines.append(f"Maintain samples: {len(maintain)}")
    lines.append(f"Total samples: {len(all_records)}")
    lines.append(f"Missing images: {missing_images}")
    lines.append(f"Empty object names: {empty_object_fields}")
    lines.append(f"Revision state violations: {revision_state_violations}")
    lines.append(f"Maintain state violations: {maintain_state_violations}")
    lines.append(f"Same image pairs: {same_image_pairs}")
    lines.append(f"Duplicate count: {duplicate_count}")
    lines.append(f"Duplicate rate: {duplicate_count / len(all_records) if all_records else 0:.6f}")
    lines.append(f"Scene count: {len(scene_ids)}")
    lines.append(f"Object count: {len({obj for obj in object_ids if obj is not None})}")
    lines.append(f"Change type counts: {dict(change_types)}")
    lines.append("")
    lines.append("Revision sample summaries:")
    lines.extend(f"  - {line}" for line in stats["revision_samples"])
    lines.append("")
    lines.append("Maintain sample summaries:")
    lines.extend(f"  - {line}" for line in stats["maintain_samples"])
    lines.append("")
    lines.append("Issues:")
    lines.extend(f"  - {item}" for item in (issues[:50] or ["none"]))

    TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved JSON to {JSON_OUT}")
    print(f"Saved text to {TXT_OUT}")


if __name__ == "__main__":
    main()

