from __future__ import annotations

from pathlib import Path
from typing import Any

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR, locate_theory_of_space_source, list_source_members, read_json_source


KEYWORDS = (
    "false",
    "fbexp",
    "move",
    "moved",
    "pos",
    "position",
    "old",
    "new",
    "ori",
    "orientation",
    "image",
    "img",
    "camera",
    "cam",
)


def matches_keyword(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in KEYWORDS)


def collect_candidate_fields(obj: Any, prefix: str = "", depth: int = 0, limit: int = 24) -> list[tuple[str, Any]]:
    results: list[tuple[str, Any]] = []
    if limit <= 0 or depth >= 4:
        return results
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if matches_keyword(str(key)):
                results.append((path, value))
            if isinstance(value, (dict, list)):
                results.extend(collect_candidate_fields(value, path, depth + 1, limit - len(results)))
            if len(results) >= limit:
                break
    elif isinstance(obj, list):
        for idx, item in enumerate(obj[:5]):
            path = f"{prefix}[{idx}]"
            if isinstance(item, (dict, list)):
                results.extend(collect_candidate_fields(item, path, depth + 1, limit - len(results)))
            if len(results) >= limit:
                break
    return results[:limit]


def summarize_json(data: Any) -> list[str]:
    lines: list[str] = []
    lines.append(f"Top-level type: {type(data).__name__}")
    if isinstance(data, dict):
        lines.append(f"Top-level keys: {list(data.keys())[:40]}")
        for key, value in list(data.items())[:2]:
            lines.append(f"  key={key!r} type={type(value).__name__} preview={repr(value)[:900]}")
    elif isinstance(data, list):
        lines.append(f"Top-level length: {len(data)}")
        for idx, value in enumerate(data[:2]):
            lines.append(f"  item[{idx}] type={type(value).__name__} preview={repr(value)[:900]}")
    for path, value in collect_candidate_fields(data):
        lines.append(f"  candidate {path}: {repr(value)[:900]}")
    return lines


def main() -> None:
    layout = locate_theory_of_space_source()
    members = list_source_members(layout)
    json_members = [m for m in members if m.lower().endswith(".json")]
    image_members = [m for m in members if m.lower().endswith((".png", ".jpg", ".jpeg"))]
    fb_members = [m for m in members if m.lower().endswith("falsebelief_exp.json")]

    out_dir = DEFAULT_OUTPUT_DIR / "data_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "theory_of_space_structure.txt"

    lines: list[str] = []
    lines.append(f"Source kind: {layout.kind}")
    lines.append(f"Source path: {layout.path}")
    lines.append(f"JSON files total: {len(json_members)}")
    lines.append(f"Image files total: {len(image_members)}")
    lines.append(f"falsebelief_exp.json total: {len(fb_members)}")

    for member in fb_members[:5]:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"File: {member}")
        data = read_json_source(layout, member)
        lines.extend(summarize_json(data))

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

