from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "dataset" / "Theory-of-Space" / "tos_dataset_0127_3room_100runs.zip"
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "theory_of_space"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3-VL-8B-Instruct"
ZIP_ROOT = "tos_dataset_0127_3room_100runs"


@dataclass(frozen=True)
class SourceLayout:
    kind: str
    path: Path


def project_root() -> Path:
    return ROOT


def candidate_theory_of_space_sources() -> list[Path]:
    candidates = [
        DEFAULT_ARCHIVE,
        DEFAULT_RAW_DIR,
        ROOT / "dataset" / "Theory-of-Space",
    ]
    return [p for p in candidates if p.exists()]


def locate_theory_of_space_source() -> SourceLayout:
    for path in candidate_theory_of_space_sources():
        if path.is_file() and path.suffix == ".zip":
            return SourceLayout(kind="zip", path=path)
        if path.is_dir():
            return SourceLayout(kind="dir", path=path)
    raise FileNotFoundError(
        "Could not locate Theory of Space source in data/raw/theory_of_space, "
        "dataset/Theory-of-Space, or the expected zip archive."
    )


def list_source_members(layout: SourceLayout) -> list[str]:
    if layout.kind == "zip":
        with ZipFile(layout.path) as zf:
            return zf.namelist()
    return [str(p.relative_to(layout.path)) for p in layout.path.rglob("*") if p.is_file()]


def _member_path(layout: SourceLayout, member: str | Path) -> str:
    member_str = str(member).replace("\\", "/")
    if layout.kind == "zip":
        if member_str.startswith(ZIP_ROOT + "/"):
            return member_str
        return f"{ZIP_ROOT}/{member_str.lstrip('/')}"
    return member_str


def read_json_source(layout: SourceLayout, member: str | Path) -> Any:
    if layout.kind == "zip":
        with ZipFile(layout.path) as zf:
            return json.loads(zf.read(_member_path(layout, member)).decode("utf-8"))
    with open(layout.path / Path(member), "r", encoding="utf-8") as f:
        return json.load(f)


def source_exists(layout: SourceLayout, member: str | Path) -> bool:
    if layout.kind == "zip":
        with ZipFile(layout.path) as zf:
            return _member_path(layout, member) in set(zf.namelist())
    return (layout.path / Path(member)).exists()


def run_name_from_member(member: str) -> str:
    parts = Path(member).parts
    for part in parts:
        if re.fullmatch(r"run\d{2}", part):
            return part
    raise ValueError(f"Could not infer run name from {member}")


def run_member(run_name: str, filename: str) -> str:
    return f"{ZIP_ROOT}/{run_name}/{filename}"


def member_path(run_name: str, filename: str) -> str:
    return run_member(run_name, filename)


def load_run_json(layout: SourceLayout, run_name: str, filename: str) -> Any:
    return read_json_source(layout, run_member(run_name, filename))


def state_snapshot(obj: dict[str, Any]) -> dict[str, Any]:
    pos = obj.get("pos") or {}
    attrs = obj.get("attributes") or {}
    rot = obj.get("rot") or {}
    return {
        "position": {
            "x": pos.get("x"),
            "y": pos.get("y"),
            "z": pos.get("z"),
        },
        "orientation": attrs.get("orientation"),
        "yaw": rot.get("y"),
        "room_id": attrs.get("room_id"),
    }


def format_state(state: dict[str, Any]) -> str:
    pos = state.get("position") or {}
    return (
        f"pos=({pos.get('x')}, {pos.get('z')}), "
        f"orientation={state.get('orientation')}, yaw={state.get('yaw')}, "
        f"room={state.get('room_id')}"
    )


def extract_member(layout: SourceLayout, member: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    if layout.kind == "zip":
        with ZipFile(layout.path) as zf:
            with zf.open(member) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
        return
    src_path = layout.path / member
    output_path.write_bytes(src_path.read_bytes())


def extract_canonical_pair(layout: SourceLayout, run_name: str, target_dir: Path) -> tuple[Path, Path, str]:
    candidates = [
        ("top_down.png", "top_down_fbexp.png", "top_down"),
        ("main_cam/img_0000.png", "main_cam/img_0000_fbexp.png", "main_cam"),
    ]
    for old_name, new_name, label in candidates:
        old_member = run_member(run_name, old_name)
        new_member = run_member(run_name, new_name)
        if source_exists(layout, old_member) and source_exists(layout, new_member):
            old_path = target_dir / run_name / old_name.replace("/", "_")
            new_path = target_dir / run_name / new_name.replace("/", "_")
            extract_member(layout, old_member, old_path)
            extract_member(layout, new_member, new_path)
            return old_path, new_path, label
    raise FileNotFoundError(f"No canonical image pair found for {run_name}")
