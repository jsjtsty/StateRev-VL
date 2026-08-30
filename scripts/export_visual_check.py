from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REVISION_PATH = Path("data/processed/theory_of_space_state_revision.jsonl")
MAINTAIN_PATH = Path("data/processed/theory_of_space_state_maintain.jsonl")
OUT_DIR = Path("outputs/visual_check")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pick_samples(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) <= count:
        return records
    rng = random.Random(seed)
    return rng.sample(records, count)


def load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def render_label(text: str, width: int, font: ImageFont.ImageFont) -> Image.Image:
    padding = 10
    probe = Image.new("RGB", (width, 10), "white")
    draw = ImageDraw.Draw(probe)
    lines: list[str] = []
    for raw_line in text.split("\n"):
        lines.extend(textwrap.wrap(raw_line, width=60) or [""])
    bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=4)
    height = (bbox[3] - bbox[1]) + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((padding, padding), "\n".join(lines), fill="black", font=font, spacing=4)
    return canvas


def compose_sample(record: dict, out_path: Path) -> None:
    old_path = Path(record["old_image"])
    new_path = Path(record["new_image"])
    old_img = Image.open(old_path).convert("RGB")
    new_img = Image.open(new_path).convert("RGB")

    target_h = max(old_img.height, new_img.height)
    if old_img.height != target_h:
        old_img = old_img.resize((int(old_img.width * target_h / old_img.height), target_h))
    if new_img.height != target_h:
        new_img = new_img.resize((int(new_img.width * target_h / new_img.height), target_h))

    gap = 12
    width = old_img.width + new_img.width + gap * 3
    header_text = (
        f"sample_id: {record.get('sample_id')}\n"
        f"object_name: {record.get('object_name')}\n"
        f"old_state: {record.get('old_state')}\n"
        f"new_state: {record.get('new_state')}\n"
        f"change_type: {record.get('change_type')}"
    )
    header = render_label(header_text, width, load_font(16))
    canvas = Image.new("RGB", (width, header.height + target_h + gap * 2), "white")
    canvas.paste(header, (0, 0))
    y = header.height + gap
    canvas.paste(old_img, (gap, y))
    canvas.paste(new_img, (gap * 2 + old_img.width, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, y - 24), "old", fill="black", font=load_font(16))
    draw.text((gap * 2 + old_img.width, y - 24), "new", fill="black", font=load_font(16))
    canvas.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    revision = load_jsonl(REVISION_PATH)
    maintain = load_jsonl(MAINTAIN_PATH)

    selected: list[dict] = []
    selected.extend(pick_samples(revision, 20, 11))
    if maintain:
        selected.extend(pick_samples(maintain, 20, 23))

    for idx, record in enumerate(selected, start=1):
        out_name = f"{idx:04d}_{record['change_type']}.png"
        compose_sample(record, OUT_DIR / out_name)
        print(f"Wrote {OUT_DIR / out_name}")


if __name__ == "__main__":
    main()

